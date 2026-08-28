#!/usr/bin/env python3
"""Dimension 2 generation benchmarks: GSM8K (8-shot CoT, EM) and HumanEval (pass@1).

These are generation/execution tasks that the log-likelihood scorer in run_eval.py
cannot handle, and lm-evaluation-harness is incompatible with the transformers 5.6
that Qwen3.5 requires. We therefore implement lightweight harnesses on top of the
project's own HFModel.generate (same model-loading path as every other benchmark).

  GSM8K   : 8-shot chain-of-thought prompt -> greedy generation -> extract the final
            number -> exact-match against the gold answer.
  HumanEval: greedy code completion -> execute the generated function against the
            benchmark's unit tests in a sandboxed subprocess -> pass@1.

SECURITY: HumanEval executes model-generated code. Each candidate runs in a separate
subprocess with a wall-clock timeout; run this only on a compute node.

Usage:
  python -m culture.evaluation.run_english_gen \
      --model_path <ckpt> --run_name cpt --tasks gsm8k,humaneval \
      --output_dir results/hi/cpt/en_gen
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from culture.evaluation.scorer import HFModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.eval.en_gen")


# --------------------------------------------------------------------------- #
# GSM8K (8-shot CoT, exact-match)
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _to_number(s: str) -> Optional[str]:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def _gold_number(answer: str) -> Optional[str]:
    if "####" in answer:
        return _to_number(answer.split("####")[-1])
    m = list(_NUM_RE.finditer(answer))
    return _to_number(m[-1].group()) if m else None


def _pred_number(text: str) -> Optional[str]:
    # Prefer the number after "answer is"; else the last number in the text.
    m = re.search(r"answer is\s*:?\s*(-?\$?\d[\d,]*(?:\.\d+)?)", text, flags=re.I)
    if m:
        return _to_number(m.group(1))
    nums = list(_NUM_RE.finditer(text))
    return _to_number(nums[-1].group()) if nums else None


def _gsm8k_fewshot_prefix(train_rows, k: int) -> str:
    parts = []
    for r in train_rows[:k]:
        reasoning = r["answer"].split("####")[0].strip().replace("\n", " ")
        gold = _gold_number(r["answer"])
        parts.append(f"Question: {r['question']}\nAnswer: {reasoning} The answer is {gold}.")
    return "\n\n".join(parts) + "\n\n"


def eval_gsm8k(model: HFModel, limit: Optional[int], num_fewshot: int,
               max_new_tokens: int, batch_size: int, chat: bool = False) -> Dict[str, Any]:
    from datasets import load_dataset
    tok = os.environ.get("HF_TOKEN")
    test = list(load_dataset("openai/gsm8k", "main", split="test", token=tok))
    train = list(load_dataset("openai/gsm8k", "main", split="train", token=tok))
    if limit:
        test = test[:limit]
    prefix = _gsm8k_fewshot_prefix(train, num_fewshot)
    prompts = [prefix + f"Question: {r['question']}\nAnswer:" for r in test]
    # Instruction-tuned models emit free-form reasoning that ends at EOS rather than
    # at "\nQuestion:", so under chat mode we rely on the numeric answer extractor.
    stop = None if chat else ["\nQuestion:", "\n\nQuestion", "\n\n\n"]
    gens = model.generate(prompts, max_new_tokens=max_new_tokens, batch_size=batch_size,
                          stop=stop, chat=chat)
    records, correct = [], 0
    for r, g in zip(test, gens):
        gold = _gold_number(r["answer"])
        pred = _pred_number(g)
        ok = (pred is not None and gold is not None and pred == gold)
        correct += ok
        records.append({"question": r["question"], "gold": gold, "pred": pred,
                        "correct": ok, "generation": g})
    n = len(test)
    metrics = {"task": "gsm8k", "n": n, "num_fewshot": num_fewshot,
               "acc": round(correct / n, 4) if n else None,
               "primary": round(correct / n, 4) if n else None, "metric": "EM"}
    return {"metrics": metrics, "records": records}


# --------------------------------------------------------------------------- #
# HumanEval (pass@1, greedy, sandboxed execution)
# --------------------------------------------------------------------------- #
_STOP_CODE = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n@",
              "\nassert ", "\n#", "\nimport ", "\nfrom "]


def _run_program(program: str, timeout: float = 12.0) -> bool:
    """Execute a standalone program in a subprocess; True iff it exits 0 in time."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return p.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _extract_code(text: str) -> str:
    """Pull a runnable Python snippet out of an instruction-tuned model's reply.

    Instruct models wrap code in Markdown fences and add prose; base models emit a
    bare completion. Prefer the first ```python fenced block; otherwise strip a lone
    leading fence; otherwise return the text unchanged.
    """
    import re
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    if "```" in text:  # unterminated fence
        return text.split("```", 1)[1].lstrip("python").lstrip("py").lstrip("\n")
    return text


def eval_humaneval(model: HFModel, limit: Optional[int], max_new_tokens: int,
                   batch_size: int, timeout: float, chat: bool = False) -> Dict[str, Any]:
    from datasets import load_dataset
    tok = os.environ.get("HF_TOKEN")
    data = list(load_dataset("openai/openai_humaneval", split="test", token=tok))
    if limit:
        data = data[:limit]
    prompts = [r["prompt"] for r in data]
    # Base models: bare code completion (stop at the next top-level stmt). Instruct
    # models: free-form reply with fenced code, so no stop tokens; extract the block.
    stop = None if chat else _STOP_CODE
    gens = model.generate(prompts, max_new_tokens=max_new_tokens, batch_size=batch_size,
                          stop=stop, chat=chat)
    records, passed = [], 0
    for r, completion in zip(data, gens):
        if chat:
            code = _extract_code(completion)
            # A full function (def entry_point) is self-contained; a bare body must be
            # re-attached to the prompt signature and indented one level.
            if f"def {r['entry_point']}" in code or code.lstrip().startswith(("def ", "import ", "from ")):
                program = code + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n"
            else:
                body = code if (code[:1] in (" ", "\t")) else ("    " + code)
                program = (r["prompt"] + body + "\n\n" + r["test"]
                           + f"\n\ncheck({r['entry_point']})\n")
        else:
            # HFModel.generate() strips leading whitespace, which destroys the body's
            # base indentation (first line ends up at column 0 while nested lines keep
            # theirs) -> IndentationError. Re-indent the body by one level if the model
            # didn't emit leading whitespace itself.
            body = completion if (completion[:1] in (" ", "\t")) else ("    " + completion)
            program = (r["prompt"] + body + "\n\n" + r["test"]
                       + f"\n\ncheck({r['entry_point']})\n")
        ok = _run_program(program, timeout=timeout)
        passed += ok
        records.append({"task_id": r["task_id"], "passed": ok, "completion": completion})
    n = len(data)
    metrics = {"task": "humaneval", "n": n,
               "pass@1": round(passed / n, 4) if n else None,
               "primary": round(passed / n, 4) if n else None, "metric": "pass@1"}
    return {"metrics": metrics, "records": records}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default="gsm8k,humaneval")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--gsm8k_num_fewshot", type=int, default=8)
    p.add_argument("--gsm8k_max_new_tokens", type=int, default=320)
    p.add_argument("--humaneval_max_new_tokens", type=int, default=512)
    p.add_argument("--gen_batch_size", type=int, default=8)
    p.add_argument("--humaneval_timeout", type=float, default=12.0)
    p.add_argument("--chat", action="store_true",
                   help="Apply the tokenizer chat template (use for instruction-tuned "
                        "checkpoints; base/CPT checkpoints should stay raw few-shot).")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or os.path.basename(os.path.normpath(args.model_path))

    logger.info("Loading model under test: %s", args.model_path)
    model = HFModel(args.model_path, dtype=args.dtype, max_length=args.max_length)

    summary: Dict[str, Any] = {"run_name": run_name, "model_path": args.model_path, "tasks": {}}
    for name in tasks:
        logger.info("=== Task: %s ===", name)
        if name == "gsm8k":
            out = eval_gsm8k(model, args.limit, args.gsm8k_num_fewshot,
                             args.gsm8k_max_new_tokens, args.gen_batch_size, chat=args.chat)
        elif name == "humaneval":
            out = eval_humaneval(model, args.limit, args.humaneval_max_new_tokens,
                                 args.gen_batch_size, args.humaneval_timeout, chat=args.chat)
        else:
            logger.warning("Unknown task %s; skipping.", name)
            continue
        with open(os.path.join(args.output_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        summary["tasks"][name] = out["metrics"]
        logger.info("%s -> %s", name, json.dumps(out["metrics"], ensure_ascii=False))

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
