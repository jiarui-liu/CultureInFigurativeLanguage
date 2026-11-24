#!/usr/bin/env python3
"""Full-knowledge-base entry audit by two open LLM judges (Qwen3.5-27B-FP8, Gemma-4-26B-A4B-it).

Same questions as the native-speaker sheet (build_annotation_sheets.py, sheet A_KB), so judge labels
can later be calibrated against people:
  meaning        correct / correct_with_noise / partly_correct / wrong
  is_figurative  yes / no
  ent_wrong      listed entities that are not concrete nouns occurring in the expression
  ent_missing    concrete nouns in the expression missing from the list

  prepare : write every entry of every language to <out>/entries.jsonl
  judge   : resumable; appends to <out>/judge_<name>_<lang>.jsonl in chunks and stops before
            --budget_min so each Slurm job stays short (resubmit to continue)
Usage:
  python -m culture.analysis.rebuttal.full_kb_audit_judge prepare --out $A
  python -m culture.analysis.rebuttal.full_kb_audit_judge judge --out $A --name qwen27b --model <path> --lang zh
"""
import argparse
import ast
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
D = REPO / "culture/data/idioms"
KB = {
    "en": D / "en/idioms_merged_llm_formatted_figurative_only.jsonl",
    "zh": D / "zh/idioms_merged_llm_formatted_figurative_only.jsonl",
    "hi": D / "hi/idioms_merged_llm_formatted_figurative_only.jsonl",
    "ar": Path("/data/group_data/r3lit_culture_pretrain/culture/bidir/hf9b/data/idioms/ar/idioms_merged_llm_formatted.jsonl"),
}
LANG = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}

PROMPT = """You are a native-speaker expert auditing one entry of a {lang} idiom and proverb dictionary that was processed automatically. Judge it from your knowledge of {lang}.

Expression: {idiom}
Figurative meaning(s) listed: {fig}
Literal meaning(s) listed: {lit}
Entities listed (concrete nouns said to occur in the expression): {ent}

Answer:
1. meaning: Is the listed figurative meaning the meaning of this expression?
   "correct" = right and clean; "correct_with_noise" = right, but the text also contains extra material such as a quotation, a story, a source citation or broken text; "partly_correct" = captures only part of the meaning or is vague; "wrong" = not the meaning of this expression.
2. is_figurative: Is the expression figurative (idiom, proverb, saying, metaphorical phrase) rather than a plain literal phrase? "yes" or "no".
3. ent_wrong: listed entities that are NOT concrete nouns occurring in the expression (empty list if all are fine).
4. ent_missing: concrete nouns occurring in the expression that are missing from the list (empty list if none).
Return JSON only: {{"meaning": "...", "is_figurative": "...", "ent_wrong": [...], "ent_missing": [...]}}"""


def _lst(v):
    if not v or v == "NAN":
        return []
    return v if isinstance(v, list) else [v]


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "entries.jsonl", "w", encoding="utf-8") as f:
        for L, p in KB.items():
            for i, l in enumerate(open(p, encoding="utf-8")):
                if not l.strip():
                    continue
                r = json.loads(l)
                o = r["output"] if isinstance(r["output"], dict) else ast.literal_eval(r["output"])
                meta = r.get("meta") or {}
                if isinstance(meta, str):
                    try:
                        meta = ast.literal_eval(meta)
                    except Exception:
                        meta = {}
                f.write(json.dumps({"id": f"{L}/{i}", "lang": L, "idiom": str(o.get("idiom") or r.get("idiom")).strip(),
                                    "fig": [str(x) for x in _lst(o.get("figurative_meanings"))][:4],
                                    "lit": [str(x) for x in _lst(o.get("literal_meanings"))][:3],
                                    "ent": [str(x) for x in _lst(o.get("entities"))],
                                    "sources": meta.get("sources", []), "varieties": meta.get("varieties_raw", [])},
                                   ensure_ascii=False) + "\n")
                n += 1
    print("entries", n)


def parse(t):
    if not t:
        return {}
    t = t.strip()
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e < 0:
        return {}
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {}


def judge(a):
    t0 = time.time()
    out = Path(a.out)
    items = [json.loads(l) for l in open(out / "entries.jsonl", encoding="utf-8") if json.loads(l)["lang"] == a.lang]
    fn = out / f"judge_{a.name}_{a.lang}.jsonl"
    done = set()
    if fn.exists():
        for l in open(fn, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [x for x in items if x["id"] not in done]
    print(a.lang, a.name, "total", len(items), "todo", len(todo), flush=True)
    if not todo:
        open(out / f"DONE_{a.name}_{a.lang}", "w").close()
        return
    from vllm import LLM, SamplingParams
    extra = {"quantization": a.quant} if a.quant else {}
    llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.92,
              max_num_seqs=256, enable_prefix_caching=True, **extra)
    sp = SamplingParams(temperature=0.0, max_tokens=200, seed=0)
    per_chunk = []
    for k in range(0, len(todo), a.chunk):
        if per_chunk and (time.time() - t0) / 60 + max(per_chunk) * 1.3 > a.budget_min:
            print("stopping for time budget", flush=True)
            break
        c0 = time.time()
        ch = todo[k:k + a.chunk]
        msgs = [[{"role": "user", "content": PROMPT.format(
            lang=LANG[x["lang"]], idiom=x["idiom"], fig=json.dumps(x["fig"], ensure_ascii=False),
            lit=json.dumps(x["lit"], ensure_ascii=False) if x["lit"] else "(none)",
            ent=json.dumps(x["ent"], ensure_ascii=False))}] for x in ch]
        try:
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            outs = llm.chat(msgs, sp, use_tqdm=False)
        with open(fn, "a", encoding="utf-8") as f:
            for x, o in zip(ch, outs):
                txt = o.outputs[0].text
                f.write(json.dumps({"id": x["id"], "judgment": parse(txt), "raw": txt}, ensure_ascii=False) + "\n")
        per_chunk.append((time.time() - c0) / 60)
        print(f"chunk {k // a.chunk} {len(ch)} items {per_chunk[-1]:.1f} min", flush=True)
    else:
        open(out / f"DONE_{a.name}_{a.lang}", "w").close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "judge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--name")
    ap.add_argument("--model")
    ap.add_argument("--lang")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--budget_min", type=float, default=50)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(Path(a.out))
    else:
        judge(a)


if __name__ == "__main__":
    main()
