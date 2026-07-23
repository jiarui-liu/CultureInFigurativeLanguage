#!/usr/bin/env python3
"""Judge stored IdiomCE generations (no GPU / no model reload).

The eval jobs (`eval_core.slurm`) generate Hindi translations with `--no_judge`
and store them in `<eval_root>/<run>/idiomce.json`. This script grades those
stored hypotheses with an OpenAI-compatible judge and folds the aggregate metrics
back into `<run>/summary.json` so the results notebook picks them up.

Judge backend: the project's `ChatModel(provider="openai")`, which honours
`OPENAI_API_KEY` and `OPENAI_API_BASE`. For Meta's MetaGen gateway, set:

    export OPENAI_API_KEY="$METAGEN_API_KEY"          # from ~/.bashrc
    export OPENAI_API_BASE="<metagen OpenAI-compatible endpoint>"
    python -m culture.evaluation.judge_idiomce --judge_model gpt-5-4-mini-genai-responses

If OPENAI_API_KEY is unset but METAGEN_API_KEY is present, we copy it over
automatically. Only ~400 items per run, so a hosted judge is fine (no need for a
local Qwen judge; that would only be worth it for >10k items).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.eval.judge_idiomce")

DEFAULT_EVAL_ROOT = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi"


def _ensure_key():
    """Fall back to the MetaGen key if a plain OPENAI_API_KEY isn't set."""
    if not os.environ.get("OPENAI_API_KEY"):
        mg = os.environ.get("METAGEN_API_KEY")
        if mg:
            os.environ["OPENAI_API_KEY"] = mg
            logger.info("OPENAI_API_KEY unset -> using METAGEN_API_KEY.")
    if os.environ.get("METAGEN_API_BASE") and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.environ["METAGEN_API_BASE"]
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("No OPENAI_API_KEY / METAGEN_API_KEY in env — cannot judge.")
    if not os.environ.get("OPENAI_API_BASE"):
        logger.warning("OPENAI_API_BASE is unset — assuming public OpenAI. For MetaGen, "
                       "export OPENAI_API_BASE to the gateway endpoint.")


def judge_run(run_dir: Path, judge_model: str, provider: str, batch_size: int):
    from culture.evaluation import judge as judge_mod  # imports openai/tenacity lazily

    idiomce_path = run_dir / "idiomce.json"
    if not idiomce_path.exists():
        logger.warning("No idiomce.json in %s — skipping.", run_dir)
        return None
    payload = json.loads(idiomce_path.read_text())
    records = payload.get("records", [])
    if not records:
        logger.warning("idiomce.json in %s has no records — skipping.", run_dir)
        return None

    logger.info("Judging %d IdiomCE hypotheses in %s with %s", len(records), run_dir.name, judge_model)
    judged = judge_mod.judge_translations(records, judge_model=judge_model,
                                          provider=provider, batch_size=batch_size)
    agg = judge_mod.aggregate(judged)
    metrics = {"task": "idiomce", "judged": True, **agg, "primary": agg["idiom_score_mean"]}

    # Write judged detail + update per-task file and summary.
    (run_dir / "idiomce_judged.json").write_text(
        json.dumps({"metrics": metrics, "records": judged}, ensure_ascii=False, indent=2))
    payload["metrics"] = metrics
    idiomce_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    summ_path = run_dir / "summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        summ.setdefault("tasks", {})["idiomce"] = metrics
        summ_path.write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    logger.info("%s IdiomCE judged -> %s", run_dir.name, json.dumps(agg, ensure_ascii=False))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    p.add_argument("--runs", default="base,cpt")
    p.add_argument("--judge_model", default=os.environ.get("JUDGE_MODEL", "gpt-5-4-mini-genai-responses"))
    p.add_argument("--provider", default="openai")
    p.add_argument("--batch_size", type=int, default=20)
    args = p.parse_args()

    _ensure_key()
    root = Path(args.eval_root)
    results = {}
    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        m = judge_run(root / run, args.judge_model, args.provider, args.batch_size)
        if m:
            results[run] = {k: m[k] for k in ("idiom_score_mean", "fluency_mean",
                                              "idiom_rendered_rate", "n_judged", "n_parse_error")}
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if {"base", "cpt"} <= set(results):
        b, c = results["base"]["idiom_score_mean"], results["cpt"]["idiom_score_mean"]
        if b is not None and c is not None:
            print(f"\nIdiomCE idiom_score  base={b}  cpt={c}  Δ={round(c - b, 4)}")


if __name__ == "__main__":
    main()
