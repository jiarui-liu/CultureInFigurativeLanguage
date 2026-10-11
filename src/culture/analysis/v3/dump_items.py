#!/usr/bin/env python3
"""Dump (qid -> item text) for every benchmark the 9B/2B grid is evaluated on.

The eval records under `$EVAL_ROOT/<lang>/<arm>/<task>.json` store per-item `qid`,
`gold`, `logprobs` and `correct`, but not the question text, so no analysis keyed on
*what an item asks* has been possible. This script reconstructs the text by calling
the same task loaders `run_eval` calls, with the same paths as the eval slurm scripts,
and writes one jsonl per task:

    {"qid": ..., "task": ..., "lang": ..., "stem": ..., "options": [...], "gold": int}

`stem` is the loader's `context` with the few-shot prefix stripped (the prefix is
identical for every item of a task, so the common prefix is removed).

    PYTHONPATH=src python -m culture.analysis.v3.dump_items --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from culture.evaluation.tasks import LOADERS
from culture.evaluation.tasks_ar import LOADERS_AR
from culture.evaluation.tasks_zh import LOADERS_ZH

LOADERS = {**LOADERS, **LOADERS_ZH, **LOADERS_AR}

REPO_DATA = os.environ.get(
    "CULTURE_REPO_DATA",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval",
)
LUSTRE_DATA = os.environ.get(
    "CULTURE_LUSTRE_DATA",
    "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval",
)
MC = f"{LUSTRE_DATA}/mc"
AR_KB = ("/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/ar-kb/"
         "data/idioms/ar/idioms_merged_llm_formatted.jsonl")

# China-specific CMMLU subjects, as used for the interaction tests at 9B and 2B.
CMMLU_CHINA = [
    "ancient_chinese", "chinese_civil_service_exam", "chinese_driving_rule",
    "chinese_food_culture", "chinese_foreign_policy", "chinese_history",
    "chinese_literature", "chinese_teacher_qualification", "construction_project_management",
    "elementary_chinese", "elementary_commonsense", "ethnology", "high_school_politics",
    "modern_chinese", "traditional_chinese_medicine", "education",
]

# task -> (lang, how to build it).  `jsonl:` tasks are read directly.
SPECS = {
    # ---------------- Arabic
    "arabculture": ("ar", lambda: LOADERS["arabculture"](
        ar_data_dir=f"{REPO_DATA}/ar/raw", num_fewshot=0, limit=None, seed=42, kb_path=AR_KB)),
    "alyah": ("ar", lambda: LOADERS["alyah"](
        ar_data_dir=f"{REPO_DATA}/ar/raw", num_fewshot=0, limit=None, seed=42, kb_path=AR_KB)),
    "dzirieval": ("ar", lambda: LOADERS["dzirieval"](
        ar_data_dir=f"{REPO_DATA}/ar/raw", num_fewshot=0, limit=None, seed=42, kb_path=AR_KB)),
    "arabic_cultural_qa": ("ar", lambda: LOADERS["arabic_cultural_qa"](
        ar_data_dir=f"{REPO_DATA}/ar/raw", num_fewshot=0, limit=None, seed=42, kb_path=AR_KB,
        dialects=["msa"])),
    "global_piqa_ar": ("ar", lambda: LOADERS["global_piqa_ar"](
        ar_data_dir=f"{REPO_DATA}/ar/raw", num_fewshot=0, limit=None, seed=42, kb_path=AR_KB,
        cultural_only=True)),
    "arabicmmlu": ("ar", lambda: LOADERS["arabicmmlu"](
        ar_data_dir=f"{REPO_DATA}/ar/raw", num_fewshot=0, limit=None, seed=42, kb_path=AR_KB)),
    "kinayat_meaning": ("ar", lambda: LOADERS["kinayat_meaning"](
        ar_data_dir=f"{REPO_DATA}/ar/raw", num_fewshot=0, limit=None, seed=42, kb_path=AR_KB)),
    "ar_figurative": ("ar", lambda: LOADERS["ar_figurative"](
        ar_data_dir=f"{REPO_DATA}/ar/raw", num_fewshot=0, limit=None, seed=42, kb_path=AR_KB)),
    # ---------------- Hindi
    "milu": ("hi", lambda: LOADERS["milu"](
        f"{LUSTRE_DATA}/hi/milu_hi_test.jsonl", num_fewshot=5, limit=None, seed=42,
        fewshot_path=f"{LUSTRE_DATA}/hi/milu_hi_val.jsonl")),
    "global_piqa": ("hi", lambda: LOADERS["global_piqa"](
        f"{LUSTRE_DATA}/hi/global_piqa_hi.tsv", num_fewshot=0, limit=None, seed=42)),
    "mabl": ("hi", lambda: LOADERS["mabl"](
        f"{LUSTRE_DATA}/hi/mabl_hi.csv", num_fewshot=0, limit=None, seed=42)),
    # ---------------- Chinese
    "ccpm": ("zh", lambda: LOADERS["ccpm"](
        f"{REPO_DATA}/zh/CCPM/valid.jsonl", num_fewshot=0, limit=None, seed=42)),
    "cmmlu": ("zh", lambda: LOADERS["cmmlu"](
        subjects=CMMLU_CHINA, num_fewshot=5, cmmlu_dir=f"{REPO_DATA}/zh/cmmlu",
        limit=None, seed=42)),
    "chengyu_bench": ("zh", lambda: LOADERS["chengyu_bench"](
        f"{REPO_DATA}/zh/ChengyuBench", subtask="connotation", num_fewshot=0,
        limit=None, seed=42)),
}

JSONL_TASKS = {
    "idiomatlas_mc_ar_seen": "ar", "idiomatlas_mc_ar_unseen": "ar",
    "idiomatlas_mc_hi_seen": "hi", "idiomatlas_mc_hi_unseen": "hi",
    "idiomatlas_mc_zh_seen": "zh", "idiomatlas_mc_zh_unseen": "zh",
    "global_piqa_zh": "zh", "global_piqa_zh_cultural": "zh",
    "global_piqa_hi": "hi", "global_piqa_hi_cultural": "hi",
    "parambench_hi_culture": "hi", "parambench_hi_other": "hi",
    "symbolism_v2_ar_letter": "ar", "symbolism_v2_hi_letter": "hi",
    "symbolism_v2_zh_letter": "zh",
}


def _common_prefix(strings):
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)
    for i, c in enumerate(s1):
        if i >= len(s2) or c != s2[i]:
            return s1[:i]
    return s1


def dump_task(name, lang, examples, out_dir):
    ctxs = [e.context for e in examples]
    pref = _common_prefix(ctxs)
    # only strip a prefix that looks like a few-shot block (ends on a blank line)
    if "\n\n" in pref:
        pref = pref[: pref.rfind("\n\n") + 2]
    else:
        pref = ""
    path = os.path.join(out_dir, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps({
                "qid": e.qid, "task": name, "lang": lang,
                "stem": e.context[len(pref):].strip(),
                "options": [str(o).strip() for o in e.options],
                "gold": e.gold,
            }, ensure_ascii=False) + "\n")
    print(f"[dump] {name}: {len(examples)} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/lustre-storage/fsx_0/user/jiaruiliu/"
                                     "culture-pretraining-data/bidir/items")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    for name, (lang, build) in SPECS.items():
        if only and name not in only:
            continue
        try:
            task = build()
        except Exception as e:  # a missing file must not kill the run
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        dump_task(name, lang, task.examples, args.out)

    for name, lang in JSONL_TASKS.items():
        if only and name not in only:
            continue
        p = f"{MC}/{name}.jsonl"
        if not os.path.exists(p):
            print(f"[skip] {name}: no file")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for o in rows:
                f.write(json.dumps({
                    "qid": o["qid"], "task": name, "lang": lang,
                    "stem": str(o.get("context", "")).strip(),
                    "options": [str(x).strip() for x in o.get("options", [])],
                    "gold": o.get("gold"),
                    "meta": o.get("meta", {}),
                }, ensure_ascii=False) + "\n")
        print(f"[dump] {name}: {len(rows)}")


if __name__ == "__main__":
    main()
