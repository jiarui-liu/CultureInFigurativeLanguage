#!/usr/bin/env python3
"""Reverse-direction (culture CPT -> idiom benchmarks) 9B paired-bootstrap CIs.

Reuses the exact bootstrap/Holm machinery from the forward-direction analysis:
  - culture.evaluation.compute_cis.{load_run, paired, B, SEED}
  - culture.bidirectional.aggregate.{calibrated, holm, CALIBRATE}

Control arm = "unfiltered" (token-matched Random CPT), matching the paper's
existing convention (see make_ci_appendix.py: "vs. \\random{}" == cpt_vs_unfiltered).

Arms: culture (ar/hi/zh), culturenotes (ar only).

Usage: python reverse_direction_9b_cis.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src")
from culture.evaluation.compute_cis import load_run, paired, B, SEED
from culture.bidirectional.aggregate import calibrated, holm, CALIBRATE

EVAL_ROOT = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval"
OUT = "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/docs/paper_stats/analysis_v2/reverse_direction_9b_cis.json"

TASKS = {
    "ar": ["kinayat_cloze", "kinayat_meaning", "ar_figurative", "arabculture",
           "arabic_cultural_qa", "arabicmmlu", "global_piqa_ar", "alyah", "dzirieval",
           "global_piqa_ar_parallel", "idiomatlas_mc_ar_seen", "idiomatlas_mc_ar_unseen",
           "symbolism_v2_ar_letter"],
    "hi": ["mabl", "global_piqa", "milu", "idiomatlas_mc_hi_seen", "idiomatlas_mc_hi_unseen",
           "symbolism_v2_hi_letter"],
    "zh": ["chid", "chengyu_bench", "chengyu_bench_app", "cmmlu", "ccpm",
           "idiomatlas_mc_zh_seen", "idiomatlas_mc_zh_unseen", "symbolism_v2_zh_letter"],
}

GROUP = {
    "kinayat_meaning": "idiom_meaning", "chengyu_bench": "idiom_meaning", "chengyu_bench_app": "idiom_meaning",
    "idiomatlas_mc_ar_seen": "idiom_meaning", "idiomatlas_mc_hi_seen": "idiom_meaning",
    "idiomatlas_mc_zh_seen": "idiom_meaning",
    "idiomatlas_mc_ar_unseen": "idiom_unseen", "idiomatlas_mc_hi_unseen": "idiom_unseen",
    "idiomatlas_mc_zh_unseen": "idiom_unseen",
    "ar_figurative": "figurative", "mabl": "figurative",
    "kinayat_cloze": "cloze", "chid": "cloze",
    "symbolism_v2_ar_letter": "symbolism", "symbolism_v2_hi_letter": "symbolism", "symbolism_v2_zh_letter": "symbolism",
    "alyah": "culture", "dzirieval": "culture", "arabculture": "culture", "arabic_cultural_qa": "culture",
    "global_piqa_ar": "culture", "global_piqa": "culture", "ccpm": "culture",
    "arabicmmlu": "regional", "milu": "regional", "cmmlu": "regional",
    "global_piqa_ar_parallel": "control",
}

ARMS = {"ar": ["culture", "culturenotes"], "hi": ["culture"], "zh": ["culture"]}


def load_with_calibration(run_dir):
    d = load_run(run_dir)
    for t in CALIBRATE & set(d):
        c = calibrated(run_dir, t)
        if c is not None:
            d[t] = c
    return d


def main():
    rng = np.random.default_rng(SEED)
    runs = {}
    for lang in ["ar", "hi", "zh"]:
        runs[lang] = {}
        for arm in ["unfiltered"] + ARMS[lang]:
            d = os.path.join(EVAL_ROOT, lang, arm)
            runs[lang][arm] = load_with_calibration(d) if os.path.isdir(d) else {}

    report = {}
    rows_for_holm = []  # (lang, arm, task) -> contrast dict, for Holm within (lang, group)
    missing = []

    for lang in ["ar", "hi", "zh"]:
        report[lang] = {}
        control = runs[lang]["unfiltered"]
        for task in TASKS[lang]:
            entry = {"n": None, "random_score": None}
            if task not in control:
                missing.append((lang, task, "unfiltered/control has no per-item data for this task "
                                             "(it was never evaluated on the Random checkpoint)"))
                for arm in ARMS[lang]:
                    entry[f"{arm}_score"] = round(float(np.mean(list(runs[lang][arm][task].values()))), 4) \
                        if task in runs[lang].get(arm, {}) else None
                    entry[f"{arm}_delta"] = None
                    entry[f"{arm}_ci_95"] = None
                report[lang][task] = entry
                continue
            rvals = control[task]
            entry["random_score"] = round(float(np.mean(list(rvals.values()))), 4)
            entry["n_random"] = len(rvals)
            for arm in ARMS[lang]:
                adict = runs[lang].get(arm, {})
                if task not in adict:
                    missing.append((lang, task, f"{arm} has no per-item data for this task"))
                    entry[f"{arm}_score"] = None
                    entry[f"{arm}_delta"] = None
                    entry[f"{arm}_ci_95"] = None
                    continue
                avals = adict[task]
                common = [q for q in avals if q in rvals]
                if len(common) != len(avals) or len(common) != len(rvals):
                    missing.append((lang, task, f"{arm} vs unfiltered item mismatch: "
                                                 f"{arm} n={len(avals)}, unfiltered n={len(rvals)}, common={len(common)}"))
                a = [avals[q] for q in common]
                b_ = [rvals[q] for q in common]
                d, lo, hi, a_win, b_win, p = paired(a, b_, rng, bN=B)
                entry[f"{arm}_score"] = round(float(np.mean(a)), 4)
                entry[f"{arm}_delta"] = round(float(d), 4)
                entry[f"{arm}_ci_95"] = [round(lo, 4), round(hi, 4)]
                entry[f"{arm}_n"] = len(common)
                entry[f"{arm}_p_mcnemar"] = round(float(p), 5)
                rows_for_holm.append({"lang": lang, "arm": arm, "task": task,
                                       "group": GROUP.get(task, "other"), "p": float(p)})
            entry["n"] = len(rvals)
            report[lang][task] = entry

    # Holm correction within (lang, arm, group), matching aggregate.py's convention
    holm_lookup = {}
    keyf = lambda r: (r["lang"], r["arm"], r["group"])
    groups = {}
    for r in rows_for_holm:
        groups.setdefault(keyf(r), []).append(r)
    for k, rs in groups.items():
        adj = holm([r["p"] for r in rs])
        for r, a in zip(rs, adj):
            holm_lookup[(r["lang"], r["arm"], r["task"])] = a

    for lang in ["ar", "hi", "zh"]:
        for task in TASKS[lang]:
            entry = report[lang][task]
            for arm in ARMS[lang]:
                key = (lang, arm, task)
                if key in holm_lookup:
                    entry[f"{arm}_p_holm_group"] = round(holm_lookup[key], 5)
                    lo, hi = entry[f"{arm}_ci_95"]
                    entry[f"{arm}_sig_holm"] = bool(holm_lookup[key] < 0.05 and not (lo <= 0 <= hi))

    report["_missing_data_notes"] = [f"{l}/{t}: {m}" for l, t, m in missing]
    report["_meta"] = {
        "bootstrap_resamples": B, "seed": SEED, "control_arm": "unfiltered",
        "holm_correction": "within (lang, arm, benchmark-group), group assignment follows "
                            "culture.bidirectional.aggregate.GROUP",
        "chengyu_bench_calibration": "median-centered log-prob-difference label-prior correction "
                                      "applied via culture.bidirectional.aggregate.calibrated() "
                                      "(chengyu_bench only; chengyu_bench_app has no Random-arm data "
                                      "to calibrate against)",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("wrote", OUT)
    print(f"\n{len(missing)} missing-data notes:")
    for m in missing:
        print(" -", m)


if __name__ == "__main__":
    main()
