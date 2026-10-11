#!/usr/bin/env python3
"""T1 statistics: is the Chinese Culture-CPT cost caused by excluding idiom-bearing documents?

Three arms trained from the same checkpoint for the same number of tokens on the same web
pool, differing only in the selection rule (`random`, culture-rich with the paper's idiom-free
restriction, culture-rich without it). The contrast that answers the question is
`culture_free - culture_all`: both are culture-selected, only one is restricted.

    PYTHONPATH=src python -m culture.analysis.v3.t1_stats
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval")
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

TASKS = ["chid", "chengyu_bench", "chengyu_bench_app", "cmmlu", "ccpm", "global_piqa_zh",
         "global_piqa_zh_cultural", "idiomatlas_mc_zh_seen", "idiomatlas_mc_zh_unseen",
         "symbolism_v2_zh_letter"]
PPL = ["ppl_zh_wiki", "ppl_zh_chengyu"]


def load(arm, task, step):
    p = f"{EVAL}/zh/t1_{arm}_{step}/{task}.json"
    if not os.path.exists(p):
        return None
    recs = json.load(open(p, encoding="utf-8"))["records"]
    seen, out = {}, {}
    for r in recs:
        q = r["qid"]
        n = seen.get(q, 0)
        seen[q] = n + 1
        out[q if n == 0 else f"{q}#{n}"] = int(r.get("correct_norm", r.get("correct", 0)))
    return out


def boot(d, reps=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):
    from math import comb
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="final")
    ap.add_argument("--out", default="t1_stats.json")
    args = ap.parse_args()
    out = {"step": args.step, "accuracy": {}, "perplexity": {}}

    for t in TASKS:
        arms = {a: load(a, t, args.step) for a in ("random", "culture_free", "culture_all")}
        if any(v is None for v in arms.values()):
            continue
        q = sorted(set(arms["random"]) & set(arms["culture_free"]) & set(arms["culture_all"]))
        v = {a: np.array([arms[a][x] for x in q]) for a in arms}
        row = {"n": len(q), "acc": {a: float(v[a].mean()) for a in v}}
        for name, (x, y) in {
            "free_minus_random": ("culture_free", "random"),
            "all_minus_random": ("culture_all", "random"),
            "free_minus_all": ("culture_free", "culture_all"),
        }.items():
            mu, lo, hi = boot(v[x] - v[y])
            row[name] = {"delta": mu, "ci": [lo, hi], "p": mcnemar(v[x], v[y]),
                         "sig": not (lo <= 0 <= hi)}
        out["accuracy"][t] = row

    for t in PPL:
        r = {}
        for a in ("random", "culture_free", "culture_all"):
            p = f"{EVAL}/zh/t1_{a}_{args.step}/{t}/perplexity.json"
            if os.path.exists(p):
                d = json.load(open(p))
                r[a] = {"ppl": d["ppl"], "bpb": d.get("bits_per_byte")}
        out["perplexity"][t] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}\n")
    print(f"{'task':28s}{'n':>6}{'free-rand':>22}{'all-rand':>22}{'free-all':>22}")
    for t, r in out["accuracy"].items():
        def f(k):
            d = r[k]
            s = f"{100 * d['delta']:+.1f} [{100 * d['ci'][0]:+.1f},{100 * d['ci'][1]:+.1f}]"
            return (s + "*") if d["sig"] else s
        print(f"{t:28s}{r['n']:>6}{f('free_minus_random'):>22}"
              f"{f('all_minus_random'):>22}{f('free_minus_all'):>22}")
    print()
    for t, r in out["perplexity"].items():
        print(t, {a: round(v["ppl"], 3) for a, v in r.items()})


if __name__ == "__main__":
    main()
