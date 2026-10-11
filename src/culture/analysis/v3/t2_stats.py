#!/usr/bin/env python3
"""T2 statistics: does stating what an entity symbolizes teach the symbolic layer?

Three arms trained on the same 410,669 Arabic idiom-bearing documents for the same number of
tokens, differing only in what is appended: nothing, the dictionary meaning tag, or statements
of what each matched idiom's entities symbolize in Arabic proverbs (built from the knowledge
base, with the probe's entities excluded). The contrast that answers the question is
`sym - untagged` on the symbolism probe, against `dict - untagged` on idiom meaning.

    PYTHONPATH=src python -m culture.analysis.v3.t2_stats
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

TASKS = ["symbolism_v2_ar_letter", "symbolism_ar", "kinayat_meaning", "kinayat_cloze",
         "ar_figurative", "idiomatlas_mc_ar_seen", "idiomatlas_mc_ar_unseen", "alyah",
         "arabculture", "dzirieval", "arabic_cultural_qa", "global_piqa_ar", "arabicmmlu"]
PPL = ["ppl_wiki_heldout"]
ARMS = ("untagged", "dict", "sym")


def load(arm, task, step=None):
    p = f"{EVAL}/ar_t2/{arm}/{task}.json"
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
    ap.add_argument("--out", default="t2_stats.json")
    args = ap.parse_args()
    out = {"step": args.step, "accuracy": {}, "perplexity": {}}

    for t in TASKS:
        arms = {a: load(a, t) for a in ARMS}
        if any(v is None for v in arms.values()):
            continue
        q = sorted(set.intersection(*[set(arms[a]) for a in ARMS]))
        v = {a: np.array([arms[a][x] for x in q]) for a in arms}
        row = {"n": len(q), "acc": {a: float(v[a].mean()) for a in v}}
        for name, (x, y) in {
            "dict_minus_untagged": ("dict", "untagged"),
            "sym_minus_untagged": ("sym", "untagged"),
            "sym_minus_dict": ("sym", "dict"),
        }.items():
            mu, lo, hi = boot(v[x] - v[y])
            row[name] = {"delta": mu, "ci": [lo, hi], "p": mcnemar(v[x], v[y]),
                         "sig": not (lo <= 0 <= hi)}
        out["accuracy"][t] = row

    for t in PPL:
        r = {}
        for a in ARMS:
            p = f"{EVAL}/ar_t2/{a}/{t}/perplexity.json"
            if os.path.exists(p):
                d = json.load(open(p))
                r[a] = {"ppl": d["ppl"], "bpb": d.get("bits_per_byte")}
        out["perplexity"][t] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}\n")
    print(f"{'task':28s}{'n':>6}{'dict-untag':>22}{'sym-untag':>22}{'sym-dict':>22}")
    for t, r in out["accuracy"].items():
        def f(k):
            d = r[k]
            s = f"{100 * d['delta']:+.1f} [{100 * d['ci'][0]:+.1f},{100 * d['ci'][1]:+.1f}]"
            return (s + "*") if d["sig"] else s
        print(f"{t:28s}{r['n']:>6}{f('dict_minus_untagged'):>22}"
              f"{f('sym_minus_untagged'):>22}{f('sym_minus_dict'):>22}")
    print()
    for t, r in out["perplexity"].items():
        print(t, {a: round(v["ppl"], 3) for a, v in r.items()})


if __name__ == "__main__":
    main()
