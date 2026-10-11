#!/usr/bin/env python3
"""Does how close a benchmark is to a corpus predict how much training on that corpus helps?

§5.3 explains the one reliable reverse-direction culture gain by saying ArabCulture is "the
benchmark closest to the content of the culture corpus".  Closeness is never measured, so the
explanation cannot be wrong.  `corpus_affinity.py` measures it for all benchmarks with the same
embedding model the paper already uses for selection; this script asks whether it predicts the
observed deltas.

For each (benchmark, arm) we take
    x = affinity(benchmark, arm-corpus) - affinity(benchmark, random-corpus)
    y = accuracy(arm) - accuracy(Random-CPT)
and report the Spearman correlation with a permutation p, pooled and per arm.  A positive
correlation turns "closest to the content" into a quantity; a null says that transfer is not
governed by surface-distributional closeness, which is equally worth reporting.

    PYTHONPATH=src python -m culture.analysis.v3.affinity_vs_transfer
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

# analysis arm name -> corpus key in corpus_affinity.json
CORPUS_OF = {"idiom_cpt": "idiom", "idiom_untagged": "idiom_untagged",
             "culture": "culture", "culture_notes": "culturenotes"}


def _rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_p(a, b, reps=20000, seed=0):
    rho = spearman(a, b)
    if rho is None:
        return None, None
    rng = np.random.default_rng(seed)
    b = np.asarray(b, float)
    hits = sum(abs(spearman(a, rng.permutation(b))) >= abs(rho) for _ in range(reps))
    return rho, (hits + 1) / (reps + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="mean_top10")
    ap.add_argument("--out", default="affinity_vs_transfer.json")
    args = ap.parse_args()

    aff = json.load(open(os.path.join(OUT_DIR, "corpus_affinity.json")))
    mc = json.load(open(os.path.join(OUT_DIR, "margins_and_churn.json")))

    pts = []
    for key, v in mc["churn"].items():
        lang, task, arm = key.split("/")
        c = CORPUS_OF.get(arm)
        if c is None or task not in aff["affinity"]:
            continue
        a = aff["affinity"][task]
        if c not in a or "random" not in a:
            continue
        pts.append({
            "lang": lang, "task": task, "arm": arm,
            "x": a[c][args.metric] - a["random"][args.metric],
            "x_abs": a[c][args.metric],
            "y": v["delta_acc"], "n": v["n"],
            "y_margin": mc["margin"].get(key, {}).get("delta_margin"),
        })

    out = {"metric": args.metric, "points": pts, "tests": {}}

    def test(name, sel):
        s = [p for p in pts if sel(p)]
        if len(s) < 6:
            return
        for yk in ("y", "y_margin"):
            vals = [(p["x"], p[yk]) for p in s if p[yk] is not None]
            if len(vals) < 6:
                continue
            x, y = zip(*vals)
            rho, p = perm_p(list(x), list(y))
            out["tests"][f"{name}/{yk}"] = {"n": len(vals), "rho": rho, "p": p}

    test("all", lambda p: True)
    for arm in CORPUS_OF:
        test(f"arm={arm}", lambda p, a=arm: p["arm"] == a)
    for lang in ("ar", "hi", "zh"):
        test(f"lang={lang}", lambda p, l=lang: p["lang"] == l)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}")
    for k, v in sorted(out["tests"].items()):
        print(f"  {k:28s} n={v['n']:3d} rho={v['rho']:+.3f} p={v['p']:.4f}")

    print("\n  top benchmarks by affinity contrast (culture corpus vs random):")
    cs = sorted((p for p in pts if p["arm"] == "culture"), key=lambda p: -p["x"])
    for p in cs:
        print(f"    {p['lang']}/{p['task']:28s} x={p['x']:+.4f} dacc={p['y']:+.3f}")


if __name__ == "__main__":
    main()
