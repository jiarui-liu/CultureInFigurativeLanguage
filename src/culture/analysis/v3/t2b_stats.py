#!/usr/bin/env python3
"""T2 statistics, generalised to the pass-4 arms.

`t2_stats.py` compares the Arabic triple (untagged / dict / sym). This version takes the arm
list and the language, so it also covers the two follow-ups:

  * ``--lang ar --arms untagged,dict,sym,symall`` -- adds the no-holdout symbolism arm. The
    contrast that matters is ``symall - sym`` on the symbolism probe: both arms state what
    entities symbolize, and they differ only in whether the probe's own 98 entities are among
    them. A null there says the layer resists continued pretraining; a gain says T2's null was
    a failure to generalise.
  * ``--lang hi --arms untagged,dict,sym`` -- the same triple in Hindi, the language whose
    idiom-bearing documents already move the probe's lure rate.

    PYTHONPATH=src python -m culture.analysis.v3.t2b_stats --lang hi
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

TASKS = {
    "ar": ["symbolism_v2_ar_letter", "symbolism_ar", "kinayat_meaning", "kinayat_cloze",
           "ar_figurative", "idiomatlas_mc_ar_seen", "idiomatlas_mc_ar_unseen", "alyah",
           "arabculture", "dzirieval", "arabic_cultural_qa", "global_piqa_ar", "arabicmmlu"],
    "hi": ["symbolism_v2_hi_letter", "symbolism_hi", "idiomatlas_mc_hi_seen",
           "idiomatlas_mc_hi_unseen", "mabl", "global_piqa", "global_piqa_hi",
           "global_piqa_hi_cultural", "global_piqa_hi_parallel4", "parambench_hi_culture",
           "parambench_hi_other", "milu"],
}
PPL = {"ar": ["ppl_wiki_heldout"],
       "hi": ["ppl_hi_proverbs_heldout", "ppl_hi_samanantar_heldout"]}


def load(sub, arm, task):
    p = f"{EVAL}/{sub}/{arm}/{task}.json"
    if not os.path.exists(p):
        return None
    recs = json.load(open(p, encoding="utf-8"))["records"]
    seen, out = {}, {}
    for r in recs:
        q = r["qid"]
        n = seen.get(q, 0)
        seen[q] = n + 1
        # ArabCulture (and others) repeat qids across variants: disambiguate by occurrence
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
    ap.add_argument("--lang", choices=["ar", "hi"], required=True)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--eval_sub", default=None, help="eval/<sub>/<arm>; default <lang>_t2")
    ap.add_argument("--baseline", default="untagged")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lang = args.lang
    sub = args.eval_sub or f"{lang}_t2"
    arms = [a.strip() for a in (args.arms or "untagged,dict,sym").split(",") if a.strip()]
    base = args.baseline
    contrasts = [(a, base) for a in arms if a != base]
    if "symall" in arms and "sym" in arms:
        contrasts.append(("symall", "sym"))

    out = {"lang": lang, "eval_sub": sub, "arms": arms, "accuracy": {}, "perplexity": {}}
    for t in TASKS[lang]:
        got = {a: load(sub, a, t) for a in arms}
        missing = [a for a, v in got.items() if v is None]
        if missing:
            print(f"[skip] {t}: missing {missing}")
            continue
        q = sorted(set.intersection(*[set(got[a]) for a in arms]))
        v = {a: np.array([got[a][x] for x in q]) for a in arms}
        row = {"n": len(q), "acc": {a: float(v[a].mean()) for a in arms}}
        for x, y in contrasts:
            mu, lo, hi = boot(v[x] - v[y])
            row[f"{x}_minus_{y}"] = {"delta": mu, "ci": [lo, hi], "p": mcnemar(v[x], v[y]),
                                     "sig": not (lo <= 0 <= hi)}
        out["accuracy"][t] = row

    for t in PPL[lang]:
        r = {}
        for a in arms:
            p = f"{EVAL}/{sub}/{a}/{t}/perplexity.json"
            if os.path.exists(p):
                d = json.load(open(p))
                r[a] = {"ppl": d["ppl"], "bpb": d.get("bits_per_byte")}
        if r:
            out["perplexity"][t] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out or f"t2_{lang}_stats.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}\n")
    cols = [f"{x}-{y}" for x, y in contrasts]
    print(f"{'task':30s}{'n':>6}" + "".join(f"{c:>24}" for c in cols))
    for t, r in out["accuracy"].items():
        cells = []
        for x, y in contrasts:
            d = r[f"{x}_minus_{y}"]
            s = f"{100 * d['delta']:+.1f} [{100 * d['ci'][0]:+.1f},{100 * d['ci'][1]:+.1f}]"
            cells.append((s + "*") if d["sig"] else s)
        print(f"{t:30s}{r['n']:>6}" + "".join(f"{c:>24}" for c in cells))
    print()
    for t, r in out["perplexity"].items():
        print(t, {a: round(v["ppl"], 3) for a, v in r.items()})


if __name__ == "__main__":
    main()
