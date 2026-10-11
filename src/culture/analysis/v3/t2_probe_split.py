#!/usr/bin/env python3
"""The sharpest version of the no-holdout test: split the symbolism probe by whether the
training text actually stated what that item's entity symbolizes.

`ar_t2_symall` writes statements for 911 entities, which covers 82 of the probe's 98 items;
the other 16 entities occur in fewer than four idioms and fall below the generator's
threshold. Those 16 items are therefore an internal control trained on the same corpus: if
stating the symbolism teaches it, the gain should sit on the 82 and not on the 16.

Reports accuracy, the gold-minus-best-distractor margin (the accuracy-free measure of
\\S\\ref{sec:results-reverse}), and the lure rate, per subset, with item bootstraps.

    PYTHONPATH=src python -m culture.analysis.v3.t2_probe_split --lang ar
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.environ.get("CULTURE_REPO",
                   "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
                   "CultureInFigurativeLanguage"), "src/culture/analysis/v2"))
import common  # noqa: E402

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval")
DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

TASK = {"ar": "symbolism_v2_ar_letter", "hi": "symbolism_v2_hi_letter"}


def records(sub, arm, task):
    p = f"{EVAL}/{sub}/{arm}/{task}.json"
    return json.load(open(p, encoding="utf-8"))["records"] if os.path.exists(p) else None


def margin(r):
    lp = np.asarray(r.get("logprobs_norm") or r["logprobs"], float)
    g = int(r["gold"])
    best = max(float(lp[i]) for i in range(len(lp)) if i != g)
    return float(lp[g]) - best


def boot(d, reps=10000, seed=0):
    d = np.asarray(d, float)
    if len(d) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ar", "hi"], required=True)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--eval_sub", default=None)
    ap.add_argument("--summaries", default=None,
                    help="the summary file that defines the covered entities")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lang = args.lang
    sub = args.eval_sub or f"{lang}_t2"
    arms = [a.strip() for a in
            (args.arms or ("untagged,dict,sym,symall" if lang == "ar" else
                           "untagged,dict,sym")).split(",") if a.strip()]
    task = TASK[lang]

    covered = set()
    sp = args.summaries or os.path.join(
        DATA, f"t2_{lang}", f"{lang}_entity_symbolism" + ("_all" if lang == "ar" else "")
        + ".json")
    if os.path.exists(sp):
        covered = set(json.load(open(sp, encoding="utf-8"))["entities"])
    print(f"[cover] {len(covered)} entities have a statement ({sp})")

    recs = {a: records(sub, a, task) for a in arms}
    missing = [a for a, v in recs.items() if v is None]
    if missing:
        print(f"[abort] no probe records for {missing}")
        return
    n = len(recs[arms[0]])
    ent = [common.norm_entity(r.get("entity"), lang) for r in recs[arms[0]]]
    mask = np.array([e in covered for e in ent])
    print(f"[probe] {n} items; {int(mask.sum())} stated, {int((~mask).sum())} not stated")

    acc = {a: np.array([int(r.get("correct_norm", r.get("correct", 0)))
                        for r in recs[a]], float) for a in arms}
    mar = {a: np.array([margin(r) for r in recs[a]], float) for a in arms}
    lure = {a: np.array([1.0 if int(r.get("pred_norm", r.get("pred", -1)))
                         == int(r.get("lure", -1)) else 0.0 for r in recs[a]], float)
            for a in arms}

    out = {"lang": lang, "task": task, "n": n, "n_stated": int(mask.sum()),
           "n_not_stated": int((~mask).sum()), "arms": arms, "subsets": {}}
    pairs = [(a, "untagged") for a in arms if a != "untagged"]
    if "symall" in arms and "sym" in arms:
        pairs.append(("symall", "sym"))

    for label, sel in (("all", np.ones(n, bool)), ("stated", mask),
                       ("not_stated", ~mask)):
        row = {"n": int(sel.sum()),
               "acc": {a: float(acc[a][sel].mean()) for a in arms},
               "margin": {a: float(mar[a][sel].mean()) for a in arms},
               "lure": {a: float(lure[a][sel].mean()) for a in arms}}
        for metric, src in (("acc", acc), ("margin", mar), ("lure", lure)):
            for x, y in pairs:
                mu, lo, hi = boot(src[x][sel] - src[y][sel])
                row[f"d_{metric}_{x}_minus_{y}"] = {
                    "delta": mu, "ci": [lo, hi], "sig": not (lo <= 0 <= hi)}
        out["subsets"][label] = row

    # interaction: does the no-holdout gain sit on the items whose entity was stated?
    for metric, src in (("acc", acc), ("margin", mar)):
        for x, y in pairs:
            d = src[x] - src[y]
            mu_s, _, _ = boot(d[mask])
            mu_n, _, _ = boot(d[~mask])
            rng = np.random.default_rng(1)
            reps = 10000
            bs = d[mask][rng.integers(0, int(mask.sum()), size=(reps, int(mask.sum())))].mean(1)
            bn = d[~mask][rng.integers(0, int((~mask).sum()),
                                       size=(reps, int((~mask).sum())))].mean(1)
            diff = bs - bn
            out.setdefault("interaction", {})[f"{metric}_{x}_minus_{y}"] = {
                "stated": mu_s, "not_stated": mu_n, "interaction": float(mu_s - mu_n),
                "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                "p": float(2 * min((diff <= 0).mean(), (diff >= 0).mean()))}

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out or f"t2_{lang}_probe_split.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}\n")

    for label, r in out["subsets"].items():
        print(f"--- {label} (n={r['n']})")
        print("   acc   ", {a: round(100 * v, 1) for a, v in r["acc"].items()})
        print("   margin", {a: round(v, 3) for a, v in r["margin"].items()})
        print("   lure  ", {a: round(100 * v, 1) for a, v in r["lure"].items()})
        for k, v in r.items():
            if k.startswith("d_"):
                s = f"{v['delta']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                print(f"   {k:34s} {s}{'*' if v['sig'] else ''}")
    for k, v in (out.get("interaction") or {}).items():
        print(f"[interaction] {k}: stated {v['stated']:+.3f} vs not {v['not_stated']:+.3f} "
              f"-> {v['interaction']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p={v['p']:.3f}")


if __name__ == "__main__":
    main()
