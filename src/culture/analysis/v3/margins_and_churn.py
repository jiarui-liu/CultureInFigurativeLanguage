#!/usr/bin/env python3
"""Three accuracy-free views of what continued pretraining does to the model.

A5  **Decision margin.**  Accuracy is a thresholded statistic: a model can move a long way
    toward the right answer without crossing the threshold on any item.  §5.5 measures the
    gold-minus-best-distractor log-probability margin on one benchmark (Kinayat-Meaning at 2B)
    and generalises informally.  Here it is computed for every arm x benchmark at 9B, which
    turns "no reliable gain" into either "no learning" or "learning the accuracy hides".

A8b **Churn.**  A delta of +0.3 points can be a model that changed nothing or a model that
    fixed 12% of the items and broke 11%.  For every arm we report items fixed, items broken,
    and the churn rate against the token-matched control -- the first statement in the paper
    about whether in-language continued pretraining *reshuffles* a benchmark or *adds* to it.

A8c **Agreement matrix.**  How often two arms give the same answer, which bounds how different
    the models can be whatever their accuracies say.

A8a **Perplexity profile.**  The `ppl_*` records already on disk, collected into one table.

    PYTHONPATH=src python -m culture.analysis.v3.margins_and_churn
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval",
)
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

ARMS = {"base": "base", "random": "unfiltered", "idiom_untagged": "untagged",
        "idiom_cpt": "cpt", "culture": "culture", "culture_notes": "culturenotes"}
REF = "random"
LANGS = ("ar", "hi", "zh")


def uniq_keys(records):
    """ArabCulture repeats a qid across country variants (3,463 rows, 2,168 distinct ids),
    so keying a dict on qid silently drops a third of the benchmark. Disambiguate by the
    occurrence number, which is stable because every arm scores the items in the same order."""
    seen = {}
    out = []
    for r in records:
        q = r["qid"]
        n = seen.get(q, 0)
        seen[q] = n + 1
        out.append(q if n == 0 else f"{q}#{n}")
    return out


def load(lang, arm, task):
    p = f"{EVAL}/{lang}/{ARMS[arm]}/{task}.json"
    if not os.path.exists(p):
        return None
    recs = json.load(open(p, encoding="utf-8"))["records"]
    for k, r in zip(uniq_keys(recs), recs):
        r["qid"] = k
    return recs


def tasks_of(lang):
    d = f"{EVAL}/{lang}/{ARMS[REF]}"
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{d}/*.json")
                  if not os.path.basename(p).startswith(("summary", "idiomce")))


def margin(rec, norm=True):
    lp = rec.get("logprobs_norm" if norm else "logprobs") or []
    g = rec.get("gold")
    if not lp or not isinstance(g, int) or g >= len(lp):
        return None
    others = [v for i, v in enumerate(lp) if i != g]
    if not others:
        return None
    return float(lp[g] - max(others))


def boot_ci(d, reps=5000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):
    """Exact two-sided McNemar on paired 0/1 correctness arrays (b = reference)."""
    from math import comb
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), n01, n10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="margins_and_churn.json")
    ap.add_argument("--eval_root", default=None,
                    help="override the 9B eval root (e.g. the 2B records from babel)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated name=dir pairs, e.g. random=i_random,idiom_cpt=i_idiom_tagged")
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()

    global EVAL, ARMS, REF
    if args.eval_root:
        EVAL = args.eval_root
    if args.arms:
        ARMS = dict(p.split("=", 1) for p in args.arms.split(","))
    if args.ref:
        REF = args.ref

    out = {"margin": {}, "churn": {}, "agreement": {}, "perplexity": {}}

    for lang in LANGS:
        for task in tasks_of(lang):
            ref = load(lang, REF, task)
            if not ref:
                continue
            ref_m = {r["qid"]: margin(r) for r in ref}
            ref_c = {r["qid"]: int(r.get("correct_norm", r.get("correct", 0))) for r in ref}
            ref_p = {r["qid"]: r.get("pred_norm", r.get("pred")) for r in ref}

            preds = {REF: ref_p}
            for arm in ARMS:
                if arm == REF:
                    continue
                rec = load(lang, arm, task)
                if not rec:
                    continue
                qids = [r["qid"] for r in rec if r["qid"] in ref_m]
                am = {r["qid"]: margin(r) for r in rec}
                ac = {r["qid"]: int(r.get("correct_norm", r.get("correct", 0))) for r in rec}
                preds[arm] = {r["qid"]: r.get("pred_norm", r.get("pred")) for r in rec}

                dm = [am[q] - ref_m[q] for q in qids
                      if am.get(q) is not None and ref_m.get(q) is not None]
                key = f"{lang}/{task}/{arm}"
                if dm:
                    mu, lo, hi = boot_ci(dm)
                    out["margin"][key] = {"n": len(dm), "delta_margin": mu,
                                          "ci": [lo, hi], "sig": not (lo <= 0 <= hi)}
                a = np.array([ac[q] for q in qids])
                b = np.array([ref_c[q] for q in qids])
                p, fixed, broken = mcnemar(a, b)
                out["churn"][key] = {
                    "n": len(qids), "acc": float(a.mean()), "acc_ref": float(b.mean()),
                    "delta_acc": float(a.mean() - b.mean()),
                    "fixed": fixed, "broken": broken,
                    "churn_rate": (fixed + broken) / max(1, len(qids)),
                    "mcnemar_p": p,
                }
            # pairwise answer agreement
            names = [a for a in preds if preds[a]]
            common = set.intersection(*[set(preds[a]) for a in names]) if names else set()
            common = sorted(common)
            if common:
                out["agreement"][f"{lang}/{task}"] = {
                    f"{x}|{y}": float(np.mean([preds[x][q] == preds[y][q] for q in common]))
                    for i, x in enumerate(names) for y in names[i + 1:]
                }

        # perplexity profile
        for arm in ARMS:
            for p in glob.glob(f"{EVAL}/{lang}/{ARMS[arm]}/ppl_*/perplexity.json"):
                d = json.load(open(p))
                probe = os.path.basename(os.path.dirname(p))[4:]
                out["perplexity"][f"{lang}/{probe}/{arm}"] = {
                    "ppl": d.get("ppl"), "bpb": d.get("bits_per_byte"),
                    "n_tokens": d.get("num_tokens"),
                }

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}")

    # ---- readable summary
    print("\n=== margin shifts significant at 95% (delta nats, gold - best distractor) ===")
    for k, v in sorted(out["margin"].items()):
        if v["sig"]:
            print(f"  {k:52s} {v['delta_margin']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] n={v['n']}")
    print("\n=== churn: flat accuracy, moving answers ===")
    for k, v in sorted(out["churn"].items()):
        if abs(v["delta_acc"]) < 0.01 and v["churn_rate"] > 0.10:
            print(f"  {k:52s} dacc={v['delta_acc']:+.3f} churn={v['churn_rate']:.2f} "
                  f"(fixed {v['fixed']} / broken {v['broken']}) n={v['n']}")


if __name__ == "__main__":
    main()
