#!/usr/bin/env python3
"""The item-matched cross-language control.

Global-PIQA's parallel split contains the same 103 items in Chinese, Hindi and Arabic, so for
the first time we can ask whether continued pretraining in a language changes *the same item*
differently from how training in another language changes it.  Two readings are possible for
any small benchmark delta: the arm learned something about the item's content (which should
show up in all three languages, since the items are translations), or it learned something
about the language (which should not).  Correlating the per-item deltas across languages
separates them.

    PYTHONPATH=src python -m culture.analysis.v3.parallel_items
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval")
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

TASK = {"ar": "global_piqa_ar_parallel", "hi": "global_piqa_hi_parallel4",
        "zh": "global_piqa_zh_parallel4"}
ARMS = {"random": "unfiltered", "idiom_untagged": "untagged", "idiom_cpt": "cpt",
        "culture": "culture", "culture_notes": "culturenotes", "base": "base"}
SUFFIX = re.compile(r"_(arb_arab|hin_deva|cmn_hans)$", re.I)


def load(lang, arm):
    p = f"{EVAL}/{lang}/{ARMS[arm]}/{TASK[lang]}.json"
    if not os.path.exists(p):
        return None
    out = {}
    for r in json.load(open(p, encoding="utf-8"))["records"]:
        out[SUFFIX.sub("", r["qid"])] = int(r.get("correct_norm", r.get("correct", 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="parallel_items.json")
    args = ap.parse_args()

    res = {"n_shared": None, "correlations": {}, "accuracy_correlation": {}}
    deltas, accs = {}, {}
    for lang in ("ar", "hi", "zh"):
        ref = load(lang, "random")
        if ref is None:
            continue
        accs[lang] = ref
        for arm in ARMS:
            if arm == "random":
                continue
            rec = load(lang, arm)
            if rec is None:
                continue
            deltas[(lang, arm)] = {q: rec[q] - ref[q] for q in rec if q in ref}

    langs = sorted({l for l, _ in deltas})
    shared = set.intersection(*[set(accs[l]) for l in langs]) if langs else set()
    shared = sorted(shared)
    res["n_shared"] = len(shared)
    print(f"shared parallel items across {langs}: {len(shared)}")

    # (a) is the control's per-item correctness itself correlated across languages?
    for i, a in enumerate(langs):
        for b in langs[i + 1:]:
            x = np.array([accs[a][q] for q in shared], float)
            y = np.array([accs[b][q] for q in shared], float)
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
            res["accuracy_correlation"][f"{a}|{b}"] = {"r": r, "n": len(shared),
                                                       "acc_a": float(x.mean()),
                                                       "acc_b": float(y.mean())}

    # (b) are the per-item *deltas* of the same arm correlated across languages?
    for arm in ARMS:
        if arm == "random":
            continue
        avail = [l for l in langs if (l, arm) in deltas]
        for i, a in enumerate(avail):
            for b in avail[i + 1:]:
                q = [x for x in shared if x in deltas[(a, arm)] and x in deltas[(b, arm)]]
                if len(q) < 30:
                    continue
                x = np.array([deltas[(a, arm)][i2] for i2 in q], float)
                y = np.array([deltas[(b, arm)][i2] for i2 in q], float)
                r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None
                # permutation p
                p = None
                if r is not None:
                    rng = np.random.default_rng(0)
                    hits = sum(abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= abs(r)
                               for _ in range(5000))
                    p = (hits + 1) / 5001
                res["correlations"][f"{arm}/{a}|{b}"] = {
                    "r": r, "p": p, "n": len(q),
                    "mean_delta_a": float(x.mean()), "mean_delta_b": float(y.mean())}
                print(f"  {arm:16s} {a}|{b}  r={r} p={p} n={len(q)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(res, open(path, "w"), indent=1)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
