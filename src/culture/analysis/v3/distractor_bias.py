#!/usr/bin/env python3
"""Where do the errors on unseen idioms go?

§5.1 reports that on the unseen split of IdiomAtlas-MC in Arabic, \\idiomcpt{}'s errors lean
toward distractors whose meanings occurred in the training tags (45.2% of errors against a
41.5% share of such distractors), and reads this as the tags installing a preference for
familiar dictionary glosses.  The claim is made in one language for one arm.  This script
computes the same quantity for every arm and every language, with a binomial test against the
distractor-composition baseline, so the mechanism is either general or it is not.

    PYTHONPATH=src:src/culture/analysis/v2 python distractor_bias.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import common  # noqa: E402

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval")
DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")
MC = f"{DATA}/eval/mc"
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

COUNT_FILES = {
    "zh": [f"{DATA}/fineweb-edu-zh-chengyu-cpt/stats/kept_idiom_counts_zh.json",
           f"{DATA}/mc4-zh-idiom-cpt/stats/kept_idiom_counts_zh.json"],
    "hi": [f"{DATA}/hi-proverbs-cpt/stats/kept_idiom_counts_hi.json"],
    "ar": [f"{DATA}/ar-amthal-cpt/stats/kept_idiom_counts_ar.json"],
}
ARMS = {"base": "base", "random": "unfiltered", "idiom_untagged": "untagged",
        "idiom_cpt": "cpt", "culture": "culture", "culture_notes": "culturenotes"}


def binom_p(k, n, p):
    """Two-sided exact binomial test."""
    from math import comb
    if n == 0:
        return 1.0
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(x for x in probs if x <= obs * (1 + 1e-9))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="distractor_bias.json")
    args = ap.parse_args()
    out = {}

    for lang in ("ar", "hi", "zh"):
        counts = Counter()
        for p in COUNT_FILES[lang]:
            if os.path.exists(p):
                for k, v in json.load(open(p, encoding="utf-8")).items():
                    counts[k] += int(v)
        rows = [json.loads(l) for l in
                open(f"{MC}/idiomatlas_mc_{lang}_unseen.jsonl", encoding="utf-8")]
        # For each item: which of the three distractors come from an idiom the tags glossed?
        info = {}
        for o in rows:
            dis = (o.get("meta") or {}).get("distractor_idioms") or []
            if not dis:
                continue
            seen_flags = [counts.get(d, 0) > 0 for d in dis]
            info[o["qid"]] = {"gold": o["gold"], "n_opts": len(o["options"]),
                              "seen_flags": seen_flags}
        if not info:
            print(f"[skip] {lang}: no distractor metadata")
            continue
        # baseline: among the wrong options, the share that are tag-glossed
        base_share = float(np.mean([np.mean(v["seen_flags"]) for v in info.values()]))

        res = {"n_items": len(info), "baseline_seen_distractor_share": base_share, "arms": {}}
        for arm in ARMS:
            p = f"{EVAL}/{lang}/{ARMS[arm]}/idiomatlas_mc_{lang}_unseen.json"
            if not os.path.exists(p):
                continue
            recs = json.load(open(p, encoding="utf-8"))["records"]
            err = tag_err = 0
            for r in recs:
                v = info.get(r["qid"])
                if v is None:
                    continue
                pred = r.get("pred_norm", r.get("pred"))
                if pred is None or pred == v["gold"]:
                    continue
                err += 1
                # distractor index: options are gold + three distractors in the stored order;
                # the j-th wrong option corresponds to seen_flags[j]
                wrong_idx = [i for i in range(v["n_opts"]) if i != v["gold"]]
                try:
                    j = wrong_idx.index(pred)
                except ValueError:
                    continue
                if j < len(v["seen_flags"]) and v["seen_flags"][j]:
                    tag_err += 1
            if err == 0:
                continue
            res["arms"][arm] = {
                "n_errors": err, "share_tag_glossed": tag_err / err,
                "excess_over_baseline": tag_err / err - base_share,
                "p": binom_p(tag_err, err, base_share),
            }
            print(f"[{lang}/{arm}] errors={err:4d} tag-glossed {tag_err/err:.3f} "
                  f"(baseline {base_share:.3f}, p={res['arms'][arm]['p']:.3f})")
        out[lang] = res

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
