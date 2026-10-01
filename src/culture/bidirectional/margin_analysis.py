#!/usr/bin/env python3
"""Continuous (log-probability) comparison of each 2B condition against Random-CPT.

Accuracy is a coarse metric for small effects. For every multiple-choice item we take the
length-normalized log-probabilities of the options and compute the gold margin
(gold minus the best distractor). The effect of a condition is the mean paired difference
of margins against Random-CPT on the same items, with a paired bootstrap 95% CI.

Usage:
  python -m culture.bidirectional.margin_analysis --eval_root $B/eval2b --prefix i_ \
      --out docs/paper_stats/v2/bidir_2b_margin.json
"""
import argparse
import json
import os

import numpy as np

from culture.bidirectional.aggregate import ARMS, GROUP


def margins(path):
    out = {}
    for r in json.load(open(path))["records"]:
        lp = r.get("logprobs_norm") or r.get("logprobs")
        if lp is None or len(lp) < 2:
            continue
        g = r["gold"]
        out[r["qid"]] = lp[g] - max(v for i, v in enumerate(lp) if i != g)
    return out


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--prefix", default="i_")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, pooled = [], {}
    for lang in sorted(os.listdir(a.eval_root)):
        rdir = os.path.join(a.eval_root, lang, a.prefix + "random")
        if not os.path.isdir(rdir):
            continue
        for f in sorted(os.listdir(rdir)):
            task = f[:-5]
            if not f.endswith(".json") or task == "summary" or task not in GROUP:
                continue
            ref = margins(os.path.join(rdir, f))
            for arm in ARMS:
                p = os.path.join(a.eval_root, lang, a.prefix + arm, f)
                if not os.path.exists(p):
                    continue
                m = margins(p)
                q = sorted(set(ref) & set(m))
                if not q:
                    continue
                d = np.array([m[k] - ref[k] for k in q])
                # standardize by the Random margin spread so tasks can be pooled
                sd = np.std([ref[k] for k in q]) or 1.0
                lo, hi = boot(d)
                rows.append({"lang": lang, "task": task, "group": GROUP[task], "arm": arm, "n": len(q),
                             "delta_margin": float(d.mean()), "ci95": [lo, hi],
                             "delta_sd": float(d.mean() / sd)})
                pooled.setdefault(f"{arm}|{GROUP[task]}", []).append(d / sd)
    agg = {}
    for k, ds in pooled.items():
        # resample items within each task, then average task means
        rng = np.random.default_rng(0)
        bs = [np.mean([x[rng.integers(0, len(x), len(x))].mean() for x in ds]) for _ in range(2000)]
        agg[k] = {"delta_sd": float(np.mean([x.mean() for x in ds])),
                  "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_tasks": len(ds)}
    json.dump({"per_task": rows, "pooled": agg}, open(a.out, "w"), indent=1)
    for k in sorted(agg):
        v = agg[k]
        print(f"{k:32s} {v['delta_sd']:+.3f} SD  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  tasks={v['n_tasks']}")


if __name__ == "__main__":
    main()
