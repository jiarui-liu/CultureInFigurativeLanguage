#!/usr/bin/env python3
"""How often each Kinayat-Meaning expression occurs in each Arabic 2B training arm.

The culture arms exclude IdiomAtlas idioms but not Kinayat expressions, so a gain of the
culture arms on Kinayat-Meaning could come from seeing the expressions themselves. This
counts, per arm, the documents that contain each expression (diacritics and alef/ya/ta
marbuta variants normalized), and splits the margin effect by exposure.

Usage:
  python -m culture.bidirectional.kinayat_exposure --B $B --out docs/paper_stats/v2/kinayat_exposure.json
"""
import argparse
import gzip
import json
import os
import re

import numpy as np

DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")


def norm(s):
    s = DIAC.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="random,culture_docs,culture_notes,idiom_untagged")
    a = ap.parse_args()
    ref = json.load(open(os.path.join(a.B, "eval2b/ar/i_random/kinayat_meaning.json")))["records"]
    exprs = {r["qid"]: norm(r["idiom"]) for r in ref}
    pats = sorted({e for e in exprs.values() if len(e) >= 4}, key=len, reverse=True)
    rx = re.compile("|".join(re.escape(p) for p in pats))
    counts = {}
    for arm in a.arms.split(","):
        c = dict.fromkeys(pats, 0)
        n = 0
        with gzip.open(os.path.join(a.B, "arms/ar", arm + ".jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                n += 1
                t = norm(json.loads(line)["text"])
                for m in set(rx.findall(t)):
                    c[m] += 1
        counts[arm] = {"docs": n, "per_expr": c,
                       "items_with_any": sum(1 for q, e in exprs.items() if c.get(e, 0) > 0)}
        print(arm, n, "docs; items whose expression occurs:", counts[arm]["items_with_any"], "/", len(exprs), flush=True)

    # margin effect of the culture arm split by whether its corpus contains the expression
    def margins(run):
        out = {}
        for r in json.load(open(os.path.join(a.B, "eval2b/ar", run, "kinayat_meaning.json")))["records"]:
            lp = json.loads(r["logprobs_norm"]) if isinstance(r["logprobs_norm"], str) else r["logprobs_norm"]
            g = int(r["gold"])
            out[r["qid"]] = lp[g] - max(v for i, v in enumerate(lp) if i != g)
        return out

    rnd = margins("i_random")
    split = {}
    for run, arm in [("i_culture", "culture_docs"), ("i_culture_notes", "culture_notes"),
                     ("i_idiom_untagged", "idiom_untagged")]:
        m = margins(run)
        for seen in (True, False):
            q = [k for k in rnd if (counts[arm]["per_expr"].get(exprs[k], 0) > 0) == seen and k in m]
            d = np.array([m[k] - rnd[k] for k in q])
            if len(d) == 0:
                continue
            rng = np.random.default_rng(0)
            bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
            split[f"{run}|{'in_corpus' if seen else 'absent'}"] = {
                "n": len(d), "delta_margin": float(d.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    json.dump({"counts": {k: {kk: vv for kk, vv in v.items() if kk != "per_expr"} for k, v in counts.items()},
               "split": split}, open(a.out, "w"), indent=1)
    for k, v in split.items():
        print(f"{k:40s} n={v['n']:4d} {v['delta_margin']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
