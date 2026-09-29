#!/usr/bin/env python3
"""Symbolism-probe (letter format) accuracy, lure rate, and paired contrasts for the 9B checkpoints.
  python -m culture.bidirectional.symbolism_stats --version _v2"""
import argparse
import json

import numpy as np

from culture.evaluation.compute_cis import paired

B = "/data/group_data/r3lit_culture_pretrain/culture/bidir"
ap = argparse.ArgumentParser()
ap.add_argument("--version", default="_v2")
ap.add_argument("--root", default=f"{B}/eval9b")
ap.add_argument("--out", default="docs/paper_stats/v2/symbolism_9b.json")
a = ap.parse_args()
rng = np.random.default_rng(0)
out = {}
for L in ["ar", "hi", "zh"]:
    items = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{B}/eval_data/mc/symbolism{a.version}_{L}_letter.jsonl")}
    acc, lure = {}, {}
    for r in ["base", "unfiltered", "untagged", "cpt"]:
        try:
            rec = json.load(open(f"{a.root}/{L}/{r}/symbolism{a.version}_{L}_letter.json"))["records"]
        except FileNotFoundError:
            continue
        rec = sorted(rec, key=lambda x: x["qid"])
        acc[r] = np.array([int(x["correct"]) for x in rec])
        lure[r] = np.array([int(int(x["pred"]) == items[x["qid"]]["meta"]["lure"]) for x in rec])
    row = {"n": len(items), **{r: {"acc": round(100 * acc[r].mean(), 1), "lure": round(100 * lure[r].mean(), 1)} for r in acc}}
    for x, y in [("cpt", "unfiltered"), ("untagged", "unfiltered"), ("unfiltered", "base")]:
        if x in acc and y in acc:
            d, lo, hi, _, _, p = paired(acc[x], acc[y], rng)
            dl, lol, hil, _, _, pl = paired(lure[x], lure[y], rng)
            row[f"{x}-{y}"] = {"acc": [round(100 * d, 1), round(100 * lo, 1), round(100 * hi, 1), round(p, 3)],
                               "lure": [round(100 * dl, 1), round(100 * lol, 1), round(100 * hil, 1), round(pl, 3)]}
    out[L] = row
    print(L, json.dumps(row))
json.dump(out, open(a.out, "w"), indent=1)
