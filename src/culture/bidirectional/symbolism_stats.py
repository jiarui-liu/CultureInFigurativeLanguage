#!/usr/bin/env python3
"""Symbolism-probe (letter format) accuracy, lure rate, and paired contrasts for the 9B checkpoints.

Covers the forward arms (base / Random / IdiomDocs / Idiom-CPT) and the reverse ones
(Culture-CPT, Culture+Notes), all scored in the letter format: log-likelihood scoring of
these short, abstract options tracks the options' own prior, so it is not comparable.

  python -m culture.bidirectional.symbolism_stats --version _v2
"""
import argparse
import json
import os

import numpy as np

from culture.evaluation.compute_cis import paired

EVAL = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval"
MC = "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc"
ARMS = ["base", "unfiltered", "untagged", "cpt", "culture", "culturenotes"]
CONTRASTS = [("cpt", "unfiltered"), ("untagged", "unfiltered"), ("culture", "unfiltered"),
             ("culturenotes", "unfiltered"), ("unfiltered", "base"),
             ("cpt", "base"), ("cpt", "untagged")]

ap = argparse.ArgumentParser()
ap.add_argument("--version", default="_v2")
ap.add_argument("--root", default=EVAL)
ap.add_argument("--mc", default=MC)
ap.add_argument("--out", default="docs/paper_stats/v2/symbolism_9b.json")
a = ap.parse_args()
rng = np.random.default_rng(0)
out = {}
for L in ["ar", "hi", "zh"]:
    items = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{a.mc}/symbolism{a.version}_{L}_letter.jsonl")}
    acc, lure = {}, {}
    for r in ARMS:
        p = f"{a.root}/{L}/{r}/symbolism{a.version}_{L}_letter.json"
        if not os.path.exists(p):
            continue
        rec = sorted(json.load(open(p))["records"], key=lambda x: x["qid"])
        acc[r] = np.array([int(x["correct"]) for x in rec])
        lure[r] = np.array([int(int(x["pred"]) == items[x["qid"]]["meta"]["lure"]) for x in rec])
    row = {"n": len(items), **{r: {"acc": round(100 * acc[r].mean(), 1), "lure": round(100 * lure[r].mean(), 1)} for r in acc}}
    for x, y in CONTRASTS:
        if x in acc and y in acc:
            d, lo, hi, _, _, p = paired(acc[x], acc[y], rng)
            dl, lol, hil, _, _, pl = paired(lure[x], lure[y], rng)
            row[f"{x}-{y}"] = {"acc": [round(100 * d, 1), round(100 * lo, 1), round(100 * hi, 1), round(p, 3)],
                               "lure": [round(100 * dl, 1), round(100 * lol, 1), round(100 * hil, 1), round(pl, 3)]}
    out[L] = row
    print(L, json.dumps(row))
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
