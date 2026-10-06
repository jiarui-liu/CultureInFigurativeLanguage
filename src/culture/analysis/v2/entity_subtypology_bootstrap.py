#!/usr/bin/env python3
"""Task 1b robustness: idiom-level bootstrap CIs for the expanded (8 concrete + 7 subtype)
table, matching entity_typology_bootstrap.py.

Splitting abstract_other into seven parts spreads the same mention total over more cells, so
the per-cell CIs widen; this is what tells you which of the subtype contrasts actually survive.

    PYTHONPATH=src python src/culture/analysis/v2/entity_subtypology_bootstrap.py
"""
import json
import os

import numpy as np

from common import OUT, dump, load_kb
from entity_typology import LANGS, cramers_v
from entity_subtypology import CONCRETE, SUBTYPES


def main():
    rng = np.random.default_rng(0)
    types = CONCRETE + SUBTYPES
    per_lang = {}
    for lang in LANGS:
        prim = json.load(open(os.path.join(OUT, f"entity_typology_labels_{lang}.json")))
        subs = json.load(open(os.path.join(OUT, f"entity_subtypology_labels_{lang}.json")))
        lab = {e: v["type"] for e, v in prim.items() if v.get("type") and v["type"] in CONCRETE}
        lab.update({e: v["subtype"] for e, v in subs.items() if v.get("subtype")})
        rows = load_kb(lang)
        M = np.zeros((len(rows), len(types)))
        for i, r in enumerate(rows):
            for e in r["entities"]:
                t = lab.get(e)
                if t in types:
                    M[i, types.index(t)] += 1
        per_lang[lang] = M[M.sum(1) > 0]

    B = 1000
    shares = {l: [] for l in LANGS}
    within = {l: [] for l in LANGS}
    vs, pair_v = [], {}
    sub_idx = [types.index(t) for t in SUBTYPES]
    for _ in range(B):
        tab = []
        for l in LANGS:
            M = per_lang[l]
            s = M[rng.integers(0, len(M), len(M))].sum(0)
            tab.append(s)
            shares[l].append(s / s.sum())
            a = s[sub_idx]
            within[l].append(a / a.sum() if a.sum() else a)
        tab = np.array(tab)
        vs.append(cramers_v(tab)[4])
        for a in range(4):
            for b in range(a + 1, 4):
                pair_v.setdefault(f"{LANGS[a]}-{LANGS[b]}", []).append(cramers_v(tab[[a, b]])[4])

    ci = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]
    res = {"replicates": B,
           "unit": "idiom (with >=1 classified entity), resampled within language",
           "types": types,
           "cramers_v_ci95": ci(vs),
           "pairwise_cramers_v_ci95": {k: ci(v) for k, v in pair_v.items()},
           "share_ci95": {l: {t: ci(np.array(shares[l])[:, j]) for j, t in enumerate(types)}
                          for l in LANGS},
           "within_abstract_share_ci95": {l: {t: ci(np.array(within[l])[:, j])
                                              for j, t in enumerate(SUBTYPES)} for l in LANGS}}
    dump(res, "entity_subtypology_bootstrap.json")
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
