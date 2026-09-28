#!/usr/bin/env python3
"""Task 1 robustness: idiom-level bootstrap CIs for the type shares and Cramer's V.

Mentions from the same idiom (and repeated mentions of the same entity) are not independent, so
the Pearson chi-square p-value overstates significance. We resample IDIOMS with replacement within
each language (1,000 replicates, seed 0) and recompute mention shares and Cramer's V.

    PYTHONPATH=src python src/culture/analysis/v2/entity_typology_bootstrap.py
"""
import json
import os

import numpy as np

from common import OUT, dump, load_kb
from entity_typology import LANGS, TYPES, cramers_v


def main():
    rng = np.random.default_rng(0)
    per_lang = {}
    for lang in LANGS:
        labs = json.load(open(os.path.join(OUT, f"entity_typology_labels_{lang}.json")))
        rows = load_kb(lang)
        M = np.zeros((len(rows), len(TYPES)))
        for i, r in enumerate(rows):
            for e in r["entities"]:
                t = labs.get(e, {}).get("type")
                if t in TYPES:
                    M[i, TYPES.index(t)] += 1
        per_lang[lang] = M[M.sum(1) > 0]
    B = 1000
    shares = {l: [] for l in LANGS}
    vs, pair_v = [], {}
    for _ in range(B):
        tab = []
        for l in LANGS:
            M = per_lang[l]
            s = M[rng.integers(0, len(M), len(M))].sum(0)
            tab.append(s)
            shares[l].append(s / s.sum())
        tab = np.array(tab)
        vs.append(cramers_v(tab)[4])
        for a in range(4):
            for b in range(a + 1, 4):
                pair_v.setdefault(f"{LANGS[a]}-{LANGS[b]}", []).append(cramers_v(tab[[a, b]])[4])
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]
    res = {"replicates": B, "unit": "idiom (with >=1 classified entity), resampled within language",
           "cramers_v_ci95": ci(vs),
           "pairwise_cramers_v_ci95": {k: ci(v) for k, v in pair_v.items()},
           "share_ci95": {l: {t: ci(np.array(shares[l])[:, j]) for j, t in enumerate(TYPES)} for l in LANGS}}
    dump(res, "entity_typology_bootstrap.json")
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
