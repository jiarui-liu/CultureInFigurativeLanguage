#!/usr/bin/env python3
"""Task 4 extension: direct (non-English-anchored) same-entity divergence for zh-hi, zh-ar, hi-ar,
under the identical protocol as entity_divergence_multi.py (which only covers en-X pairs).

Reuses the entity -> English-anchor translations already computed by entity_divergence_multi.py
(docs/paper_stats/analysis_v2/entity_translations_{lang}_en.json) to find cross-language entity
correspondence (two source languages sharing the same English anchor noun) with NO new translation
LLM calls; only gloss() (paraphrase) and embed() need the LLM / embedding model.

    PYTHONPATH=src python src/culture/analysis/v2/entity_divergence_cross.py
"""
import json
import os
import random
from collections import defaultdict

import numpy as np

from common import OUT, dump, embed, entity_counter, entity_index, load_kb
from divergence import full_set_calibration, size_matched, summarize_size_matched
from entity_divergence_en_zh import idiom_vectors_native, stack
from gloss import gloss

CAP = 20
MIN_N = 5
PAIRS = [("zh", "hi"), ("zh", "ar"), ("hi", "ar")]


def load_translations(lang):
    p = os.path.join(OUT, f"entity_translations_{lang}_en.json")
    with open(p, encoding="utf-8") as f:
        tr = json.load(f)
    groups = defaultdict(list)
    for e, t in tr.items():
        if t and t.get("en") and t["en"] != "none":
            groups[t["en"]].append(e)
    return groups


def main():
    rng = random.Random(0)
    kbs, idxs, figs = {}, {}, {}
    for lang in ("zh", "hi", "ar"):
        kb = load_kb(lang)
        kbs[lang] = kb
        idxs[lang] = entity_index(kb)
        figs[lang] = {r["idiom"]: r["fig"] for r in kb}

    results = {"method": __doc__, "pairs": {}}
    for a, b in PAIRS:
        ga, gb = load_translations(a), load_translations(b)
        shared = sorted(set(ga) & set(gb))
        recs = []
        for en_ent in shared:
            a_ids = sorted({j for e in ga[en_ent] for j in idxs[a].get(e, []) if kbs[a][j]["fig"]})
            b_ids = sorted({j for e in gb[en_ent] for j in idxs[b].get(e, []) if kbs[b][j]["fig"]})
            if not a_ids or not b_ids:
                continue
            recs.append({"entity_en": en_ent, "entities_a": ga[en_ent], "entities_b": gb[en_ent],
                         "n_a_full": len(a_ids), "n_b_full": len(b_ids),
                         "a_sample": [kbs[a][j]["idiom"] for j in (rng.sample(a_ids, CAP) if len(a_ids) > CAP else a_ids)],
                         "b_sample": [kbs[b][j]["idiom"] for j in (rng.sample(b_ids, CAP) if len(b_ids) > CAP else b_ids)]})
        key = f"{a}-{b}"
        results["pairs"][key] = {"shared_english_anchors": len(shared), "matched_entities": len(recs)}
        if not recs:
            results["pairs"][key]["note"] = "no shared entities with >=1 idiom on both sides; skipped"
            print(key, "skipped: no shared entities")
            continue

        a_items = {i: figs[a][i] for r in recs for i in r["a_sample"]}
        b_items = {i: figs[b][i] for r in recs for i in r["b_sample"]}
        g_a = gloss(a, list(a_items.items()))
        g_b = gloss(b, list(b_items.items()))
        gk_a = list(g_a)
        gv_a = dict(zip(gk_a, embed([g_a[k] for k in gk_a])))
        gk_b = list(g_b)
        gv_b = dict(zip(gk_b, embed([g_b[k] for k in gk_b])))
        nv_a = idiom_vectors_native(list(a_items.items()))
        nv_b = idiom_vectors_native(list(b_items.items()))

        out = results["pairs"][key]
        for name, (VA, VB) in {"gloss": (gv_a, gv_b), "native": (nv_a, nv_b)}.items():
            A = [stack(VA, r["a_sample"]) for r in recs]
            B = [stack(VB, r["b_sample"]) for r in recs]
            q = [i for i in range(len(recs)) if len(A[i]) >= MIN_N and len(B[i]) >= MIN_N]
            if not q:
                out[name] = {"note": "no entity reached MIN_N=5 idioms on both sides"}
                continue
            per = size_matched([A[i] for i in q], [B[i] for i in q], k=5, reps=30)
            summ = summarize_size_matched(per)
            cal = full_set_calibration(A, B, q)
            pct = np.array([cal[i]["percentile"] for i in q])
            summ["full_set_calibration"] = {
                "n_entities": len(q), "mean_percentile": round(float(pct.mean()), 4),
                "median_percentile": round(float(np.median(pct)), 4),
                "frac_true_pair_top1_a2b": round(float(np.mean([cal[i]["rank_a2b"] == 1 for i in q])), 4),
                "mean_centroid_div_true": round(float(np.mean([cal[i]["centroid_div"] for i in q])), 4),
                "mean_centroid_div_random": round(float(np.mean([cal[i]["mean_div_random"] for i in q])), 4)}
            out[name] = summ
            out["n_qualifying_ge5"] = len(q)
        print(key, json.dumps(out, ensure_ascii=False)[:1500])
    dump(results, "entity_divergence_cross.json")


if __name__ == "__main__":
    main()
