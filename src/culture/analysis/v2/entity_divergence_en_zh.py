#!/usr/bin/env python3
"""Task 2: prompt-independent divergence score for same-entity-different-meaning (en-zh).

Uses the 516 entity analyses of docs/data/tab1_entity_meanings.json (the paper reports 515).
Idiom sets:
  as_analyzed : exactly the (<=20 per side) idioms the GPT-5.2 summary was based on, keeping
                those with >=1 figurative meaning;
  full        : all idioms of the paper's figurative-only KBs whose entity list contains any
                entity of the entity's cluster (en_cluster / zh_cluster of
                cultural_analysis_clusters.json), i.e. the original retrieval without the cap.
Representations:
  gloss  : Qwen3-Embedding-0.6B embedding of an LLM English paraphrase of the idiom's figurative
           meaning (both languages paraphrased by the same prompt) -- primary;
  native : mean of Qwen3-Embedding-0.6B embeddings of the idiom's figurative-meaning strings in
           the source language (cross-lingual embedding) -- robustness.

    PYTHONPATH=src python src/culture/analysis/v2/entity_divergence_en_zh.py
"""
import csv
import json
import os
import random

import numpy as np
from scipy.stats import spearmanr

from common import OUT, REPO, DATA, dump, embed, entity_index, flatten, load_kb, norm_entity
from divergence import full_set_calibration, size_matched, summarize_size_matched
from gloss import gloss

CASES = ["dog", "dragon", "heart", "red", "turtle", "moon", "phoenix"]
MIN_N = 5


def idiom_vectors_native(items):
    """items: list of (idiom, fig list) -> dict idiom -> unit vector (mean of meaning embeddings)."""
    texts, owner = [], []
    for idiom, fig in items:
        for f in fig:
            texts.append(f)
            owner.append(idiom)
    E = embed(texts)
    acc = {}
    for v, o in zip(E, owner):
        acc.setdefault(o, []).append(v)
    return {o: (lambda m: m / np.linalg.norm(m))(np.mean(vs, 0)) for o, vs in acc.items()}


def idiom_vectors_gloss(lang, items):
    g = gloss(lang, items)
    keys = [k for k in g]
    E = embed([g[k] for k in keys])
    return {k: v for k, v in zip(keys, E)}, g


def stack(vecs, idioms):
    xs = [vecs[i] for i in idioms if i in vecs]
    return np.stack(xs) if xs else np.zeros((0, 1024), dtype="float32")


def main():
    random.seed(0)
    tab1 = json.load(open(os.path.join(REPO, "docs/data/tab1_entity_meanings.json")))
    pairs = tab1["en_to_zh"] + tab1["zh_to_en"]
    clusters = json.load(open(os.path.join(DATA, "cross_lingual_analysis/cultural_analysis_clusters.json")))
    ck = {(c["en_cluster"]["primary_entity"], c["zh_cluster"]["primary_entity"], c["translation_direction"]): c
          for c in clusters}

    # ---------------- idiom sets
    en_kb, zh_kb = load_kb("en"), load_kb("zh")
    en_idx, zh_idx = entity_index(en_kb), entity_index(zh_kb)
    recs = []
    for p in pairs:
        c = ck[(p["entity_en"], p["entity_zh"], p["direction"])]
        en_ents = {norm_entity(e, "en") for e in c["en_cluster"]["cluster_entities"]} - {None}
        zh_ents = set(c["zh_cluster"]["cluster_entities"])
        full_en = sorted({en_kb[j]["idiom"] for e in en_ents for j in en_idx.get(e, []) if en_kb[j]["fig"]})
        full_zh = sorted({zh_kb[j]["idiom"] for e in zh_ents for j in zh_idx.get(e, []) if zh_kb[j]["fig"]})
        aa_en = [(i["idiom"], flatten(i["figurative_meanings"])) for i in p["idioms_en"]]
        aa_zh = [(i["idiom"], flatten(i["figurative_meanings"])) for i in p["idioms_zh"]]
        recs.append({
            "entity_en": p["entity_en"], "entity_zh": p["entity_zh"], "direction": p["direction"],
            "en_cluster": sorted(en_ents), "zh_cluster": sorted(zh_ents),
            "n_en_analyzed": len(aa_en), "n_zh_analyzed": len(aa_zh),
            "aa_en": [x for x in aa_en if x[1]], "aa_zh": [x for x in aa_zh if x[1]],
            "full_en": full_en, "full_zh": full_zh,
            "n_shared": len(p["shared_meanings"]), "n_en_unique": len(p["en_unique_aspects"]),
            "n_zh_unique": len(p["zh_unique_aspects"]),
            "n_en_primary": len(p["en_primary_meanings"]), "n_zh_primary": len(p["zh_primary_meanings"]),
        })
    for r in recs:
        r["n_en"] = len(r["aa_en"])
        r["n_zh"] = len(r["aa_zh"])
        r["n_en_full"] = len(r["full_en"])
        r["n_zh_full"] = len(r["full_zh"])
    qual = [i for i, r in enumerate(recs) if r["n_en"] >= MIN_N and r["n_zh"] >= MIN_N]

    # ---------------- vectors
    en_items = {x[0]: x[1] for r in recs for x in r["aa_en"]}
    zh_items = {x[0]: x[1] for r in recs for x in r["aa_zh"]}
    gv_en, g_en = idiom_vectors_gloss("en", list(en_items.items()))
    gv_zh, g_zh = idiom_vectors_gloss("zh", list(zh_items.items()))
    en_fig_full = {r_["idiom"]: r_["fig"] for r_ in en_kb}
    zh_fig_full = {r_["idiom"]: r_["fig"] for r_ in zh_kb}
    nat_en_items = dict(en_items)
    nat_zh_items = dict(zh_items)
    for r in recs:
        for i in r["full_en"]:
            nat_en_items.setdefault(i, en_fig_full[i])
        for i in r["full_zh"]:
            nat_zh_items.setdefault(i, zh_fig_full[i])
    nv_en = idiom_vectors_native(list(nat_en_items.items()))
    nv_zh = idiom_vectors_native(list(nat_zh_items.items()))
    # full-retrieval sets use the KB's own meanings for idioms (same idiom string -> same vector)
    print("glossed", len(g_en), "/", len(en_items), len(g_zh), "/", len(zh_items))

    settings = {
        "gloss_as_analyzed": ([stack(gv_en, [x[0] for x in r["aa_en"]]) for r in recs],
                              [stack(gv_zh, [x[0] for x in r["aa_zh"]]) for r in recs]),
        "native_as_analyzed": ([stack(nv_en, [x[0] for x in r["aa_en"]]) for r in recs],
                               [stack(nv_zh, [x[0] for x in r["aa_zh"]]) for r in recs]),
        "native_full": ([stack(nv_en, r["full_en"]) for r in recs],
                        [stack(nv_zh, r["full_zh"]) for r in recs]),
    }
    result = {"n_entity_analyses": len(recs), "min_idioms_per_side": MIN_N,
              "n_qualifying_as_analyzed": len(qual),
              "n_qualifying_full": sum(1 for r in recs if r["n_en_full"] >= MIN_N and r["n_zh_full"] >= MIN_N),
              "mean_idioms_as_analyzed_all": {"en": round(np.mean([r["n_en_analyzed"] for r in recs]), 2),
                                              "zh": round(np.mean([r["n_zh_analyzed"] for r in recs]), 2)},
              "mean_idioms_with_fig_all": {"en": round(np.mean([r["n_en"] for r in recs]), 2),
                                           "zh": round(np.mean([r["n_zh"] for r in recs]), 2)},
              "n_glossed": {"en": len(g_en), "en_total": len(en_items), "zh": len(g_zh), "zh_total": len(zh_items)},
              "settings": {}}
    cal_primary = None
    for name, (A, B) in settings.items():
        if name == "native_full":
            q = [i for i in range(len(recs)) if len(A[i]) >= MIN_N and len(B[i]) >= MIN_N]
        else:
            q = [i for i in qual if len(A[i]) >= MIN_N and len(B[i]) >= MIN_N]
        Aq = [A[i] for i in q]
        Bq = [B[i] for i in q]
        per = size_matched(Aq, Bq, k=5, reps=30)
        summ = summarize_size_matched(per)
        cal = full_set_calibration(A, B, q)
        pct = np.array([cal[i]["percentile"] for i in q])
        summ["full_set_calibration"] = {
            "n_entities": len(q),
            "mean_percentile": round(float(pct.mean()), 4),
            "median_percentile": round(float(np.median(pct)), 4),
            "frac_true_pair_top1_a2b": round(float(np.mean([cal[i]["rank_a2b"] == 1 for i in q])), 4),
            "frac_percentile_le_0.05": round(float((pct <= 0.05).mean()), 4),
            "frac_percentile_le_0.10": round(float((pct <= 0.10).mean()), 4),
            "frac_percentile_ge_0.50": round(float((pct >= 0.50).mean()), 4),
            "mean_centroid_div_true": round(float(np.mean([cal[i]["centroid_div"] for i in q])), 4),
            "mean_centroid_div_random": round(float(np.mean([cal[i]["mean_div_random"] for i in q])), 4),
        }
        # agreement with the prompt-based counts
        shared_ratio = [recs[i]["n_shared"] / max(1, recs[i]["n_shared"] + recs[i]["n_en_unique"] + recs[i]["n_zh_unique"]) for i in q]
        rho, p = spearmanr([cal[i]["centroid_div"] for i in q], shared_ratio)
        rho2, p2 = spearmanr([cal[i]["percentile"] for i in q], shared_ratio)
        summ["spearman_vs_llm_shared_ratio"] = {"centroid_div": [round(float(rho), 3), float(p)],
                                                "percentile": [round(float(rho2), 3), float(p2)]}
        result["settings"][name] = summ
        for i in q:
            recs[i].setdefault("scores", {})[name] = {k: round(v, 4) if isinstance(v, float) else v for k, v in cal[i].items()}
        if name == "gloss_as_analyzed":
            cal_primary = (q, cal)

    # ---------------- rank entities (primary setting, qualifying entities)
    q, cal = cal_primary
    order = sorted(q, key=lambda i: -cal[i]["centroid_div"])
    row = lambda i: {"entity_en": recs[i]["entity_en"], "entity_zh": recs[i]["entity_zh"],
                     "n_en": recs[i]["n_en"], "n_zh": recs[i]["n_zh"],
                     "centroid_div": round(cal[i]["centroid_div"], 4), "percentile": round(cal[i]["percentile"], 3),
                     "rank_en2zh": cal[i]["rank_a2b"],
                     "llm_shared": recs[i]["n_shared"], "llm_unique": recs[i]["n_en_unique"] + recs[i]["n_zh_unique"]}
    result["top15_most_divergent"] = [row(i) for i in order[:15]]
    result["top15_least_divergent"] = [row(i) for i in order[::-1][:15]]
    order_p = sorted(q, key=lambda i: -cal[i]["percentile"])
    result["top15_most_divergent_by_percentile"] = [row(i) for i in order_p[:15]]

    # ---------------- the paper's prompt-based stats on all vs qualifying
    def llm_stats(ids):
        return {"n": len(ids),
                "mean_shared": round(float(np.mean([recs[i]["n_shared"] for i in ids])), 2),
                "mean_en_unique": round(float(np.mean([recs[i]["n_en_unique"] for i in ids])), 2),
                "mean_zh_unique": round(float(np.mean([recs[i]["n_zh_unique"] for i in ids])), 2),
                "mean_unique_total": round(float(np.mean([recs[i]["n_en_unique"] + recs[i]["n_zh_unique"] for i in ids])), 2),
                "frac_unique_gt_shared": round(float(np.mean([recs[i]["n_en_unique"] + recs[i]["n_zh_unique"] > recs[i]["n_shared"] for i in ids])), 4),
                "frac_en_unique_gt_shared": round(float(np.mean([recs[i]["n_en_unique"] > recs[i]["n_shared"] for i in ids])), 4),
                "frac_each_side_unique_ge_shared": round(float(np.mean([min(recs[i]["n_en_unique"], recs[i]["n_zh_unique"]) >= recs[i]["n_shared"] for i in ids])), 4),
                "shared_count_distribution": {str(k): int(v) for k, v in zip(*np.unique([recs[i]["n_shared"] for i in ids], return_counts=True))},
                "mean_idioms_analyzed": {"en": round(float(np.mean([recs[i]["n_en_analyzed"] for i in ids])), 2),
                                         "zh": round(float(np.mean([recs[i]["n_zh_analyzed"] for i in ids])), 2)}}
    result["llm_counts_all"] = llm_stats(list(range(len(recs))))
    result["llm_counts_qualifying"] = llm_stats(qual)

    # ---------------- idiom counts for the entity-case table
    cases = {}
    for i, r in enumerate(recs):
        if r["entity_en"] in CASES:
            cases[r["entity_en"]] = {"entity_zh": r["entity_zh"], "direction": r["direction"],
                                     "n_analyzed_en": r["n_en_analyzed"], "n_analyzed_zh": r["n_zh_analyzed"],
                                     "n_with_fig_en": r["n_en"], "n_with_fig_zh": r["n_zh"],
                                     "n_full_kb_en": r["n_en_full"], "n_full_kb_zh": r["n_zh_full"],
                                     "en_cluster": r["en_cluster"], "zh_cluster": r["zh_cluster"],
                                     "qualifies_ge5": i in qual,
                                     "scores": r.get("scores", {})}
    result["entity_cases"] = cases
    dump(result, "entity_divergence_en_zh.json")

    with open(os.path.join(OUT, "entity_divergence_en_zh_per_entity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity_en", "entity_zh", "direction", "n_en", "n_zh", "n_en_full", "n_zh_full", "qualifies",
                    "gloss_centroid_div", "gloss_percentile", "native_centroid_div", "native_percentile",
                    "llm_shared", "llm_en_unique", "llm_zh_unique"])
        for i, r in enumerate(recs):
            s = r.get("scores", {})
            g, n = s.get("gloss_as_analyzed", {}), s.get("native_as_analyzed", {})
            w.writerow([r["entity_en"], r["entity_zh"], r["direction"], r["n_en"], r["n_zh"], r["n_en_full"],
                        r["n_zh_full"], int(i in qual), g.get("centroid_div"), g.get("percentile"),
                        n.get("centroid_div"), n.get("percentile"), r["n_shared"], r["n_en_unique"], r["n_zh_unique"]])
    dump({k: v for k, v in g_zh.items()}, "gloss_cache_zh_tab1.json")
    print(json.dumps({k: result[k] for k in ("n_qualifying_as_analyzed", "llm_counts_all", "llm_counts_qualifying")}, ensure_ascii=False, indent=1))
    print(json.dumps(result["settings"]["gloss_as_analyzed"], indent=1)[:3000])


if __name__ == "__main__":
    main()
