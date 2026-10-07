#!/usr/bin/env python3
"""Sensitivity of the en-zh meaning pairing to its two cosine thresholds (paper: 0.70 / 0.95).

Two sweeps, both from the precomputed figurative-meaning embeddings:

(a) pairing threshold. Two idioms are paired when the maximum cosine over their
    figurative meanings reaches T. For T in 0.65..0.90 we report the number of pairs,
    how many idioms they cover, and the shared-entity rate computed exactly as in
    paper_writing/code/shared_entity_rate.py (high-recall GPT translation lexicon, so
    an upper bound).

(b) attachment threshold. Holding the pairing threshold at 0.70, each pair in file
    order anchors a bilingual cluster, which absorbs the still-unused same-language
    idioms whose meaning reaches A of the anchor's *matched* meaning. This is a
    vectorized reimplementation of
    culture.analysis.intra_lingual_idiom_clusters.build_combined_clusters.
    For A in 0.90..0.98 we report the cluster count and the size distribution.

  PYTHONPATH=src python src/culture/analysis/v2/pairing_threshold_sensitivity.py
"""
import json
import os
from collections import Counter

import numpy as np

REPO = os.environ.get(
    "CULTURE_REPO",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage")
DATA = os.path.join(REPO, "culture/data/idioms")
OUT = os.path.join(REPO, "docs/paper_stats/analysis_v2")
SLOTS = {"something", "someone", "thing", "person", "place", "way"}
PAIR_T = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
ATTACH_T = [0.90, 0.93, 0.95, 0.97, 0.98]
BASE_PAIR_T = 0.70


def load_emb(lang):
    p = os.path.join(DATA, lang, "figurative_embeddings")
    meta = json.load(open(p + "_meta.json"))
    with np.load(p + ".npz") as d:
        emb = d["embeddings"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-6)
    owner = np.empty(len(emb), dtype=np.int32)
    for i, m in enumerate(meta):
        owner[m["embedding_start_idx"]:m["embedding_end_idx"]] = i
    return meta, emb, owner


def best_cross_sims(en_emb, en_owner, zh_emb, zh_owner, n_en, n_zh, floor, chunk=512):
    """Max cosine per (en idiom, zh idiom) pair that reaches `floor`.

    Returns (keys, vals) with key = en_idiom * n_zh + zh_idiom, sorted by key."""
    keys, vals = [], []
    for s in range(0, len(en_emb), chunk):
        block = en_emb[s:s + chunk] @ zh_emb.T          # (chunk, n_zh_meanings)
        rows, cols = np.nonzero(block >= floor)
        if len(rows) == 0:
            continue
        keys.append(en_owner[s + rows].astype(np.int64) * n_zh + zh_owner[cols])
        vals.append(block[rows, cols])
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    uniq, start = np.unique(keys, return_index=True)
    return uniq, np.maximum.reduceat(vals, start)


def shared_entity_rate(pairs):
    """pairs: iterable of dicts with zh_entities / en_entities. Returns the paper's numbers."""
    z2e = json.load(open(os.path.join(DATA, "cross_lingual_analysis/translations_zh_to_en.json")))
    e2z = json.load(open(os.path.join(DATA, "cross_lingual_analysis/translations_en_to_zh.json")))
    low = lambda xs: {x.lower() for x in xs}
    n_all = n_both = n_cov = n_share = 0
    for p in pairs:
        n_all += 1
        ze = set(p.get("zh_entities") or [])
        ee = low(p.get("en_entities") or []) - SLOTS
        if not ze or not ee:
            continue
        n_both += 1
        n_cov += any(z in z2e for z in ze) or any(e in e2z for e in ee)
        n_share += any(low(z2e.get(z, [])) & ee for z in ze) or any(set(e2z.get(e, [])) & ze for e in ee)
    return {"n_pairs": n_all, "n_both_entities": n_both, "n_translatable": n_cov,
            "n_share_entity": n_share,
            "share_pct": round(100 * n_share / max(n_cov, 1), 1)}


def sweep_pairing(en, zh):
    """(a) pairing threshold. Below 0.70 the stored pair file does not reach, so recompute."""
    en_meta, en_emb, en_owner = en
    zh_meta, zh_emb, zh_owner = zh
    stored = [json.loads(l) for l in open(os.path.join(DATA, "cross_lingual_pairs.jsonl"))]
    print(f"stored pairs: {len(stored)}")

    floor = min(PAIR_T)
    print(f"recomputing cross-lingual max cosines at floor {floor} ...", flush=True)
    keys, best = best_cross_sims(en_emb, en_owner, zh_emb, zh_owner, len(en_meta), len(zh_meta), floor)
    n_zh = len(zh_meta)
    ei_all, zi_all = (keys // n_zh).astype(np.int32), (keys % n_zh).astype(np.int32)
    print(f"  {len(keys)} idiom pairs at >= {floor}", flush=True)

    en_ent = [m.get("entities") or [] for m in en_meta]
    zh_ent = [m.get("entities") or [] for m in zh_meta]
    rows = []
    for t in PAIR_T:
        m = best >= t
        ei, zi = ei_all[m], zi_all[m]
        recs = [{"en_entities": en_ent[a], "zh_entities": zh_ent[b]}
                for a, b in zip(ei.tolist(), zi.tolist())]
        r = shared_entity_rate(recs)
        r.update({"threshold": t,
                  "n_en_idioms": int(len(np.unique(ei))),
                  "n_zh_idioms": int(len(np.unique(zi)))})
        rows.append(r)
        print(f"  T={t:.2f}  pairs={r['n_pairs']:7d}  en={r['n_en_idioms']:6d} zh={r['n_zh_idioms']:6d}  "
              f"share_entity={r['share_pct']:.1f}% of {r['n_translatable']}", flush=True)
    recomputed_at_070 = next(r for r in rows if r["threshold"] == 0.70)
    stored_rate = shared_entity_rate(stored)
    return {"sweep": rows,
            "stored_pair_file": {**stored_rate, "threshold": 0.70},
            "recompute_vs_stored_note":
                f"recomputed {recomputed_at_070['n_pairs']} pairs at 0.70 vs {len(stored)} in "
                f"cross_lingual_pairs.jsonl; the stored file was built with the same rule, so the "
                f"two should agree up to float16 rounding of the released embeddings"}


def cluster_once(en, zh, pairs, attach):
    en_meta, en_emb, _ = en
    zh_meta, zh_emb, _ = zh
    en_ix = {m["idiom"]: i for i, m in enumerate(en_meta)}
    zh_ix = {m["idiom"]: i for i, m in enumerate(zh_meta)}
    en_span = np.array([[m["embedding_start_idx"], m["embedding_end_idx"]] for m in en_meta])
    zh_span = np.array([[m["embedding_start_idx"], m["embedding_end_idx"]] for m in zh_meta])
    # spans are contiguous and in idiom order, so a max-reduceat gives the per-idiom maximum
    assert np.array_equal(en_span[:, 0], np.r_[0, np.cumsum(en_span[:, 1] - en_span[:, 0])[:-1]])
    assert np.array_equal(zh_span[:, 0], np.r_[0, np.cumsum(zh_span[:, 1] - zh_span[:, 0])[:-1]])
    en_starts, zh_starts = en_span[:, 0], zh_span[:, 0]

    def meaning_idx(meta, span, target):
        ms = meta.get("figurative_meanings") or []
        for i, m in enumerate(ms):
            if m == target:
                return span[0] + i
        for i, m in enumerate(ms):
            if target and (target in m or m in target):
                return span[0] + i
        return span[0] if len(ms) else -1

    used_en, used_zh = np.zeros(len(en_meta), bool), np.zeros(len(zh_meta), bool)
    sizes = []
    for p in pairs:
        ei, zi = en_ix.get(p["en_idiom"]), zh_ix.get(p["zh_idiom"])
        if ei is None or zi is None or used_en[ei] or used_zh[zi]:
            continue
        em = meaning_idx(en_meta[ei], en_span[ei], p.get("en_matched_meaning", ""))
        zm = meaning_idx(zh_meta[zi], zh_span[zi], p.get("zh_matched_meaning", ""))
        if em < 0 or zm < 0:
            continue
        e_hit = np.maximum.reduceat(en_emb @ en_emb[em], en_starts) >= attach
        z_hit = np.maximum.reduceat(zh_emb @ zh_emb[zm], zh_starts) >= attach
        e_sel = np.flatnonzero(e_hit & ~used_en)
        z_sel = np.flatnonzero(z_hit & ~used_zh)
        e_sel = np.union1d(e_sel, [ei])
        z_sel = np.union1d(z_sel, [zi])
        used_en[e_sel] = True
        used_zh[z_sel] = True
        sizes.append((len(e_sel), len(z_sel)))
    n = len(sizes)
    c = Counter(sizes)
    return {"attach_threshold": attach, "n_clusters": n,
            "pct_1en_1zh": round(100 * c[(1, 1)] / max(n, 1), 1),
            "pct_ge5_idioms": round(100 * sum(1 for a, b in sizes if a + b >= 5) / max(n, 1), 1),
            "mean_en_per_cluster": round(float(np.mean([a for a, _ in sizes])), 2) if n else None,
            "mean_zh_per_cluster": round(float(np.mean([b for _, b in sizes])), 2) if n else None,
            "n_en_idioms": int(used_en.sum()), "n_zh_idioms": int(used_zh.sum())}


def main():
    en = load_emb("en")
    zh = load_emb("zh")
    res = {"pairing_threshold": sweep_pairing(en, zh)}

    pairs = [json.loads(l) for l in open(os.path.join(DATA, "cross_lingual_pairs.jsonl"))]
    pairs = [p for p in pairs if (p.get("similarity") or 0) >= BASE_PAIR_T]
    print(f"\nclustering {len(pairs)} pairs (pairing threshold {BASE_PAIR_T})", flush=True)
    res["attach_threshold"] = []
    for a in ATTACH_T:
        r = cluster_once(en, zh, pairs, a)
        res["attach_threshold"].append(r)
        print("  " + json.dumps(r), flush=True)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "pairing_threshold_sensitivity.json")
    json.dump(res, open(p, "w"), ensure_ascii=False, indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
