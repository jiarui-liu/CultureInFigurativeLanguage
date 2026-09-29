#!/usr/bin/env python3
"""Task 3: same-meaning-different-entity (en-zh) checks.

(a) cluster-size distribution of the bilingual meaning clusters (docs/data/tab2_semantic_clusters.json);
(b) basic pair statistics of culture/data/idioms/cross_lingual_pairs.jsonl;
(c) re-computation of the shared-entity rate (paper: 5.5% overall, 7.5% at cos>=0.75), with the
    exact logic of paper_writing/code/shared_entity_rate.py, and per bin;
(d) LLM-judged precision of the pairing: 200 pairs, 50 per similarity bin, judged yes/partial/no
    by a primary LLM judge and by independent second judges. This is an LLM proxy, NOT human annotation.

    PYTHONPATH=src python src/culture/analysis/v2/meaning_clusters.py
"""
import csv
import json
import os
import random
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

from common import DATA, OUT, REPO, dump
from culture.bidirectional.llm_api import parse_json
from local_llm import NAMES, PRIMARY, SECOND, THIRD, generate

BINS = [(0.70, 0.72), (0.72, 0.75), (0.75, 0.80), (0.80, 1.01)]
BIN_NAMES = ["0.70-0.72", "0.72-0.75", "0.75-0.80", ">=0.80"]
SLOTS = {"something", "someone", "thing", "person", "place", "way"}

JUDGE_PROMPT = """You are checking an automatic alignment of English and Chinese idioms by meaning.
For each pair below, decide whether the two FIGURATIVE meanings express the same idea.
- "yes": same idea; one could translate one idiom by the other in most contexts.
- "partial": clearly related or overlapping, but differing in scope, intensity, evaluation, or a key component.
- "no": different ideas.

{items}

Return a JSON object mapping each pair id (as a string) to one of "yes", "partial", "no". Return only the JSON object."""


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def bin_of(s):
    for i, (a, b) in enumerate(BINS):
        if a <= s < b:
            return i
    return None


def judge(sample, model, tag, per_call=10):
    batches = [sample[i:i + per_call] for i in range(0, len(sample), per_call)]
    prompts = []
    for b in batches:
        lines = []
        for i, p in enumerate(b, 1):
            lines.append(f"Pair {i}\n  English idiom: {p['en_idiom']}\n  English meaning: {p['en_matched_meaning']}\n"
                         f"  Chinese idiom: {p['zh_idiom']}\n  Chinese meaning: {p['zh_matched_meaning']}")
        prompts.append(JUDGE_PROMPT.format(items="\n\n".join(lines)))
    if model.startswith("openrouter:"):
        from culture.bidirectional.llm_api import complete_many
        outs = complete_many(prompts, workers=1, model=model.split(":", 1)[1], provider="openrouter",
                             json_mode=True, tag=tag, max_tokens=8000)
    else:
        outs = generate(prompts, tag=tag, model=model, max_tokens=1000)
    labels = []
    for b, o in zip(batches, outs):
        d = parse_json(o) or {}
        if isinstance(d, list) and len(d) == len(b):
            d = {str(i): x for i, x in enumerate(d, 1)}
        if not isinstance(d, dict):
            d = {}
        for i, _ in enumerate(b, 1):
            v = d.get(str(i), d.get(f"Pair {i}", d.get(f"pair {i}")))
            v = v.strip().lower() if isinstance(v, str) else None
            labels.append(v if v in ("yes", "partial", "no") else None)
    return labels


def precision_table(sample, labels, pop_share):
    res = {}
    overall = {"strict": 0.0, "lenient": 0.0}
    for bi, name in enumerate(BIN_NAMES):
        ls = [l for p, l in zip(sample, labels) if p["bin"] == bi and l is not None]
        n = len(ls)
        c = Counter(ls)
        res[name] = {"n": n, "yes": c["yes"], "partial": c["partial"], "no": c["no"],
                     "precision_strict": round(c["yes"] / n, 3) if n else None,
                     "precision_strict_ci95": wilson(c["yes"], n),
                     "precision_lenient": round((c["yes"] + c["partial"]) / n, 3) if n else None,
                     "precision_lenient_ci95": wilson(c["yes"] + c["partial"], n)}
        if n:
            overall["strict"] += pop_share[name] * c["yes"] / n
            overall["lenient"] += pop_share[name] * (c["yes"] + c["partial"]) / n
    res["population_weighted"] = {k: round(v, 3) for k, v in overall.items()}
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="both", choices=["primary", "second", "both"])
    ap.add_argument("--or_judge", action="store_true", help="add nemotron-3-ultra (OpenRouter free tier) as a judge")
    args = ap.parse_args()
    # ---------------- (a) cluster sizes
    tab2 = json.load(open(os.path.join(REPO, "docs/data/tab2_semantic_clusters.json")))
    en = np.array([c["en_idiom_count"] for c in tab2])
    zh = np.array([c["zh_idiom_count"] for c in tab2])
    cap = lambda x: np.minimum(x, 10)
    joint = Counter(zip(cap(en).tolist(), cap(zh).tolist()))
    n = len(tab2)
    sizes = {
        "n_clusters": n,
        "n_en_idioms_total": int(en.sum()), "n_zh_idioms_total": int(zh.sum()),
        "frac_1en_1zh": round(float(((en == 1) & (zh == 1)).mean()), 4),
        "frac_single_on_at_least_one_side": round(float(((en == 1) | (zh == 1)).mean()), 4),
        "frac_ge2_both_sides": round(float(((en >= 2) & (zh >= 2)).mean()), 4),
        "n_ge2_both_sides": int(((en >= 2) & (zh >= 2)).sum()),
        "frac_ge2_at_least_one_side": round(float(((en >= 2) | (zh >= 2)).mean()), 4),
        "frac_en_eq1": round(float((en == 1).mean()), 4), "frac_zh_eq1": round(float((zh == 1).mean()), 4),
        "frac_total_ge5": round(float(((en + zh) >= 5).mean()), 4),
        "n_ratio_ge3_and_larger_side_ge3": int((((en >= 3 * zh) | (zh >= 3 * en)) & (np.maximum(en, zh) >= 3)).sum()),
        "en_size_hist": {str(k): int(v) for k, v in sorted(Counter(cap(en).tolist()).items())},
        "zh_size_hist": {str(k): int(v) for k, v in sorted(Counter(cap(zh).tolist()).items())},
        "max_en": int(en.max()), "max_zh": int(zh.max()),
        "mean_en": round(float(en.mean()), 3), "mean_zh": round(float(zh.mean()), 3),
        "note": "sizes capped at 10 (10 = 10+) in histograms",
    }
    with open(os.path.join(OUT, "cluster_size_joint_hist.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["en_idioms(10=10+)", "zh_idioms(10=10+)", "n_clusters"])
        for (a, b), v in sorted(joint.items()):
            w.writerow([a, b, v])

    # ---------------- (b) pairs + (c) shared-entity rate
    pairs = [json.loads(l) for l in open(os.path.join(DATA, "cross_lingual_pairs.jsonl"))]
    sims = np.array([p["similarity"] for p in pairs])
    pstats = {"n_pairs": len(pairs), "n_zh_idioms": len({p["zh_idiom"] for p in pairs}),
              "n_en_idioms": len({p["en_idiom"] for p in pairs}),
              "frac_sim_lt_0.75": round(float((sims < 0.75).mean()), 4),
              "bin_counts": {name: int(sum(1 for s in sims if bin_of(s) == i)) for i, name in enumerate(BIN_NAMES)}}
    pop_share = {k: v / len(pairs) for k, v in pstats["bin_counts"].items()}

    z2e = json.load(open(os.path.join(DATA, "cross_lingual_analysis/translations_zh_to_en.json")))
    e2z = json.load(open(os.path.join(DATA, "cross_lingual_analysis/translations_en_to_zh.json")))
    low = lambda xs: {x.lower() for x in xs}
    agg = {"all": [0, 0, 0], ">=0.75": [0, 0, 0]}
    per_bin = {b: [0, 0, 0] for b in BIN_NAMES}
    for p in pairs:
        ze = set(p.get("zh_entities") or [])
        ee = low(p.get("en_entities") or []) - SLOTS
        if not ze or not ee:
            continue
        cov = any(z in z2e for z in ze) or any(e in e2z for e in ee)
        share = any(low(z2e.get(z, [])) & ee for z in ze) or any(set(e2z.get(e, [])) & ze for e in ee)
        keys = ["all"] + ([">=0.75"] if p["similarity"] >= 0.75 else [])
        for k in keys:
            agg[k][0] += 1; agg[k][1] += cov; agg[k][2] += share
        b = BIN_NAMES[bin_of(p["similarity"])]
        per_bin[b][0] += 1; per_bin[b][1] += cov; per_bin[b][2] += share
    fmt = lambda v: {"both_have_entities": v[0], "translatable": v[1], "share_entity": v[2],
                     "share_rate_of_translatable": round(v[2] / max(v[1], 1), 4)}
    shared = {k: fmt(v) for k, v in agg.items()}
    shared["per_bin"] = {k: fmt(v) for k, v in per_bin.items()}

    # ---------------- (d) LLM-judged precision
    rng = random.Random(0)
    by_bin = {i: [p for p in pairs if bin_of(p["similarity"]) == i] for i in range(len(BINS))}
    sample = []
    for i in range(len(BINS)):
        for p in rng.sample(by_bin[i], 50):
            q = {k: p[k] for k in ("en_idiom", "en_matched_meaning", "zh_idiom", "zh_matched_meaning", "similarity")}
            q["bin"] = i
            sample.append(q)
    rng.shuffle(sample)  # mix bins inside each judge call so the judge cannot infer the bin
    judges = {} if args.judges == "second" else {NAMES[PRIMARY]: (PRIMARY, "v2_pair_judge_primary")}
    if args.judges in ("both", "second"):
        judges[NAMES[SECOND]] = (SECOND, "v2_pair_judge_qwen")
        judges[NAMES[THIRD]] = (THIRD, "v2_pair_judge_aya")
    if args.or_judge:
        judges["nemotron-3-ultra-550b"] = ("openrouter:nvidia/nemotron-3-ultra-550b-a55b:free", "v2_pair_judge_or")
    labels_by = {}
    prec = {}
    for name, (model, tag) in judges.items():
        labels_by[name] = judge(sample, model, tag)
        prec[name] = precision_table(sample, labels_by[name], pop_share)
        prec[name]["n_valid"] = sum(l is not None for l in labels_by[name])
    ms = list(judges)
    agree = {}
    bin2 = lambda x: "yes" if x == "yes" else "not"
    for x in range(len(ms)):
        for y in range(x + 1, len(ms)):
            both = [(a, b) for a, b in zip(labels_by[ms[x]], labels_by[ms[y]]) if a and b]
            if not both:
                continue
            agree[f"{ms[x]} vs {ms[y]}"] = {
                "n": len(both),
                "raw_agreement_3way": round(float(np.mean([a == b for a, b in both])), 3),
                "cohen_kappa_3way": round(float(cohen_kappa_score([a for a, _ in both], [b for _, b in both])), 3),
                "cohen_kappa_yes_vs_not": round(float(cohen_kappa_score([bin2(a) for a, _ in both], [bin2(b) for _, b in both])), 3)}
    if len(ms) >= 2:  # majority vote over all judges (ties -> the more conservative label)
        order = {"no": 0, "partial": 1, "yes": 2}
        maj = []
        for ls in zip(*[labels_by[m] for m in ms]):
            ls = [l for l in ls if l]
            if not ls:
                maj.append(None)
                continue
            c = Counter(ls)
            top = max(c.values())
            maj.append(min([l for l in c if c[l] == top], key=lambda l: order[l]))
        prec["majority_vote"] = precision_table(sample, maj, pop_share)
    for q, *ls in zip(sample, *[labels_by[m] for m in judges]):
        q["labels"] = dict(zip(judges, ls))
    dump(sample, "pair_judge_sample.json" if args.judges != "second" else "pair_judge_sample_second_only.json")

    res = {"cluster_sizes": sizes, "pairs": pstats, "shared_entity_rate": shared,
           "llm_judged_precision": {"note": "LLM-judged proxy for human annotation (not human labels). 50 pairs per bin, uniform random within bin, seed 0; judge sees idioms and the matched figurative meanings; 10 pairs per call, bins shuffled.",
                                    "judges": prec, "inter_judge_agreement": agree}}
    dump(res, "meaning_clusters.json" if args.judges != "second" else "meaning_clusters_second_only.json")
    print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
