#!/usr/bin/env python3
"""How the training corpus samples the knowledge base, and whether the benchmarks ever
mention the imagery the idioms are built from.

A4b **Corpus selection bias.**  Only 2.1K of 16.6K Hindi proverbs and 20.4K of 27.3K Chinese
     chengyu occur in any scanned document, so the idiom corpus is a *biased* sample of
     \\dataname{}.  The paper reports the coverage numbers but never asks whether the
     web-attested idioms differ in kind from the rest.  If they do -- if the proverbs that
     survive into the corpus are the less culture-specific ones -- that bounds what
     Idiom-CPT can possibly teach, and it is a property of every pipeline that mines idioms
     from web text, not just ours.  We compare attested and unattested idioms on the entity
     typology of \\S4.1, on how many entities they use, and on length, and we also report
     the *exposure-weighted* type distribution, which is what the model actually reads.

A4  **Entity coverage of the benchmarks.**  \\S5.4 argues from a taxonomy of what items
     *test*; this is the complementary, annotation-free check on what they *mention*.  For
     every benchmark we count the share of items that mention a knowledge-base entity at
     all, and correlate an entity's frequency in idioms with its frequency in benchmark
     items.  Low coverage and a weak correlation say the benchmarks and the idioms talk
     about different things, independently of any LLM judgement.

    PYTHONPATH=src:src/culture/analysis/v2 python kb_corpus_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import common  # noqa: E402

DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")
ITEMS = os.environ.get("CULTURE_ITEMS_DIR", f"{DATA}/bidir/items")
V2 = os.path.join(os.environ.get(
    "CULTURE_REPO",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage"),
    "docs/paper_stats/analysis_v2")
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

COUNT_FILES = {
    "zh": [f"{DATA}/fineweb-edu-zh-chengyu-cpt/stats/kept_idiom_counts_zh.json",
           f"{DATA}/mc4-zh-idiom-cpt/stats/kept_idiom_counts_zh.json"],
    "hi": [f"{DATA}/hi-proverbs-cpt/stats/kept_idiom_counts_hi.json"],
    "ar": [f"{DATA}/ar-amthal-cpt/stats/kept_idiom_counts_ar.json"],
}
LANGS = ("ar", "hi", "zh")
TASKS_BY_LANG = {
    "ar": ["arabculture", "alyah", "dzirieval", "arabic_cultural_qa", "global_piqa_ar",
           "arabicmmlu", "kinayat_meaning", "ar_figurative",
           "idiomatlas_mc_ar_seen", "idiomatlas_mc_ar_unseen"],
    "hi": ["milu", "global_piqa_hi", "global_piqa_hi_cultural", "mabl",
           "parambench_hi_culture", "parambench_hi_other",
           "idiomatlas_mc_hi_seen", "idiomatlas_mc_hi_unseen"],
    "zh": ["ccpm", "cmmlu", "chengyu_bench", "global_piqa_zh", "global_piqa_zh_cultural",
           "idiomatlas_mc_zh_seen", "idiomatlas_mc_zh_unseen"],
}


def load_counts(lang):
    c = Counter()
    for p in COUNT_FILES[lang]:
        if os.path.exists(p):
            for k, v in json.load(open(p, encoding="utf-8")).items():
                c[k] += int(v)
    return c


def load_types(lang):
    p = f"{V2}/entity_typology_labels_{lang}.json"
    if not os.path.exists(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))
    d = d.get("labels", d)
    # values are either the type string or {"type": ..., "mentions": ...}
    return {k: (v["type"] if isinstance(v, dict) else v) for k, v in d.items()}


def chi2_cramers_v(table):
    t = np.asarray(table, float)
    n = t.sum()
    if n == 0:
        return 0.0, 0.0, 0
    exp = np.outer(t.sum(1), t.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((t - exp) ** 2 / exp)
    k = min(t.shape) - 1
    return float(chi2), float(np.sqrt(chi2 / (n * k))) if k else 0.0, int((t.shape[0] - 1) * (t.shape[1] - 1))


def _rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kb_corpus_benchmark.json")
    args = ap.parse_args()
    out = {"selection_bias": {}, "entity_coverage": {}}

    for lang in LANGS:
        kb = common.load_kb(lang)
        counts = load_counts(lang)
        types = load_types(lang)
        att = [r for r in kb if counts.get(r["idiom"], 0) > 0]
        unatt = [r for r in kb if counts.get(r["idiom"], 0) == 0]

        # How concentrated is the corpus over idioms?  A gloss-driven, dose-dependent
        # effect can only reach the head of this distribution.
        v = np.array(sorted((c for c in counts.values() if c > 0), reverse=True), float)
        if v.size:
            tot = v.sum()
            cum = np.cumsum(v[::-1]) / tot
            out.setdefault("occurrence_concentration", {})[lang] = {
                "n_attested": int(v.size), "total_occurrences": int(tot),
                "top1pct_share": float(v[: max(1, v.size // 100)].sum() / tot),
                "top10pct_share": float(v[: max(1, v.size // 10)].sum() / tot),
                "median": float(np.median(v)),
                "gini": float(1 - 2 * cum.sum() / v.size + 1 / v.size),
                "share_under_10_docs": float((v < 10).mean()),
            }

        def typedist(rows, weight=None):
            c = Counter()
            for r in rows:
                w = weight(r) if weight else 1
                for e in r["entities"]:
                    t = types.get(e)
                    if t:
                        c[t] += w
            return c

        ta, tu = typedist(att), typedist(unatt)
        tw = typedist(att, weight=lambda r: counts.get(r["idiom"], 0))
        keys = sorted(set(ta) | set(tu))
        chi2, v, dof = chi2_cramers_v([[ta[k] for k in keys], [tu[k] for k in keys]])

        def share(c):
            tot = sum(c.values()) or 1
            return {k: c[k] / tot for k in keys}

        out["selection_bias"][lang] = {
            "n_kb": len(kb), "n_attested": len(att), "n_unattested": len(unatt),
            "coverage": len(att) / max(1, len(kb)),
            "type_share_attested": share(ta),
            "type_share_unattested": share(tu),
            "type_share_exposure_weighted": share(tw),
            "chi2": chi2, "dof": dof, "cramers_v": v,
            "mean_entities_attested": float(np.mean([len(r["entities"]) for r in att] or [0])),
            "mean_entities_unattested": float(np.mean([len(r["entities"]) for r in unatt] or [0])),
            "mean_len_attested": float(np.mean([len(r["idiom"] or "") for r in att] or [0])),
            "mean_len_unattested": float(np.mean([len(r["idiom"] or "") for r in unatt] or [0])),
            "mean_fig_attested": float(np.mean([len(r["fig"]) for r in att] or [0])),
            "mean_fig_unattested": float(np.mean([len(r["fig"]) for r in unatt] or [0])),
        }
        top_shift = sorted(
            ((k, share(ta)[k] - share(tu)[k]) for k in keys), key=lambda x: -abs(x[1]))[:4]
        print(f"[{lang}] coverage {len(att)}/{len(kb)} = {len(att)/max(1,len(kb)):.1%}, "
              f"V={v:.3f}, biggest type shifts " +
              ", ".join(f"{k} {d:+.1%}" for k, d in top_shift))

        # ---------------- entity coverage of benchmarks
        ent_freq = common.entity_counter(kb)          # entity -> # idioms
        # Only entities long enough for substring matching to be meaningful.
        minlen = 1 if lang == "zh" else 3
        ents = [e for e, c in ent_freq.items() if e and len(e) >= minlen and c >= 3]
        sf = common.surface_forms(lang, kb)
        forms = {e: {f for f, _ in sf[e].most_common(4)} | {e} for e in ents}

        for task in TASKS_BY_LANG[lang]:
            p = f"{ITEMS}/{task}.jsonl"
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
            hits = Counter()
            n_with = 0
            for o in rows:
                txt = (o["stem"] + " " + " ".join(o.get("options") or []))
                if lang != "zh":
                    txt = " " + re.sub(r"\s+", " ", txt) + " "
                found = set()
                for e in ents:
                    for f in forms[e]:
                        if not f:
                            continue
                        if (f in txt) if lang == "zh" else (f" {f} " in txt or f" {f}" in txt[-len(f) - 1:]):
                            found.add(e)
                            break
                if found:
                    n_with += 1
                hits.update(found)
            shared = [e for e in ents if hits[e] > 0]
            rho = spearman([ent_freq[e] for e in ents], [hits[e] for e in ents])
            out["entity_coverage"][f"{lang}/{task}"] = {
                "n_items": len(rows),
                "share_items_with_kb_entity": n_with / max(1, len(rows)),
                "n_distinct_entities": len(shared),
                "n_candidate_entities": len(ents),
                "spearman_idiom_freq_vs_benchmark_freq": rho,
                "top_entities": hits.most_common(12),
            }
            print(f"  {task:28s} items with a KB entity {n_with/max(1,len(rows)):5.1%}  "
                  f"distinct {len(shared):5d}  rho={rho}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
