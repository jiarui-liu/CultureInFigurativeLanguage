#!/usr/bin/env python3
"""Task 1 stability check: re-annotate a stratified random sample of 300 entities (75 per
language, uniform from each language's top-2000 entities, seed 0) with independent LLMs and
compare with the primary labels (Cohen's kappa, raw agreement, confusion matrix).

Second annotators: Qwen3.5-9B and aya-expanse-8b (local vLLM; one model per process, so run
this script once per annotator with --only); optionally nvidia/nemotron-3-ultra-550b-a55b:free
via OpenRouter (free tier), if reachable.

    PYTHONPATH=src python src/culture/analysis/v2/entity_typology_validate.py [--skip_openrouter]
"""
import argparse
import json
import os
import random

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common import OUT, dump, entity_counter, load_kb, surface_forms
from entity_typology import LANGS, LANG_NAME, PROMPT, TYPE_DEFS, TYPES, build_items, classify
from culture.bidirectional.llm_api import complete_many, parse_json
from local_llm import NAMES, PRIMARY, SECOND, THIRD


def classify_openrouter(lang, ents, sf, examples, model):
    batches = [ents[i:i + 50] for i in range(0, len(ents), 50)]
    prompts = [PROMPT.format(lang=LANG_NAME[lang], defs=TYPE_DEFS, labels=", ".join(TYPES),
                             items=build_items(lang, b, sf, examples)) for b in batches]
    outs = complete_many(prompts, workers=1, model=model, provider="openrouter", json_mode=True,
                         tag="v2_typology_val_or", max_tokens=12000)
    lab = {}
    for b, o in zip(batches, outs):
        d = parse_json(o) or {}
        for i, e in enumerate(b, 1):
            t = d.get(str(i)) if isinstance(d, dict) else None
            if isinstance(t, str) and t.strip().lower() in TYPES:
                lab[e] = t.strip().lower()
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_openrouter", action="store_true")
    ap.add_argument("--only", default=None, help="run only annotators whose name contains this string")
    args = ap.parse_args()
    random.seed(0)
    prim, sample, ctx = {}, {}, {}
    for lang in LANGS:
        labs = json.load(open(os.path.join(OUT, f"entity_typology_labels_{lang}.json")))
        prim[lang] = {e: v["type"] for e, v in labs.items() if v["type"]}
        top = list(labs)[:2000]  # file is ordered by mentions
        pool = [e for e in top if e in prim[lang]]
        sample[lang] = random.sample(pool, 75)
        rows = load_kb(lang)
        sf = surface_forms(lang, rows)
        ex = {}
        for r in rows:
            for e in r["entities"]:
                ex.setdefault(e, r["idiom"])
        ctx[lang] = (sf, ex)

    annotators = {NAMES[SECOND]: lambda lang, ents: classify(lang, ents, *ctx[lang], model=SECOND,
                                                             tag="v2_typology_val_qwen9b")[0],
                  NAMES[THIRD]: lambda lang, ents: classify(lang, ents, *ctx[lang], model=THIRD,
                                                            tag="v2_typology_val_aya")[0]}
    if not args.skip_openrouter:
        m = "nvidia/nemotron-3-ultra-550b-a55b:free"
        annotators[m] = lambda lang, ents, m=m: classify_openrouter(lang, ents, *ctx[lang], m)

    if args.only:
        annotators = {k: v for k, v in annotators.items() if args.only in k}
    vp = os.path.join(OUT, "entity_typology_validation.json")
    old = json.load(open(vp)) if os.path.exists(vp) else {}
    val = {**old, "sample": "75 entities per language, uniform random from the top-2000 entities by mentions (seed 0)",
           "primary": NAMES[PRIMARY]}
    for name, fn in annotators.items():
        y1, y2, per = [], [], {}
        try:
            for lang in LANGS:
                lab2 = fn(lang, sample[lang])
                a = [prim[lang][e] for e in sample[lang] if e in lab2]
                b = [lab2[e] for e in sample[lang] if e in lab2]
                per[lang] = {"n": len(a),
                             "agreement": round(float(np.mean([x == y for x, y in zip(a, b)])), 3) if a else None,
                             "kappa": round(float(cohen_kappa_score(a, b)), 3) if len(set(a + b)) > 1 else None}
                y1 += a
                y2 += b
        except Exception as e:
            val[name] = {"error": str(e)[:300]}
            continue
        if not y1:
            val[name] = {"error": "no valid outputs"}
            continue
        val[name] = {"n": len(y1), "raw_agreement": round(float(np.mean([a == b for a, b in zip(y1, y2)])), 3),
                     "cohen_kappa": round(float(cohen_kappa_score(y1, y2)), 3), "per_language": per,
                     "confusion_rows_primary_cols_second": {"labels": TYPES,
                                                            "matrix": confusion_matrix(y1, y2, labels=TYPES).tolist()},
                     "disagreements_examples": [
                         {"lang": l, "entity": e, "primary": prim[l][e]} for l in LANGS for e in sample[l][:0]]}
        print(name, val[name]["cohen_kappa"], val[name]["raw_agreement"])
    dump(val, "entity_typology_validation.json")


if __name__ == "__main__":
    main()
