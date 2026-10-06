#!/usr/bin/env python3
"""Task 1b stability check: re-annotate a random sample of the abstract entities with
independent LLMs and compare with the primary subtype labels (Cohen's kappa, raw agreement,
confusion matrix). Mirrors entity_typology_validate.py.

This matters more here than at the first level: the subtypology cuts a region the first-level
annotator already found hard, so kappa is the evidence that the seven subtypes are separable
at all rather than a distinction only one model can see.

Second annotators: Qwen3.5-9B and aya-expanse-8b (local vLLM; one model per process, so run
this script once per annotator with --only).

    PYTHONPATH=src python src/culture/analysis/v2/entity_subtypology_validate.py [--only qwen]
"""
import argparse
import json
import os
import random

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common import OUT, dump, load_kb, surface_forms
from entity_typology import LANGS
from entity_subtypology import SUBTYPES, classify_sub
from local_llm import FOURTH, NAMES, PRIMARY, SECOND, THIRD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only annotators whose name contains this")
    ap.add_argument("--per_lang", type=int, default=75)
    args = ap.parse_args()
    random.seed(0)

    prim, sample, ctx = {}, {}, {}
    for lang in LANGS:
        labs = json.load(open(os.path.join(OUT, f"entity_subtypology_labels_{lang}.json")))
        prim[lang] = {e: v["subtype"] for e, v in labs.items() if v.get("subtype")}
        # file is written in mentions order; sample from the head, as the first level does
        pool = [e for e in list(labs)[:2000] if e in prim[lang]]
        sample[lang] = random.sample(pool, min(args.per_lang, len(pool)))
        rows = load_kb(lang)
        ex = {}
        for r in rows:
            for e in r["entities"]:
                ex.setdefault(e, r["idiom"])
        ctx[lang] = (surface_forms(lang, rows), ex)

    annotators = {
        NAMES[SECOND]: lambda lang, ents: classify_sub(lang, ents, *ctx[lang], model=SECOND,
                                                       tag="v2_subtypology_val_qwen9b")[0],
        NAMES[THIRD]: lambda lang, ents: classify_sub(lang, ents, *ctx[lang], model=THIRD,
                                                      tag="v2_subtypology_val_aya")[0],
        NAMES[FOURTH]: lambda lang, ents: classify_sub(lang, ents, *ctx[lang], model=FOURTH,
                                                       tag="v2_subtypology_val_gemma")[0],
    }
    if args.only:
        annotators = {k: v for k, v in annotators.items() if args.only in k}

    vp = os.path.join(OUT, "entity_subtypology_validation.json")
    old = json.load(open(vp)) if os.path.exists(vp) else {}
    val = {**old,
           "sample": f"{args.per_lang} abstract entities per language, uniform random from the "
                     "top-2000 abstract entities by mentions (seed 0)",
           "primary": NAMES[PRIMARY], "labels": SUBTYPES}
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
        val[name] = {"n": len(y1),
                     "raw_agreement": round(float(np.mean([a == b for a, b in zip(y1, y2)])), 3),
                     "cohen_kappa": round(float(cohen_kappa_score(y1, y2)), 3),
                     "per_language": per,
                     "confusion_rows_primary_cols_second": {
                         "labels": SUBTYPES,
                         "matrix": confusion_matrix(y1, y2, labels=SUBTYPES).tolist()}}
        print(name, val[name]["cohen_kappa"], val[name]["raw_agreement"])
    dump(val, "entity_subtypology_validation.json")


if __name__ == "__main__":
    main()
