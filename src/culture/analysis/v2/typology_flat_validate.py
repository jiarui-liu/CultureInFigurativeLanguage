#!/usr/bin/env python3
"""Is the 15-way scheme separable, or only each level of it?

entity_typology_validate.py measures the 9-way choice and entity_subtypology_validate.py
measures the 7-way choice *given that the entity is already known to be abstract* -- its
prompt says so outright. Neither measures the flat 15-way scheme the figures actually show,
and by construction neither can see a cross-level confusion: an entity the first pass wrongly
routed to abstract_other is told, in the second pass, that it is "not an animal, body part,
natural feature, food, person, deity, artefact or sum of money", so it can never be sent back.

That matters because the two codebooks overlap in writing. occupation_economy claims "work and
labour" while action_conflict claims "work and effort as abstractions"; nature_cosmos claims
"seasons and the day-night cycle" while time_change claims "day as a unit, year"; body_mind
claims mental faculties while morality_truth claims "wisdom and folly, knowledge".

Here we put all fifteen labels in one flat codebook, with no hint of the two-level structure,
re-annotate a stratified sample, and report kappa plus the full confusion -- separating
within-level disagreement from cross-level leakage.

    PYTHONPATH=src:src/culture/analysis/v2 python typology_flat_validate.py [--per_lang 90]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common import OUT, dump, load_kb, surface_forms
from entity_typology import LANG_NAME, LANGS, TYPE_DEFS, TYPES, build_items, parse_json
from entity_subtypology import SUBTYPE_DEFS, SUBTYPES
from local_llm import FOURTH, NAMES, PRIMARY, SECOND, THIRD, generate

CONCRETE = [t for t in TYPES if t != "abstract_other"]
FLAT = CONCRETE + SUBTYPES  # 8 + 7
LEVEL = {t: "concrete" for t in CONCRETE}
LEVEL.update({t: "subtype" for t in SUBTYPES})

# The two codebooks, concatenated with the abstract_other line dropped: the subtypes now
# stand on their own rather than as children of a bin the annotator is told about.
FLAT_DEFS = "\n".join(
    [l for l in TYPE_DEFS.strip().split("\n") if not l.startswith("- abstract_other")]
    + SUBTYPE_DEFS.strip().split("\n")
)

FLAT_PROMPT = """You are annotating the nouns ("entities") that occur in {lang} idioms and proverbs.
Assign each entity to exactly ONE semantic type from this fixed typology. Judge the sense the entity has in the example idiom; if it is ambiguous, pick the type of its most basic, literal sense. Use residual_other only when none of the other types applies.

{defs}

Entities (id. entity  |  example idiom containing it):
{items}

Return a JSON object that maps every id (as a string) to one type label from: {labels}.
Return only the JSON object."""


def classify_flat(lang, ents, sf, examples, model, tag, batch=50):
    batches = [ents[i:i + batch] for i in range(0, len(ents), batch)]
    prompts = [FLAT_PROMPT.format(lang=LANG_NAME[lang], defs=FLAT_DEFS,
                                  labels=", ".join(FLAT),
                                  items=build_items(lang, b, sf, examples))
               for b in batches]
    outs = generate(prompts, tag=tag, model=model, max_tokens=3000)
    labels = {}
    for b, o in zip(batches, outs):
        d = parse_json(o) or {}
        if not isinstance(d, dict):
            continue
        vals = list(d.values())
        by_pos = len(vals) == len(b) and not any(str(i) in d for i in range(1, len(b) + 1))
        for i, e in enumerate(b, 1):
            t = d.get(str(i), d.get(e))
            if t is None and by_pos:
                t = vals[i - 1]
            if isinstance(t, str):
                t = t.strip().lower().replace(" ", "_").replace("&", "").replace("/", "_")
            if t in FLAT:
                labels[e] = t
    return labels


def composed_label(lang):
    """The label the figures show: level-1 type, or its subtype when it was abstract."""
    t1 = json.load(open(os.path.join(OUT, f"entity_typology_labels_{lang}.json"),
                        encoding="utf-8"))
    p2 = os.path.join(OUT, f"entity_subtypology_labels_{lang}.json")
    t2 = json.load(open(p2, encoding="utf-8")) if os.path.exists(p2) else {}
    out = {}
    for e, v in t1.items():
        ty = v.get("type")
        if ty == "abstract_other":
            sub = (t2.get(e) or {}).get("subtype")
            if sub:
                out[e] = sub
        elif ty in CONCRETE:
            out[e] = ty
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_lang", type=int, default=90)
    ap.add_argument("--model", choices=["primary", "second", "third", "fourth"],
                    default="fourth")
    ap.add_argument("--out", default="typology_flat_validation.json")
    args = ap.parse_args()
    model = {"primary": PRIMARY, "second": SECOND, "third": THIRD,
             "fourth": FOURTH}[args.model]
    rng = random.Random(0)

    y1, y2 = [], []
    per_lang = {}
    for lang in LANGS:
        comp = composed_label(lang)
        if not comp:
            continue
        # Stratify so the rarer labels are actually represented.
        by_lab = defaultdict(list)
        for e, t in comp.items():
            by_lab[t].append(e)
        per = max(1, args.per_lang // len(FLAT))
        sample = []
        for t in FLAT:
            pool = by_lab.get(t, [])
            sample += rng.sample(pool, min(per, len(pool)))
        rows = load_kb(lang)
        ex = {}
        for r in rows:
            for e in r["entities"]:
                ex.setdefault(e, r["idiom"])
        sf = surface_forms(lang, rows)
        got = classify_flat(lang, sample, sf, ex, model,
                            tag=f"v2_flat_val_{args.model}")
        a = [comp[e] for e in sample if e in got]
        b = [got[e] for e in sample if e in got]
        per_lang[lang] = {
            "n": len(a),
            "agreement": round(float(np.mean([x == y for x, y in zip(a, b)])), 3) if a else None,
            "kappa": round(float(cohen_kappa_score(a, b)), 3) if len(set(a + b)) > 1 else None,
        }
        y1 += a
        y2 += b
        print(f"  {lang}: n={len(a)} agreement={per_lang[lang]['agreement']} "
              f"kappa={per_lang[lang]['kappa']}")

    cross = sum(1 for a, b in zip(y1, y2) if a != b and LEVEL[a] != LEVEL[b])
    within = sum(1 for a, b in zip(y1, y2) if a != b and LEVEL[a] == LEVEL[b])
    pairs = Counter((a, b) for a, b in zip(y1, y2) if a != b)

    out = {
        "method": __doc__,
        "annotator": NAMES.get(model, model),
        "labels": FLAT,
        "n": len(y1),
        "raw_agreement": round(float(np.mean([a == b for a, b in zip(y1, y2)])), 3),
        "cohen_kappa": round(float(cohen_kappa_score(y1, y2)), 3),
        "per_language": per_lang,
        "disagreements": {
            "total": int(cross + within),
            "within_level": int(within),
            "cross_level": int(cross),
            "cross_level_share": round(cross / max(1, cross + within), 3),
        },
        "top_confusions": [{"composed": a, "flat": b, "n": n,
                            "cross_level": LEVEL[a] != LEVEL[b]}
                           for (a, b), n in pairs.most_common(20)],
        "confusion_rows_composed_cols_flat": {
            "labels": FLAT,
            "matrix": confusion_matrix(y1, y2, labels=FLAT).tolist(),
        },
    }
    print("wrote", dump(out, args.out))
    d = out["disagreements"]
    print(f"\nkappa={out['cohen_kappa']} raw={out['raw_agreement']} n={out['n']}")
    print(f"disagreements: {d['total']} "
          f"({d['within_level']} within-level, {d['cross_level']} cross-level "
          f"= {d['cross_level_share']:.0%})")
    print("\ntop confusions (composed -> flat):")
    for c in out["top_confusions"][:12]:
        print(f"  {c['composed']:22s} -> {c['flat']:22s} {c['n']:3d}"
              f"{'   [CROSS-LEVEL]' if c['cross_level'] else ''}")


if __name__ == "__main__":
    main()
