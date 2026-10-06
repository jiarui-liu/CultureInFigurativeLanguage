#!/usr/bin/env python3
"""Task 1b: split the abstract_other catch-all into seven semantic subtypes.

abstract_other is 30.1% of English and 32.6% of Chinese entity mentions but only 11.9% of
Hindi, so the largest cell of the typology is also the least informative one. This pass
re-annotates every entity the primary typology called abstract_other, using the same model,
the same batching and the same example-idiom disambiguation, but a subtypology designed from
the contents of that bucket (time, speech and reputation, life/death/fate, morality, action,
quantity and space, plus an explicit residual).

The residual bin is the point of the exercise as much as the others: abstract_other is where
the annotator falls back when nothing fits (in entity_typology_validation.json it absorbs more
cross-annotator disagreement than any other type), so measuring how much of it is genuinely
unclassifiable is part of the result.

    PYTHONPATH=src python src/culture/analysis/v2/entity_subtypology.py [--top N]
Outputs: docs/paper_stats/analysis_v2/entity_subtypology.json,
         entity_subtypology_shares.csv, entity_subtypology_labels_{lang}.json
"""
import argparse
import csv
import json
import os

from common import OUT, dump, entity_counter, load_kb, surface_forms
from culture.bidirectional.llm_api import parse_json
from entity_typology import (LANG_NAME, LANGS, TYPES, build_items, stats_block,
                             contingency)
from local_llm import NAMES, PRIMARY, generate

ABSTRACT = "abstract_other"
CONCRETE = [t for t in TYPES if t != ABSTRACT]

SUBTYPES = ["time_change", "speech_name", "life_death_fate", "morality_truth",
            "action_conflict", "quantity_space", "residual_other"]
SUBTYPE_DEFS = """- time_change: time and its divisions (day as a unit, year, hour, moment, age, era), earliness and lateness, youth and old age as periods, and words for change, beginning and ending.
- speech_name: speech and language as such (word, speech, talk, saying, song, news, question, answer), and name, fame, reputation, honour and disgrace.
- life_death_fate: life, death, birth, health and sickness, luck, chance, fate, destiny, fortune and misfortune, and the world or worldly existence taken as a whole.
- morality_truth: moral and evaluative qualities: good and evil, virtue, righteousness, justice, truth and falsehood, sin, shame, law as a moral order, wisdom and folly, knowledge and learning.
- action_conflict: actions, deeds, work and effort as abstractions, plans and schemes, games and contests, quarrels, fights and war, success and failure.
- quantity_space: numbers, measures and amounts, size, weight, shape and form, and abstract spatial relations (side, edge, top, line, point, distance, direction).
- residual_other: anything that fits none of the above: grammatical placeholders and pronouns, generic "thing"/"matter", colours, interjections, and entities too vague or too garbled to place."""

SUB_PROMPT = """You are annotating abstract nouns that occur in {lang} idioms and proverbs.
Each of these has already been judged to be abstract rather than a concrete thing (not an animal, body part, natural feature, food, person, deity, artefact or sum of money). Assign each one to exactly ONE subtype from this fixed subtypology. Judge the sense the entity has in the example idiom; if it is ambiguous, pick the subtype of its most basic, literal sense. Use residual_other only when none of the six substantive subtypes applies.

{defs}

Entities (id. entity  |  example idiom containing it):
{items}

Return a JSON object that maps every id (as a string) to one subtype label from: {labels}.
Return only the JSON object."""


def classify_sub(lang, ents, sf, examples, model=PRIMARY, tag="v2_subtypology_local", batch=50):
    """Same contract as entity_typology.classify, against SUBTYPES."""
    batches = [ents[i:i + batch] for i in range(0, len(ents), batch)]
    prompts = [SUB_PROMPT.format(lang=LANG_NAME[lang], defs=SUBTYPE_DEFS,
                                 labels=", ".join(SUBTYPES),
                                 items=build_items(lang, b, sf, examples)) for b in batches]
    outs = generate(prompts, tag=tag, model=model, max_tokens=3000)
    labels, missing = {}, []
    for b, o in zip(batches, outs):
        d = parse_json(o) or {}
        if not isinstance(d, dict):
            d = {}
        vals = list(d.values())
        by_pos = len(vals) == len(b) and not any(str(i) in d for i in range(1, len(b) + 1))
        for i, e in enumerate(b, 1):
            t = d.get(str(i), d.get(e))
            if t is None and by_pos:
                t = vals[i - 1]
            if isinstance(t, str):
                t = t.strip().lower().replace(" ", "_").replace("&", "").replace("/", "_")
            if t in SUBTYPES:
                labels[e] = t
            else:
                missing.append(e)
    return labels, missing


def load_primary():
    """{lang: {entity: type}} from the first-level pass."""
    out = {}
    for lang in LANGS:
        p = os.path.join(OUT, f"entity_typology_labels_{lang}.json")
        out[lang] = {e: v["type"] for e, v in json.load(open(p)).items() if v.get("type")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0,
                    help="subclassify only the top-N abstract entities per language (0 = all)")
    args = ap.parse_args()

    primary = load_primary()
    counts, sub, coverage = {}, {}, {}
    for lang in LANGS:
        rows = load_kb(lang)
        c = entity_counter(rows)
        counts[lang] = c
        # most_common() order, so --top takes the head of the abstract bucket by mentions
        ents = [e for e, _ in c.most_common() if primary[lang].get(e) == ABSTRACT]
        if args.top:
            ents = ents[:args.top]
        sf = surface_forms(lang, rows)
        examples = {}
        for r in rows:
            for e in r["entities"]:
                examples.setdefault(e, r["idiom"])

        lab, missing = classify_sub(lang, ents, sf, examples)
        if missing:  # one retry in smaller batches, as in the first-level pass
            lab2, missing = classify_sub(lang, missing, sf, examples, batch=10,
                                         tag="v2_subtypology_local_retry")
            lab.update(lab2)
        sub[lang] = lab

        abstract_mentions = sum(c[e] for e in ents)
        coverage[lang] = {
            "abstract_entities": len(ents),
            "abstract_mentions": abstract_mentions,
            "subclassified_entities": len(lab),
            "unclassified_entities": len(missing),
            "mention_coverage_subclassified":
                round(sum(c[e] for e in lab) / abstract_mentions, 4) if abstract_mentions else 0.0,
            "residual_share_of_abstract":
                round(sum(c[e] for e, t in lab.items() if t == "residual_other")
                      / abstract_mentions, 4) if abstract_mentions else 0.0,
        }
        dump({e: {"subtype": lab.get(e), "mentions": c[e]} for e in ents},
             f"entity_subtypology_labels_{lang}.json")
        print(lang, coverage[lang])

    # (a) composition of the abstract bucket itself, (b) the full 8+7 type table
    expanded_types = CONCRETE + SUBTYPES
    expanded = {l: {**{e: t for e, t in primary[l].items() if t != ABSTRACT}, **sub[l]}
                for l in LANGS}

    result = {
        "method": {
            "input": "every entity labelled abstract_other by entity_typology.py",
            "annotator": f"{NAMES[PRIMARY]} (local vLLM, greedy, thinking off), 50 entities per "
                         "call, one example idiom per entity",
            "subtypology": SUBTYPE_DEFS,
            "expanded_table": "the 8 concrete types of the first-level typology plus the 7 "
                              "subtypes, i.e. abstract_other is replaced by its parts",
            "test": "Pearson chi-square test of independence, language x type, on mention "
                    "counts; Cramer's V; adjusted standardized residuals",
        },
        "coverage": coverage,
        "within_abstract": stats_block(contingency(counts, sub, types=SUBTYPES), types=SUBTYPES),
        "expanded": stats_block(contingency(counts, expanded, types=expanded_types),
                                types=expanded_types),
    }
    dump(result, "entity_subtypology.json")

    with open(os.path.join(OUT, "entity_subtypology_shares.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "type", "level", "mentions", "share_of_all_mentions",
                    "adjusted_residual", "share_of_abstract"])
        E, W = result["expanded"], result["within_abstract"]
        for l in LANGS:
            for t in expanded_types:
                w.writerow([l, t, "subtype" if t in SUBTYPES else "concrete",
                            E["counts"][l][t], E["shares"][l][t], E["adjusted_residuals"][l][t],
                            W["shares"][l][t] if t in SUBTYPES else ""])
    print(json.dumps({k: result["within_abstract"][k]
                      for k in ("chi2", "p", "cramers_v", "shares")}, indent=1))


if __name__ == "__main__":
    main()
