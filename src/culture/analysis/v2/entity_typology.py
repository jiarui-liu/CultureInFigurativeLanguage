#!/usr/bin/env python3
"""Task 1: LLM-assigned semantic typology of idiom entities in en / zh / hi / ar.

Every distinct (normalized) entity of each language is assigned by an LLM to one type of a
fixed typology, given one example idiom that contains it (for disambiguation). No keyword
lists are used for the assignment. Stability is checked by re-annotating a stratified random
sample of 300 entities with other LLMs and computing Cohen's kappa.

Statistics are computed over entity MENTIONS (entity occurrences, de-duplicated within an
idiom): per-language shares, chi-square test of independence (language x type), Cramer's V,
and adjusted standardized residuals (Agresti 2007).

    PYTHONPATH=src python src/culture/analysis/v2/entity_typology.py [--top N]
Outputs: docs/paper_stats/analysis_v2/entity_typology.json, entity_typology_shares.csv,
         entity_typology_labels_{lang}.json, entity_typology_validation.json
"""
import argparse
import csv
import json
import os
import random

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common import OUT, dump, entity_counter, load_kb, surface_forms, norm_entity
from culture.bidirectional.llm_api import parse_json
from local_llm import NAMES, PRIMARY, generate

LANGS = ["en", "zh", "hi", "ar"]
LANG_NAME = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
TYPES = ["animal", "body_mind", "nature_cosmos", "food_drink", "kinship_social",
         "religion_supernatural", "occupation_economy", "artefact_household", "abstract_other"]
TYPE_DEFS = """- animal: real or mythical animals (including birds, fish, insects) and their parts (horn, tail, feather).
- body_mind: parts of the human body, bodily fluids and functions, the senses, and mental or emotional faculties (mind, soul as a faculty, memory).
- nature_cosmos: the natural world and the cosmos: sky, heavenly bodies, weather, seasons and the day-night cycle, land and landforms, water bodies, fire, stone and minerals, plants, trees and flowers.
- food_drink: foods, dishes, ingredients, drinks, and grains or produce considered as food.
- kinship_social: people and social relations: kin terms, generic people (man, woman, child, people), social roles and ranks (king, master, servant, guest, neighbour, enemy, thief), social or ethnic groups, and named persons or characters.
- religion_supernatural: gods, spirits, demons, ghosts, religious figures and functionaries, religious places, rituals, and sacred texts or concepts.
- occupation_economy: money, wealth, prices, debt, trade, markets, work and labour, and people or activities defined by an occupation or trade.
- artefact_household: man-made objects: tools, vessels, furniture, clothing and ornaments, buildings and their parts, roads, vehicles, weapons, and household items.
- abstract_other: abstract nouns, actions, qualities, colours, numbers and measures, grammatical placeholders, and anything that fits none of the types above."""

PROMPT = """You are annotating the concrete nouns ("entities") that occur in {lang} idioms and proverbs.
Assign each entity to exactly ONE semantic type from this fixed typology. Judge the sense the entity has in the example idiom; if it is ambiguous, pick the type of its most basic, literal sense.

{defs}

Entities (id. entity  |  example idiom containing it):
{items}

Return a JSON object that maps every id (as a string) to one type label from: {labels}.
Return only the JSON object."""


def build_items(lang, ents, sf, examples):
    lines = []
    for i, e in enumerate(ents, 1):
        shown = sf[e].most_common(1)[0][0] if lang in ("ar", "hi") and sf.get(e) else e
        if lang == "ar" and shown != e:
            shown = f"{shown} (normalized: {e})"
        lines.append(f"{i}. {shown}  |  {examples.get(e, '')}")
    return "\n".join(lines)


def classify(lang, ents, sf, examples, model=PRIMARY, tag="v2_typology_local", batch=50):
    batches = [ents[i:i + batch] for i in range(0, len(ents), batch)]
    prompts = [PROMPT.format(lang=LANG_NAME[lang], defs=TYPE_DEFS, labels=", ".join(TYPES),
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
            # ids are requested; weaker models sometimes key by the entity string instead,
            # or by some other string in the original order (then map by position)
            t = d.get(str(i), d.get(e))
            if t is None and by_pos:
                t = vals[i - 1]
            if isinstance(t, str):
                t = t.strip().lower().replace(" ", "_").replace("&", "").replace("/", "_")
            if t in TYPES:
                labels[e] = t
            else:
                missing.append(e)
    return labels, missing


def cramers_v(tab):
    chi2, p, dof, exp = chi2_contingency(tab)
    n = tab.sum()
    r, c = tab.shape
    return chi2, p, dof, exp, float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def adjusted_residuals(tab, exp):
    n = tab.sum()
    rp = tab.sum(1, keepdims=True) / n
    cp = tab.sum(0, keepdims=True) / n
    return (tab - exp) / np.sqrt(exp * (1 - rp) * (1 - cp))


def contingency(counts, labels, restrict=None, types=TYPES):
    tab = np.zeros((len(LANGS), len(types)), dtype=float)
    for i, l in enumerate(LANGS):
        for e, c in counts[l].items():
            if restrict is not None and e not in restrict[l]:
                continue
            t = labels[l].get(e)
            if t in types:
                tab[i, types.index(t)] += c
    return tab


def stats_block(tab, types=TYPES):
    chi2, p, dof, exp, v = cramers_v(tab)
    res = adjusted_residuals(tab, exp)
    shares = tab / tab.sum(1, keepdims=True)
    pair_v = {}
    for a in range(len(LANGS)):
        for b in range(a + 1, len(LANGS)):
            sub = tab[[a, b]]
            sub = sub[:, sub.sum(0) > 0]
            pair_v[f"{LANGS[a]}-{LANGS[b]}"] = round(cramers_v(sub)[4], 4)
    return {
        "n_mentions": {l: int(tab[i].sum()) for i, l in enumerate(LANGS)},
        "counts": {l: {t: int(tab[i, j]) for j, t in enumerate(types)} for i, l in enumerate(LANGS)},
        "shares": {l: {t: round(float(shares[i, j]), 4) for j, t in enumerate(types)} for i, l in enumerate(LANGS)},
        "chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": round(v, 4),
        "pairwise_cramers_v": pair_v,
        "adjusted_residuals": {l: {t: round(float(res[i, j]), 1) for j, t in enumerate(types)} for i, l in enumerate(LANGS)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="classify only the top-N entities per language (0 = all)")
    ap.add_argument("--val_n", type=int, default=300)
    args = ap.parse_args()
    random.seed(0)

    counts, labels, coverage, idiom_rows = {}, {}, {}, {}
    for lang in LANGS:
        rows = load_kb(lang)
        idiom_rows[lang] = rows
        c = entity_counter(rows)
        counts[lang] = c
        ents = [e for e, _ in c.most_common(args.top or None)]
        sf = surface_forms(lang, rows)
        examples = {}
        for r in rows:
            for e in r["entities"]:
                if e not in examples:
                    examples[e] = r["idiom"]
        lab, missing = classify(lang, ents, sf, examples)
        if missing:  # one retry in smaller batches
            lab2, missing = classify(lang, missing, sf, examples, batch=10, tag="v2_typology_local_retry")
            lab.update(lab2)
        labels[lang] = lab
        tot = sum(c.values())
        coverage[lang] = {
            "n_idioms": len(rows), "n_idioms_with_entity": sum(1 for r in rows if r["entities"]),
            "distinct_entities": len(c), "mentions": tot, "classified_entities": len(lab),
            "unclassified_entities": len(missing),
            "mention_coverage_classified": round(sum(c[e] for e in lab) / tot, 4),
            "mention_coverage_top2000": round(sum(v for _, v in c.most_common(2000)) / tot, 4),
        }
        dump({e: {"type": lab.get(e), "mentions": c[e]} for e, _ in c.most_common()},
             f"entity_typology_labels_{lang}.json")
        print(lang, coverage[lang])

    result = {"method": {
        "entities": "all distinct normalized entities per language (mention = entity listed for an idiom, de-duplicated within idiom; English lower-cased, 6 slot fillers dropped; Arabic normalize_ar + article strip; Arabic entities from the GPT-5.4-enriched HF release)",
        "annotator": f"{NAMES[PRIMARY]} (local vLLM, greedy, thinking off), 50 entities per call, one example idiom per entity",
        "typology": TYPE_DEFS, "test": "Pearson chi-square test of independence, language x type, on mention counts; Cramer's V; adjusted standardized residuals"},
        "coverage": coverage}

    result["all_entities"] = stats_block(contingency(counts, labels))
    top2000 = {l: {e for e, _ in counts[l].most_common(2000)} for l in LANGS}
    result["top2000_entities"] = stats_block(contingency(counts, labels, restrict=top2000))
    concrete = [t for t in TYPES if t != "abstract_other"]
    result["all_entities_excluding_abstract"] = stats_block(contingency(counts, labels, types=concrete), types=concrete)

    # idiom-level view: share of idioms (with >=1 classified entity) that contain >=1 entity of each type
    idiom_level = {}
    for lang in LANGS:
        n, hits = 0, {t: 0 for t in TYPES}
        for r in idiom_rows[lang]:
            ts = {labels[lang].get(e) for e in r["entities"]} - {None}
            if not ts:
                continue
            n += 1
            for t in ts:
                hits[t] += 1
        idiom_level[lang] = {"n_idioms": n, **{t: round(hits[t] / n, 4) for t in TYPES}}
    result["idiom_level_share_with_type"] = idiom_level

    # top entities per type per language (for the paper's examples)
    result["top_entities_per_type"] = {
        l: {t: [e for e, _ in counts[l].most_common() if labels[l].get(e) == t][:8] for t in TYPES}
        for l in LANGS}

    # stability check (second annotators) lives in entity_typology_validate.py
    dump(result, "entity_typology.json")

    with open(os.path.join(OUT, "entity_typology_shares.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "type", "mentions", "share", "adjusted_residual", "share_top2000"])
        A, T = result["all_entities"], result["top2000_entities"]
        for l in LANGS:
            for t in TYPES:
                w.writerow([l, t, A["counts"][l][t], A["shares"][l][t], A["adjusted_residuals"][l][t], T["shares"][l][t]])
    print(json.dumps({k: result["all_entities"][k] for k in ("chi2", "p", "cramers_v", "shares")}, indent=1))


if __name__ == "__main__":
    main()
