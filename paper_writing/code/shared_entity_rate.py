#!/usr/bin/env python3
"""How often do English-Chinese idiom pairs with matching figurative meanings
share an entity?  String comparison across scripts is vacuous (always 0), so we
go through the high-recall GPT entity translations produced by
cross_lingual_same_entity_diff_meaning.py (top-500 entities per language).
High recall makes the reported sharing rate an upper bound."""
import json, os
D = "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms"
SLOTS = {"something", "someone", "thing", "person", "place", "way"}
z2e = json.load(open(os.path.join(D, "cross_lingual_analysis/translations_zh_to_en.json")))
e2z = json.load(open(os.path.join(D, "cross_lingual_analysis/translations_en_to_zh.json")))
low = lambda xs: {x.lower() for x in xs}
n_all = n_both = n_cov = n_share = 0
hi = {"n": 0, "cov": 0, "share": 0}
for line in open(os.path.join(D, "cross_lingual_pairs.jsonl")):
    p = json.loads(line); n_all += 1
    ze = set(p.get("zh_entities") or [])
    ee = low(p.get("en_entities") or []) - SLOTS
    if not ze or not ee:
        continue
    n_both += 1
    covered = any(z in z2e for z in ze) or any(e in e2z for e in ee)
    share = any(low(z2e.get(z, [])) & ee for z in ze) or any(set(e2z.get(e, [])) & ze for e in ee)
    n_cov += covered; n_share += share
    sim = p.get("similarity") or p.get("cosine_similarity") or 0
    if sim >= 0.75:
        hi["n"] += 1; hi["cov"] += covered; hi["share"] += share
print(f"pairs={n_all} both_sides_have_entities(no slots)={n_both} translatable={n_cov} "
      f"share_entity={n_share} ({100*n_share/n_cov:.1f}% of translatable)")
print("sim>=0.75:", hi, f"{100*hi['share']/max(hi['cov'],1):.1f}%")
