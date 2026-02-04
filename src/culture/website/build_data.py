"""
Pre-process raw analysis JSON files into compact data files for the website.

Usage:
    python build_data.py

Reads:
    - cultural_analysis_clusters.json  (Tab 1: same entity, different meanings)
    - bilingual_clusters.json          (Tab 2: same meaning, different entities)

Writes:
    - data/tab1_entity_meanings.json
    - data/tab2_semantic_clusters.json
"""

import json
import os
import re

DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "culture", "data", "idioms",
)

TAB1_SRC = os.path.join(DATA_ROOT, "cross_lingual_analysis", "cultural_analysis_clusters.json")
TAB2_SRC = os.path.join(DATA_ROOT, "semantic_clusters", "bilingual_clusters.json")

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def parse_cultural_analysis(analysis):
    """Parse cultural_analysis field which may be raw JSON or JSON inside markdown code blocks."""
    if not analysis:
        return {}
    try:
        if isinstance(analysis, dict):
            return analysis
        if isinstance(analysis, str):
            json_match = re.search(
                r'```(?:json)?\s*\n?({[^`]+})\s*\n?```', analysis, re.DOTALL
            )
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(analysis)
    except (json.JSONDecodeError, TypeError):
        return {}
    return {}


def build_tab1():
    """Build compact data for Tab 1: Same Entity Across Different Meanings."""
    with open(TAB1_SRC, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for result in raw:
        analysis = parse_cultural_analysis(result.get("cultural_analysis", ""))

        # Handle both field formats: clusters use en_cluster/zh_cluster,
        # legacy pairs use entity_en/entity_zh
        en_cluster = result.get("en_cluster", {})
        zh_cluster = result.get("zh_cluster", {})
        entity_en = en_cluster.get("primary_entity", "") if en_cluster else result.get("entity_en", "")
        entity_zh = zh_cluster.get("primary_entity", "") if zh_cluster else result.get("entity_zh", "")

        # Derive matched_translations: prefer analysis field, fall back to cluster entities
        direction = result.get("translation_direction", "")
        matched_trans = analysis.get("matched_translations", [])
        if not matched_trans:
            # Use target-language cluster entities as matched translations
            if direction == "en_to_zh" and zh_cluster:
                matched_trans = zh_cluster.get("cluster_entities", [])
            elif direction == "zh_to_en" and en_cluster:
                matched_trans = en_cluster.get("cluster_entities", [])

        # Extract idioms used to generate the analysis
        idioms_en = [
            {
                "idiom": idi.get("idiom", ""),
                "entity": idi.get("entity", ""),
                "figurative_meanings": idi.get("figurative_meanings", []),
                "literal_meanings": idi.get("literal_meanings", []),
            }
            for idi in result.get("idioms_en", [])
        ]
        idioms_zh = [
            {
                "idiom": idi.get("idiom", ""),
                "entity": idi.get("entity", ""),
                "figurative_meanings": idi.get("figurative_meanings", []),
                "literal_meanings": idi.get("literal_meanings", []),
            }
            for idi in result.get("idioms_zh", [])
        ]

        rows.append({
            "entity_en": entity_en,
            "entity_zh": entity_zh,
            "direction": direction,
            "matched_translations": matched_trans,
            "en_primary_meanings": analysis.get("english_primary_meanings", []),
            "zh_primary_meanings": analysis.get("chinese_primary_meanings", []),
            "shared_meanings": analysis.get("shared_meanings", []),
            "en_unique_aspects": analysis.get("english_unique_aspects", []),
            "zh_unique_aspects": analysis.get("chinese_unique_aspects", []),
            "cultural_explanation": analysis.get("cultural_explanation", ""),
            "summary": analysis.get("summary", ""),
            "idioms_en": idioms_en,
            "idioms_zh": idioms_zh,
        })

    # Sort: en_to_zh first, then zh_to_en (same order as summary_df)
    en_to_zh = [r for r in rows if r["direction"] == "en_to_zh"]
    zh_to_en = [r for r in rows if r["direction"] == "zh_to_en"]

    out = {"en_to_zh": en_to_zh, "zh_to_en": zh_to_en}
    out_path = os.path.join(OUT_DIR, "tab1_entity_meanings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Tab 1: wrote {len(en_to_zh)} en_to_zh + {len(zh_to_en)} zh_to_en rows -> {out_path}")


def build_tab2():
    """Build compact data for Tab 2: Same Meaning Across Different Entities."""
    with open(TAB2_SRC, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Keep the structure but only the fields needed for the website
    clusters = []
    for item in raw:
        clusters.append({
            "shared_meaning": item.get("shared_meaning", ""),
            "zh_idiom_count": item.get("zh_idiom_count", 0),
            "en_idiom_count": item.get("en_idiom_count", 0),
            "zh_idioms": [
                {
                    "idiom": idi.get("idiom", ""),
                    "entities": idi.get("entities", []),
                    "figurative_meanings": idi.get("figurative_meanings", []),
                    "literal_meanings": idi.get("literal_meanings", []),
                }
                for idi in item.get("zh_idioms", [])
            ],
            "en_idioms": [
                {
                    "idiom": idi.get("idiom", ""),
                    "entities": idi.get("entities", []),
                    "figurative_meanings": idi.get("figurative_meanings", []),
                    "literal_meanings": idi.get("literal_meanings", []),
                }
                for idi in item.get("en_idioms", [])
            ],
        })

    out_path = os.path.join(OUT_DIR, "tab2_semantic_clusters.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False)
    print(f"Tab 2: wrote {len(clusters)} clusters -> {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build_tab1()
    build_tab2()
    print("Done.")
