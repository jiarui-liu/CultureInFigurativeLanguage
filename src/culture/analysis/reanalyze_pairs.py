#!/usr/bin/env python3
"""
Re-analyze existing cross-lingual pairs without recomputing embeddings.
Useful for updating analysis after code changes.
"""

import argparse
import json
from collections import defaultdict


def load_pairs(pairs_path: str):
    """Load pairs from JSONL file."""
    pairs = []
    with open(pairs_path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def analyze_entity_differences(pairs):
    """
    Analyze how entities differ between Chinese and English idioms
    that share similar figurative meanings.
    """
    analysis = {
        "total_pairs": len(pairs),
        "pairs_with_different_entities": 0,
        "pairs_with_overlapping_entities": 0,
        "pairs_both_have_entities": 0,
        "pairs_only_zh_has_entities": 0,
        "pairs_only_en_has_entities": 0,
        "pairs_neither_has_entities": 0,
        "zh_entity_categories": defaultdict(int),
        "en_entity_categories": defaultdict(int),
    }

    for pair in pairs:
        zh_entities = set(pair["zh_entities"])
        en_entities = set(pair["en_entities"])

        has_zh = len(zh_entities) > 0
        has_en = len(en_entities) > 0

        if has_zh and has_en:
            analysis["pairs_both_have_entities"] += 1
            if zh_entities & en_entities:
                analysis["pairs_with_overlapping_entities"] += 1
            else:
                analysis["pairs_with_different_entities"] += 1
        elif has_zh:
            analysis["pairs_only_zh_has_entities"] += 1
        elif has_en:
            analysis["pairs_only_en_has_entities"] += 1
        else:
            analysis["pairs_neither_has_entities"] += 1

        for e in zh_entities:
            analysis["zh_entity_categories"][e] += 1
        for e in en_entities:
            analysis["en_entity_categories"][e] += 1

    # Get ALL examples where both languages have entities
    examples_with_entities = [
        p for p in pairs
        if p["zh_entities"] and p["en_entities"]
    ]

    analysis["examples"] = examples_with_entities
    analysis["num_examples_with_both_entities"] = len(examples_with_entities)
    analysis["zh_entity_categories"] = dict(analysis["zh_entity_categories"])
    analysis["en_entity_categories"] = dict(analysis["en_entity_categories"])

    return analysis


def save_analysis(analysis, output_path: str):
    """Save analysis to JSON file."""
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"Saved analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-analyze existing cross-lingual pairs"
    )
    parser.add_argument(
        "--pairs_input",
        type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/cross_lingual_pairs.jsonl",
        help="Input JSONL file with pairs"
    )
    parser.add_argument(
        "--analysis_output",
        type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/entity_analysis.json",
        help="Output JSON file for analysis"
    )

    args = parser.parse_args()

    print(f"Loading pairs from {args.pairs_input}...")
    pairs = load_pairs(args.pairs_input)
    print(f"Loaded {len(pairs)} pairs")

    print("Analyzing entity differences...")
    analysis = analyze_entity_differences(pairs)

    print(f"\nResults:")
    print(f"  Total pairs: {analysis['total_pairs']}")
    print(f"  Pairs with both entities: {analysis['pairs_both_have_entities']}")
    print(f"  Examples saved: {analysis['num_examples_with_both_entities']}")

    save_analysis(analysis, args.analysis_output)
    print("Done!")


if __name__ == "__main__":
    main()
