#!/usr/bin/env python3
"""
Cross-lingual idiom similarity analysis.

This script:
1. Computes embeddings for figurative meanings of idioms (one embedding per meaning)
2. Finds pairs of idioms (Chinese-English) with similar figurative meanings
3. Analyzes how entities differ between languages for idioms with the same meaning

Storage format:
- Embeddings: compressed .npz file (efficient binary format)
- Metadata: .json file with idiom info and indices into embedding array
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def flatten_meanings(meanings: List) -> List[str]:
    """
    Flatten nested lists and filter to valid non-empty strings.
    Some data has nested lists like [['meaning1', 'meaning2']] instead of ['meaning1'].
    """
    result = []
    for item in meanings:
        if isinstance(item, str):
            if item.strip():
                result.append(item.strip())
        elif isinstance(item, list):
            # Recursively flatten nested lists
            result.extend(flatten_meanings(item))
    return result


def load_idioms_with_figurative_meanings(jsonl_path: str) -> List[Dict]:
    """
    Load idioms that have figurative meanings from a JSONL file.
    Returns list of dicts with idiom, figurative_meanings, entities, and literal_meanings.
    """
    idioms = []
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            output = obj.get("output", {})
            if not output:
                continue
            raw_figurative = output.get("figurative_meanings", [])
            raw_literal = output.get("literal_meanings", [])

            # Flatten and clean the meanings
            figurative_meanings = flatten_meanings(raw_figurative)
            literal_meanings = flatten_meanings(raw_literal)

            # Only include idioms that have at least one figurative meaning
            if figurative_meanings:
                idioms.append({
                    "idiom": obj.get("idiom", ""),
                    "figurative_meanings": figurative_meanings,
                    "entities": output.get("entities", []),
                    "literal_meanings": literal_meanings,
                    "index": obj.get("index", -1)
                })

    return idioms


def compute_figurative_embeddings_separate(
    idioms: List[Dict],
    model: SentenceTransformer
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compute embeddings for each figurative meaning separately.
    One idiom can have multiple embeddings (one per figurative meaning).

    Returns:
        embeddings: numpy array of shape (total_meanings, embedding_dim)
        metadata: list of dicts with idiom info and embedding indices
    """
    all_meanings = []
    metadata = []
    current_idx = 0

    for idiom_data in idioms:
        meanings = idiom_data["figurative_meanings"]
        if not meanings:
            continue

        n_meanings = len(meanings)
        all_meanings.extend(meanings)

        metadata.append({
            "idiom": idiom_data["idiom"],
            "entities": idiom_data["entities"],
            "literal_meanings": idiom_data["literal_meanings"],
            "figurative_meanings": meanings,
            "embedding_start_idx": current_idx,
            "embedding_end_idx": current_idx + n_meanings
        })

        current_idx += n_meanings

    print(f"Total figurative meanings to embed: {len(all_meanings)}")

    embeddings = model.encode(
        all_meanings,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings, metadata


def save_embeddings_efficient(
    embeddings: np.ndarray,
    metadata: List[Dict],
    output_base_path: str
):
    """
    Save embeddings efficiently using compressed numpy format.

    Creates two files:
    - {output_base_path}.npz: compressed embeddings array
    - {output_base_path}_meta.json: metadata with idiom info and indices
    """
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

    # Save embeddings as compressed numpy array
    npz_path = f"{output_base_path}.npz"
    np.savez_compressed(npz_path, embeddings=embeddings.astype(np.float16))

    # Save metadata as compact JSON
    meta_path = f"{output_base_path}_meta.json"
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(metadata, f, ensure_ascii=False, separators=(',', ':'))

    # Report file sizes
    npz_size = os.path.getsize(npz_path) / (1024 * 1024)
    meta_size = os.path.getsize(meta_path) / (1024 * 1024)

    print(f"Saved embeddings to {npz_path} ({npz_size:.2f} MB)")
    print(f"Saved metadata to {meta_path} ({meta_size:.2f} MB)")
    print(f"Total: {npz_size + meta_size:.2f} MB")


def load_embeddings_efficient(input_base_path: str) -> Tuple[List[Dict], np.ndarray]:
    """
    Load embeddings from efficient storage format.

    Args:
        input_base_path: Base path (without .npz or _meta.json extension)

    Returns:
        metadata: List of idiom dicts with embedding indices
        embeddings: numpy array of embeddings
    """
    npz_path = f"{input_base_path}.npz"
    meta_path = f"{input_base_path}_meta.json"

    # Load embeddings
    with np.load(npz_path) as data:
        embeddings = data["embeddings"].astype(np.float32)

    # Load metadata
    with open(meta_path, "r", encoding="utf8") as f:
        metadata = json.load(f)

    return metadata, embeddings


def find_cross_lingual_pairs(
    zh_metadata: List[Dict],
    zh_embeddings: np.ndarray,
    en_metadata: List[Dict],
    en_embeddings: np.ndarray,
    similarity_threshold: float = 0.7,
    top_k: int = None
) -> List[Dict]:
    """
    Find pairs of Chinese and English idioms with similar figurative meanings.
    Compares each figurative meaning embedding separately.

    Args:
        zh_metadata: List of Chinese idiom metadata dicts
        zh_embeddings: Embeddings for all Chinese figurative meanings
        en_metadata: List of English idiom metadata dicts
        en_embeddings: Embeddings for all English figurative meanings
        similarity_threshold: Minimum cosine similarity to consider as a match
        top_k: Optional limit on matches per idiom. If None, keeps all above threshold.

    Returns:
        List of paired idioms with similarity scores
    """
    # Compute full similarity matrix between all meanings
    # Shape: (n_zh_meanings, n_en_meanings)
    similarity_matrix = cosine_similarity(zh_embeddings, en_embeddings)

    # For each Chinese idiom, find best matching English idioms
    # by taking max similarity across all meaning pairs
    pairs = []
    seen_pairs = set()

    for zh_meta in tqdm(zh_metadata):
        zh_start = zh_meta["embedding_start_idx"]
        zh_end = zh_meta["embedding_end_idx"]

        # Get similarities for all meanings of this Chinese idiom
        zh_sims = similarity_matrix[zh_start:zh_end]  # Shape: (n_zh_meanings, n_en_meanings)

        # For each English idiom, find max similarity across all meaning pairs
        en_idiom_max_sims = []
        for en_idx, en_meta in tqdm(enumerate(en_metadata)):
            en_start = en_meta["embedding_start_idx"]
            en_end = en_meta["embedding_end_idx"]

            # Max similarity between any ZH meaning and any EN meaning
            pair_sims = zh_sims[:, en_start:en_end]
            max_sim = pair_sims.max()

            # Also track which specific meanings matched best
            best_zh_meaning_idx, best_en_meaning_idx = np.unravel_index(
                pair_sims.argmax(), pair_sims.shape
            )

            en_idiom_max_sims.append({
                "en_idx": en_idx,
                "max_sim": max_sim,
                "best_zh_meaning_idx": int(best_zh_meaning_idx),
                "best_en_meaning_idx": int(best_en_meaning_idx)
            })

        # Sort by similarity and optionally take top-k
        en_idiom_max_sims.sort(key=lambda x: x["max_sim"], reverse=True)

        # If top_k is None, consider all; otherwise limit to top_k
        candidates = en_idiom_max_sims if top_k is None else en_idiom_max_sims[:top_k]

        for match in candidates:
            if match["max_sim"] < similarity_threshold:
                continue

            en_meta = en_metadata[match["en_idx"]]

            # Create unique pair key to avoid duplicates
            pair_key = (zh_meta["idiom"], en_meta["idiom"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            pairs.append({
                "zh_idiom": zh_meta["idiom"],
                "zh_figurative_meanings": zh_meta["figurative_meanings"],
                "zh_matched_meaning": zh_meta["figurative_meanings"][match["best_zh_meaning_idx"]],
                "zh_entities": zh_meta["entities"],
                "zh_literal_meanings": zh_meta["literal_meanings"],
                "en_idiom": en_meta["idiom"],
                "en_figurative_meanings": en_meta["figurative_meanings"],
                "en_matched_meaning": en_meta["figurative_meanings"][match["best_en_meaning_idx"]],
                "en_entities": en_meta["entities"],
                "en_literal_meanings": en_meta["literal_meanings"],
                "similarity": float(match["max_sim"])
            })

    # Sort by similarity (descending)
    pairs.sort(key=lambda x: x["similarity"], reverse=True)

    return pairs


def analyze_entity_differences(pairs: List[Dict]) -> Dict:
    """
    Analyze how entities differ between Chinese and English idioms
    that share similar figurative meanings.

    Returns analysis statistics and examples.
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
        "examples": []
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


def save_pairs(pairs: List[Dict], output_path: str):
    """Save cross-lingual pairs to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} pairs to {output_path}")


def save_analysis(analysis: Dict, output_path: str):
    """Save entity difference analysis to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"Saved analysis to {output_path}")


def print_analysis_summary(analysis: Dict):
    """Print a summary of the entity difference analysis."""
    print("\n" + "="*60)
    print("CROSS-LINGUAL IDIOM ENTITY ANALYSIS")
    print("="*60)

    print(f"\nTotal pairs found: {analysis['total_pairs']}")
    print(f"  - Both have entities: {analysis['pairs_both_have_entities']}")
    print(f"  - Only Chinese has entities: {analysis['pairs_only_zh_has_entities']}")
    print(f"  - Only English has entities: {analysis['pairs_only_en_has_entities']}")
    print(f"  - Neither has entities: {analysis['pairs_neither_has_entities']}")

    if analysis["pairs_with_different_entities"] > 0:
        print(f"\nPairs with different entities (same meaning, different entities): "
              f"{analysis['pairs_with_different_entities']}")

    print("\n" + "-"*60)
    print("TOP CHINESE ENTITIES (by frequency):")
    zh_entities = sorted(
        analysis["zh_entity_categories"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    for entity, count in zh_entities:
        print(f"  {entity}: {count}")

    print("\n" + "-"*60)
    print("TOP ENGLISH ENTITIES (by frequency):")
    en_entities = sorted(
        analysis["en_entity_categories"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    for entity, count in en_entities:
        print(f"  {entity}: {count}")

    print("\n" + "-"*60)
    print("EXAMPLE PAIRS (same meaning, different entities):")
    for i, example in enumerate(analysis["examples"][:10], 1):
        print(f"\n{i}. Similarity: {example['similarity']:.4f}")
        print(f"   Chinese: {example['zh_idiom']}")
        print(f"     Matched meaning: {example['zh_matched_meaning']}")
        print(f"     Entities: {example['zh_entities']}")
        print(f"   English: {example['en_idiom']}")
        print(f"     Matched meaning: {example['en_matched_meaning']}")
        print(f"     Entities: {example['en_entities']}")


def main_compute_embeddings(args):
    """Main function for computing figurative meaning embeddings."""
    print(f"Loading idioms from {args.input}...")
    idioms = load_idioms_with_figurative_meanings(args.input)
    print(f"Loaded {len(idioms)} idioms with figurative meanings")

    print(f"Loading model {args.model}...")
    model = SentenceTransformer(args.model)

    print("Computing embeddings (one per figurative meaning)...")
    embeddings, metadata = compute_figurative_embeddings_separate(idioms, model)

    print(f"Saving to {args.embedding_output}...")
    save_embeddings_efficient(embeddings, metadata, args.embedding_output)

    print("Done!")


def main_find_pairs(args):
    """Main function for finding cross-lingual pairs."""
    print(f"Loading Chinese embeddings from {args.zh_embeddings}...")
    zh_metadata, zh_embeddings = load_embeddings_efficient(args.zh_embeddings)
    print(f"Loaded {len(zh_metadata)} Chinese idioms ({len(zh_embeddings)} embeddings)")

    print(f"Loading English embeddings from {args.en_embeddings}...")
    en_metadata, en_embeddings = load_embeddings_efficient(args.en_embeddings)
    print(f"Loaded {len(en_metadata)} English idioms ({len(en_embeddings)} embeddings)")

    top_k_str = str(args.top_k) if args.top_k else "unlimited"
    print(f"Finding cross-lingual pairs (threshold: {args.threshold}, top_k: {top_k_str})...")
    pairs = find_cross_lingual_pairs(
        zh_metadata, zh_embeddings,
        en_metadata, en_embeddings,
        similarity_threshold=args.threshold,
        top_k=args.top_k
    )
    print(f"Found {len(pairs)} pairs above threshold")

    print(f"Saving pairs to {args.pairs_output}...")
    save_pairs(pairs, args.pairs_output)

    print("Analyzing entity differences...")
    analysis = analyze_entity_differences(pairs)

    print(f"Saving analysis to {args.analysis_output}...")
    save_analysis(analysis, args.analysis_output)

    print_analysis_summary(analysis)

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-lingual idiom similarity analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Subcommand: compute embeddings
    embed_parser = subparsers.add_parser(
        "embed",
        help="Compute figurative meaning embeddings for idioms"
    )
    embed_parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSONL file with idioms"
    )
    embed_parser.add_argument(
        "--embedding_output", type=str, required=True,
        help="Output base path for embeddings (creates .npz and _meta.json)"
    )
    embed_parser.add_argument(
        "--model", type=str,
        default="/home/jiaruil5/math_rl/mix_teachers/r3lit_rl/models/Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer model to use"
    )

    # Subcommand: find pairs
    pairs_parser = subparsers.add_parser(
        "pairs",
        help="Find cross-lingual idiom pairs with similar meanings"
    )
    pairs_parser.add_argument(
        "--zh_embeddings", type=str, required=True,
        help="Chinese embedding base path (without .npz extension)"
    )
    pairs_parser.add_argument(
        "--en_embeddings", type=str, required=True,
        help="English embedding base path (without .npz extension)"
    )
    pairs_parser.add_argument(
        "--pairs_output", type=str, required=True,
        help="Output JSONL file for cross-lingual pairs"
    )
    pairs_parser.add_argument(
        "--analysis_output", type=str, required=True,
        help="Output JSON file for entity analysis"
    )
    pairs_parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Minimum cosine similarity threshold"
    )
    pairs_parser.add_argument(
        "--top_k", type=int, default=None,
        help="Optional: limit matches per idiom. If not set, keeps all above threshold."
    )

    args = parser.parse_args()

    if args.command == "embed":
        main_compute_embeddings(args)
    elif args.command == "pairs":
        main_find_pairs(args)
    else:
        parser.print_help()
