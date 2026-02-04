#!/usr/bin/env python3
"""
Intra-lingual idiom clustering with cross-lingual bridge.

This script:
1. Computes within-language embedding similarities for Chinese and English idioms
2. Uses cross-lingual pairs to bridge Chinese and English clusters
3. Creates semantic sets of idioms sharing the same meaning across both languages
4. Extracts the shared meaning, entities, and idioms for each set

Output format:
Each cluster contains:
- shared_meaning: The central/representative meaning of the cluster
- zh_idioms: List of Chinese idioms with their entities and figurative meanings
- en_idioms: List of English idioms with their entities and figurative meanings
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set
from tqdm import tqdm


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


def load_cross_lingual_pairs(jsonl_path: str) -> List[Dict]:
    """Load cross-lingual idiom pairs from JSONL file."""
    pairs = []
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def compute_intra_lingual_pairs(
    metadata: List[Dict],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.7
) -> List[Tuple[int, int, float]]:
    """
    Compute all pairs of idioms within a language with similarity >= threshold.

    Args:
        metadata: List of idiom metadata dicts
        embeddings: Embeddings array for all figurative meanings
        similarity_threshold: Minimum cosine similarity threshold

    Returns:
        List of tuples (idiom_idx1, idiom_idx2, max_similarity)
    """
    n_idioms = len(metadata)

    print(f"Computing intra-lingual similarities for {n_idioms} idioms...")

    # Precompute embedding ranges for faster access
    embedding_ranges = [(meta["embedding_start_idx"], meta["embedding_end_idx"])
                        for meta in metadata]

    # Generate all pairs (upper triangle only to avoid duplicates)
    total_pairs = n_idioms * (n_idioms - 1) // 2
    print(f"Processing {total_pairs} pairs...")

    pairs = []

    for i in tqdm(range(n_idioms)):
        i_start, i_end = embedding_ranges[i]
        i_embeddings = embeddings[i_start:i_end]

        for j in range(i + 1, n_idioms):
            j_start, j_end = embedding_ranges[j]
            j_embeddings = embeddings[j_start:j_end]

            # Compute max similarity across all meaning pairs
            sim_matrix = i_embeddings @ j_embeddings.T
            max_sim = sim_matrix.max()

            if max_sim >= similarity_threshold:
                pairs.append((i, j, float(max_sim)))

    print(f"Found {len(pairs)} pairs above threshold {similarity_threshold}")
    return pairs


def _compute_batch_similarities(args):
    """Worker function for parallel processing."""
    batch_indices, embeddings, embedding_ranges, similarity_threshold = args
    results = []
    for i, j in batch_indices:
        i_start, i_end = embedding_ranges[i]
        j_start, j_end = embedding_ranges[j]
        sim_matrix = embeddings[i_start:i_end] @ embeddings[j_start:j_end].T
        max_sim = sim_matrix.max()
        if max_sim >= similarity_threshold:
            results.append((int(i), int(j), float(max_sim)))
    return results


def compute_intra_lingual_pairs_parallel(
    metadata: List[Dict],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.7,
    n_workers: int = 8,
    batch_size: int = 5000
) -> List[Tuple[int, int, float]]:
    """
    Parallel version using multiprocessing for very large datasets.

    Args:
        metadata: List of idiom metadata dicts
        embeddings: Embeddings array for all figurative meanings
        similarity_threshold: Minimum cosine similarity threshold
        n_workers: Number of parallel workers
        batch_size: Number of candidate pairs per batch

    Returns:
        List of tuples (idiom_idx1, idiom_idx2, max_similarity)
    """
    n_idioms = len(metadata)

    print(f"Computing intra-lingual similarities for {n_idioms} idioms (parallel, {n_workers} workers)...")

    # Precompute embedding ranges
    embedding_ranges = [(meta["embedding_start_idx"], meta["embedding_end_idx"])
                        for meta in metadata]

    # Generate all pairs (upper triangle only)
    all_pairs = [(i, j) for i in range(n_idioms) for j in range(i + 1, n_idioms)]
    print(f"Processing {len(all_pairs)} pairs...")

    # Split into batches
    batches = [all_pairs[i:i + batch_size] for i in range(0, len(all_pairs), batch_size)]

    print(f"Processing {len(batches)} batches with {n_workers} workers...")
    pairs = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_compute_batch_similarities,
                          (batch, embeddings, embedding_ranges, similarity_threshold))
            for batch in batches
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            pairs.extend(future.result())

    print(f"Found {len(pairs)} pairs above threshold {similarity_threshold}")
    return pairs


def build_adjacency_sets(pairs: List[Tuple[int, int, float]]) -> Dict[int, Set[int]]:
    """Build adjacency sets from pairs."""
    neighbors = defaultdict(set)
    for i, j, sim in pairs:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return dict(neighbors)


def find_cliques_greedy(neighbors: Dict[int, Set[int]], n_items: int) -> List[Set[int]]:
    """
    Find maximal cliques using greedy expansion.

    Only groups idioms that are ALL mutually similar (no transitive closure).
    """
    used = set()
    cliques = []

    # Sort by degree (most connected first)
    items_by_degree = sorted(
        range(n_items),
        key=lambda x: len(neighbors.get(x, set())),
        reverse=True
    )

    for seed in items_by_degree:
        if seed in used:
            continue

        # Start clique with seed
        clique = {seed}
        candidates = neighbors.get(seed, set()) - used

        # Sort candidates by degree
        sorted_candidates = sorted(
            candidates,
            key=lambda x: len(neighbors.get(x, set())),
            reverse=True
        )

        # Add candidates only if connected to ALL current clique members
        for cand in sorted_candidates:
            cand_neighbors = neighbors.get(cand, set())
            if all(member in cand_neighbors for member in clique):
                clique.add(cand)

        if len(clique) > 1:
            cliques.append(clique)
            used.update(clique)

    return cliques


def find_meaning_embedding_idx(metadata: Dict, target_meaning: str) -> int:
    """
    Find the embedding index for a specific meaning within an idiom's embeddings.

    Args:
        metadata: Idiom metadata with figurative_meanings and embedding indices
        target_meaning: The specific meaning to find

    Returns:
        Embedding index if found, -1 otherwise
    """
    meanings = metadata.get("figurative_meanings", [])
    start_idx = metadata.get("embedding_start_idx", 0)

    for i, meaning in enumerate(meanings):
        if meaning == target_meaning:
            return start_idx + i

    # If exact match not found, try partial match
    for i, meaning in enumerate(meanings):
        if target_meaning in meaning or meaning in target_meaning:
            return start_idx + i

    # Fallback to first meaning
    return start_idx if meanings else -1


def build_combined_clusters(
    zh_metadata: List[Dict],
    zh_embeddings: np.ndarray,
    en_metadata: List[Dict],
    en_embeddings: np.ndarray,
    cross_lingual_pairs: List[Dict],
    similarity_threshold: float = 0.95
) -> List[Dict]:
    """
    Build clusters using cross-lingual pairs as anchors.

    Each cluster is centered on a cross-lingual pair, expanded ONLY with idioms
    similar to the SPECIFIC matched meaning (not any meaning).

    This prevents the issue where idioms with multiple meanings get grouped
    based on an unrelated secondary meaning.

    Args:
        zh_metadata: Chinese idiom metadata
        zh_embeddings: Chinese embedding array
        en_metadata: English idiom metadata
        en_embeddings: English embedding array
        cross_lingual_pairs: Cross-lingual pairs from the JSONL file
        similarity_threshold: Threshold for meaning similarity

    Returns:
        List of cluster dicts with shared_meaning, zh_idioms, en_idioms
    """
    # Create idiom to index mappings
    zh_idiom_to_idx = {meta["idiom"]: i for i, meta in enumerate(zh_metadata)}
    en_idiom_to_idx = {meta["idiom"]: i for i, meta in enumerate(en_metadata)}

    # Build bilingual clusters using cross-lingual pairs as anchors
    print(f"Building clusters from {len(cross_lingual_pairs)} cross-lingual pairs...")
    print(f"Using similarity threshold: {similarity_threshold}")

    bilingual_clusters = []
    used_zh_idioms = set()
    used_en_idioms = set()

    for pair in tqdm(cross_lingual_pairs, desc="Processing cross-lingual pairs"):
        zh_idiom = pair["zh_idiom"]
        en_idiom = pair["en_idiom"]
        zh_matched_meaning = pair.get("zh_matched_meaning", "")
        en_matched_meaning = pair.get("en_matched_meaning", "")

        if zh_idiom not in zh_idiom_to_idx or en_idiom not in en_idiom_to_idx:
            continue

        zh_idx = zh_idiom_to_idx[zh_idiom]
        en_idx = en_idiom_to_idx[en_idiom]

        # Skip if anchor idioms already used
        if zh_idx in used_zh_idioms or en_idx in used_en_idioms:
            continue

        # Find embedding index for the specific matched meanings
        zh_meaning_emb_idx = find_meaning_embedding_idx(zh_metadata[zh_idx], zh_matched_meaning)
        en_meaning_emb_idx = find_meaning_embedding_idx(en_metadata[en_idx], en_matched_meaning)

        if zh_meaning_emb_idx < 0 or en_meaning_emb_idx < 0:
            continue

        zh_anchor_emb = zh_embeddings[zh_meaning_emb_idx]
        en_anchor_emb = en_embeddings[en_meaning_emb_idx]

        # Start with anchor idioms
        zh_indices = {zh_idx}
        en_indices = {en_idx}

        # Find Chinese idioms with meanings similar to zh_matched_meaning
        for other_idx, other_meta in enumerate(zh_metadata):
            if other_idx == zh_idx or other_idx in used_zh_idioms:
                continue

            other_start = other_meta.get("embedding_start_idx", 0)
            other_end = other_meta.get("embedding_end_idx", other_start + 1)
            other_embs = zh_embeddings[other_start:other_end]

            # Compute similarity between anchor meaning and all of other's meanings
            sims = other_embs @ zh_anchor_emb
            max_sim = sims.max() if len(sims) > 0 else 0

            if max_sim >= similarity_threshold:
                zh_indices.add(other_idx)

        # Find English idioms with meanings similar to en_matched_meaning
        for other_idx, other_meta in enumerate(en_metadata):
            if other_idx == en_idx or other_idx in used_en_idioms:
                continue

            other_start = other_meta.get("embedding_start_idx", 0)
            other_end = other_meta.get("embedding_end_idx", other_start + 1)
            other_embs = en_embeddings[other_start:other_end]

            # Compute similarity between anchor meaning and all of other's meanings
            sims = other_embs @ en_anchor_emb
            max_sim = sims.max() if len(sims) > 0 else 0

            if max_sim >= similarity_threshold:
                en_indices.add(other_idx)

        # Mark all idioms as used
        used_zh_idioms.update(zh_indices)
        used_en_idioms.update(en_indices)

        bilingual_clusters.append({
            "zh_indices": list(zh_indices),
            "en_indices": list(en_indices),
            "anchor_pair": pair
        })

    # For monolingual clusters, we skip them since they're less important
    # and would require a different approach (no anchor meaning to compare to)
    zh_only_clusters = []
    en_only_clusters = []

    print(f"Bilingual clusters: {len(bilingual_clusters)}")
    print(f"Chinese-only clusters: {len(zh_only_clusters)}")
    print(f"English-only clusters: {len(en_only_clusters)}")

    return bilingual_clusters, zh_only_clusters, en_only_clusters


def extract_shared_meaning(
    zh_idioms_data: List[Dict],
    en_idioms_data: List[Dict],
    anchor_pair: Dict = None
) -> str:
    """
    Extract the shared meaning from a cluster.

    Uses the anchor cross-lingual pair's matched meaning as the authoritative
    shared meaning, since the cluster is built around this pair.

    Args:
        zh_idioms_data: List of Chinese idiom data in the cluster
        en_idioms_data: List of English idiom data in the cluster
        anchor_pair: The cross-lingual pair that anchors this cluster

    Returns:
        The most representative shared meaning
    """
    # Use the anchor pair's matched meaning (most reliable)
    if anchor_pair:
        # Prefer English matched meaning as it's more readable
        en_meaning = anchor_pair.get("en_matched_meaning", "")
        if en_meaning:
            return en_meaning
        zh_meaning = anchor_pair.get("zh_matched_meaning", "")
        if zh_meaning:
            return zh_meaning

    # Fallback: use the first figurative meaning from English (or Chinese if no English)
    if en_idioms_data and en_idioms_data[0].get("figurative_meanings"):
        return en_idioms_data[0]["figurative_meanings"][0]
    if zh_idioms_data and zh_idioms_data[0].get("figurative_meanings"):
        return zh_idioms_data[0]["figurative_meanings"][0]

    return ""


def format_cluster_output(
    cluster_data: Dict,
    zh_metadata: List[Dict],
    en_metadata: List[Dict]
) -> Dict:
    """
    Format a cluster for output.

    Args:
        cluster_data: Dict with zh_indices, en_indices, and optional anchor_pair
        zh_metadata: Chinese idiom metadata
        en_metadata: English idiom metadata

    Returns:
        Formatted cluster dict
    """
    zh_indices = cluster_data.get("zh_indices", [])
    en_indices = cluster_data.get("en_indices", [])
    anchor_pair = cluster_data.get("anchor_pair")

    zh_idioms_data = [zh_metadata[i] for i in zh_indices]
    en_idioms_data = [en_metadata[i] for i in en_indices] if en_indices else []

    # Extract shared meaning from anchor pair
    shared_meaning = extract_shared_meaning(
        zh_idioms_data, en_idioms_data, anchor_pair
    )

    # Format Chinese idioms
    zh_idioms = []
    for data in zh_idioms_data:
        zh_idioms.append({
            "idiom": data["idiom"],
            "entities": data.get("entities", []),
            "figurative_meanings": data.get("figurative_meanings", []),
            "literal_meanings": data.get("literal_meanings", [])
        })

    # Format English idioms
    en_idioms = []
    for data in en_idioms_data:
        en_idioms.append({
            "idiom": data["idiom"],
            "entities": data.get("entities", []),
            "figurative_meanings": data.get("figurative_meanings", []),
            "literal_meanings": data.get("literal_meanings", [])
        })

    result = {
        "shared_meaning": shared_meaning,
        "zh_idiom_count": len(zh_idioms),
        "en_idiom_count": len(en_idioms),
        "zh_idioms": zh_idioms,
        "en_idioms": en_idioms
    }

    # Add anchor idioms if this is a bilingual cluster
    if anchor_pair:
        result["anchor_zh_idiom"] = anchor_pair.get("zh_idiom", "")
        result["anchor_en_idiom"] = anchor_pair.get("en_idiom", "")

    return result


def save_clusters_jsonl(clusters: List[Dict], output_path: str):
    """Save clusters to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        for cluster in clusters:
            f.write(json.dumps(cluster, ensure_ascii=False) + "\n")

    print(f"Saved {len(clusters)} clusters to {output_path}")


def save_clusters_json(clusters: List[Dict], output_path: str):
    """Save clusters to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(clusters)} clusters to {output_path}")


def save_intra_lingual_pairs(
    pairs: List[Tuple[int, int, float]],
    metadata: List[Dict],
    output_path: str
):
    """Save intra-lingual pairs to JSONL for inspection."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf8") as f:
        for idx1, idx2, sim in pairs:
            pair_data = {
                "idiom1": metadata[idx1]["idiom"],
                "idiom2": metadata[idx2]["idiom"],
                "similarity": sim,
                "idiom1_figurative_meanings": metadata[idx1].get("figurative_meanings", []),
                "idiom2_figurative_meanings": metadata[idx2].get("figurative_meanings", []),
                "idiom1_entities": metadata[idx1].get("entities", []),
                "idiom2_entities": metadata[idx2].get("entities", [])
            }
            f.write(json.dumps(pair_data, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} pairs to {output_path}")


def print_cluster_summary(bilingual_clusters: List[Dict], zh_only: List[Dict], en_only: List[Dict]):
    """Print summary statistics about clusters."""
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)

    if bilingual_clusters:
        zh_counts = [c["zh_idiom_count"] for c in bilingual_clusters]
        en_counts = [c["en_idiom_count"] for c in bilingual_clusters]
        print(f"\nBilingual clusters: {len(bilingual_clusters)}")
        print(f"  Chinese idioms per cluster: min={min(zh_counts)}, max={max(zh_counts)}, avg={sum(zh_counts)/len(zh_counts):.2f}")
        print(f"  English idioms per cluster: min={min(en_counts)}, max={max(en_counts)}, avg={sum(en_counts)/len(en_counts):.2f}")
        print(f"  Total Chinese idioms in bilingual clusters: {sum(zh_counts)}")
        print(f"  Total English idioms in bilingual clusters: {sum(en_counts)}")

    if zh_only:
        zh_counts = [c.get("zh_idiom_count", 0) for c in zh_only]
        if zh_counts and max(zh_counts) > 0:
            print(f"\nChinese-only clusters: {len(zh_only)}")
            print(f"  Idioms per cluster: min={min(zh_counts)}, max={max(zh_counts)}, avg={sum(zh_counts)/len(zh_counts):.2f}")
            print(f"  Total Chinese idioms in Chinese-only clusters: {sum(zh_counts)}")

    if en_only:
        en_counts = [c.get("en_idiom_count", 0) for c in en_only]
        if en_counts and max(en_counts) > 0:
            print(f"\nEnglish-only clusters: {len(en_only)}")
            print(f"  Idioms per cluster: min={min(en_counts)}, max={max(en_counts)}, avg={sum(en_counts)/len(en_counts):.2f}")
            print(f"  Total English idioms in English-only clusters: {sum(en_counts)}")


def main(args):
    """Main function."""
    # Load embeddings
    print(f"Loading Chinese embeddings from {args.zh_embeddings}...")
    zh_metadata, zh_embeddings = load_embeddings_efficient(args.zh_embeddings)
    print(f"Loaded {len(zh_metadata)} Chinese idioms ({len(zh_embeddings)} embeddings)")

    print(f"\nLoading English embeddings from {args.en_embeddings}...")
    en_metadata, en_embeddings = load_embeddings_efficient(args.en_embeddings)
    print(f"Loaded {len(en_metadata)} English idioms ({len(en_embeddings)} embeddings)")

    # Load cross-lingual pairs
    print(f"\nLoading cross-lingual pairs from {args.cross_lingual_pairs}...")
    cross_lingual_pairs = load_cross_lingual_pairs(args.cross_lingual_pairs)
    print(f"Loaded {len(cross_lingual_pairs)} cross-lingual pairs")

    # Compute or load intra-lingual pairs
    zh_pairs_path = os.path.join(args.output_dir, "zh_intra_lingual_pairs.jsonl")
    en_pairs_path = os.path.join(args.output_dir, "en_intra_lingual_pairs.jsonl")

    if args.recompute_intra or not (os.path.exists(zh_pairs_path) and os.path.exists(en_pairs_path)):
        # Select computation function based on --parallel flag
        compute_fn = (compute_intra_lingual_pairs_parallel if args.parallel
                      else compute_intra_lingual_pairs)
        compute_kwargs = {"n_workers": args.n_workers} if args.parallel else {}

        # Compute Chinese intra-lingual pairs
        print(f"\nComputing Chinese intra-lingual pairs (threshold={args.threshold})...")
        zh_intra_pairs = compute_fn(
            zh_metadata, zh_embeddings, args.threshold, **compute_kwargs
        )
        save_intra_lingual_pairs(zh_intra_pairs, zh_metadata, zh_pairs_path)

        # Compute English intra-lingual pairs
        print(f"\nComputing English intra-lingual pairs (threshold={args.threshold})...")
        en_intra_pairs = compute_fn(
            en_metadata, en_embeddings, args.threshold, **compute_kwargs
        )
        save_intra_lingual_pairs(en_intra_pairs, en_metadata, en_pairs_path)
    else:
        # Load existing pairs
        print(f"\nLoading existing intra-lingual pairs from {args.output_dir}...")
        zh_intra_pairs = []
        with open(zh_pairs_path, "r", encoding="utf8") as f:
            zh_idiom_to_idx = {meta["idiom"]: i for i, meta in enumerate(zh_metadata)}
            for line in f:
                data = json.loads(line)
                idx1 = zh_idiom_to_idx.get(data["idiom1"])
                idx2 = zh_idiom_to_idx.get(data["idiom2"])
                if idx1 is not None and idx2 is not None:
                    zh_intra_pairs.append((idx1, idx2, data["similarity"]))
        print(f"Loaded {len(zh_intra_pairs)} Chinese intra-lingual pairs")

        en_intra_pairs = []
        with open(en_pairs_path, "r", encoding="utf8") as f:
            en_idiom_to_idx = {meta["idiom"]: i for i, meta in enumerate(en_metadata)}
            for line in f:
                data = json.loads(line)
                idx1 = en_idiom_to_idx.get(data["idiom1"])
                idx2 = en_idiom_to_idx.get(data["idiom2"])
                if idx1 is not None and idx2 is not None:
                    en_intra_pairs.append((idx1, idx2, data["similarity"]))
        print(f"Loaded {len(en_intra_pairs)} English intra-lingual pairs")

    # Build combined clusters
    print("\nBuilding combined clusters...")
    bilingual_raw, zh_only_raw, en_only_raw = build_combined_clusters(
        zh_metadata, zh_embeddings,
        en_metadata, en_embeddings,
        cross_lingual_pairs,
        similarity_threshold=args.threshold
    )

    # Format clusters for output
    print("\nFormatting bilingual clusters...")
    bilingual_clusters = []
    for cluster_data in tqdm(bilingual_raw):
        formatted = format_cluster_output(
            cluster_data, zh_metadata, en_metadata
        )
        bilingual_clusters.append(formatted)

    # Sort by total idiom count (largest first)
    bilingual_clusters.sort(key=lambda x: x["zh_idiom_count"] + x["en_idiom_count"], reverse=True)

    # Format Chinese-only clusters
    print("\nFormatting Chinese-only clusters...")
    zh_only_clusters = []
    for cluster_data in tqdm(zh_only_raw):
        formatted = format_cluster_output(
            cluster_data, zh_metadata, en_metadata
        )
        zh_only_clusters.append(formatted)
    zh_only_clusters.sort(key=lambda x: x["zh_idiom_count"], reverse=True)

    # Format English-only clusters
    print("\nFormatting English-only clusters...")
    en_only_clusters = []
    for cluster_data in tqdm(en_only_raw):
        formatted = format_cluster_output(
            cluster_data, zh_metadata, en_metadata
        )
        en_only_clusters.append(formatted)
    en_only_clusters.sort(key=lambda x: x["en_idiom_count"], reverse=True)

    # Print summary
    print_cluster_summary(bilingual_clusters, zh_only_clusters, en_only_clusters)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Save bilingual clusters
    bilingual_jsonl = os.path.join(args.output_dir, "bilingual_clusters.jsonl")
    bilingual_json = os.path.join(args.output_dir, "bilingual_clusters.json")
    save_clusters_jsonl(bilingual_clusters, bilingual_jsonl)
    save_clusters_json(bilingual_clusters, bilingual_json)

    # Save Chinese-only clusters
    if zh_only_clusters:
        zh_only_jsonl = os.path.join(args.output_dir, "zh_only_clusters.jsonl")
        save_clusters_jsonl(zh_only_clusters, zh_only_jsonl)

    # Save English-only clusters
    if en_only_clusters:
        en_only_jsonl = os.path.join(args.output_dir, "en_only_clusters.jsonl")
        save_clusters_jsonl(en_only_clusters, en_only_jsonl)

    # Print example clusters
    print("\n" + "=" * 60)
    print("EXAMPLE BILINGUAL CLUSTERS")
    print("=" * 60)
    for i, cluster in enumerate(bilingual_clusters[:5], 1):
        print(f"\n--- Cluster {i} ---")
        print(f"Anchor: {cluster.get('anchor_zh_idiom', 'N/A')} <-> {cluster.get('anchor_en_idiom', 'N/A')}")
        print(f"Shared meaning: {cluster['shared_meaning']}")
        print(f"Chinese idioms ({cluster['zh_idiom_count']}):")
        for zh in cluster["zh_idioms"][:3]:
            print(f"  - {zh['idiom']} (entities: {zh['entities']})")
        if cluster["zh_idiom_count"] > 3:
            print(f"  ... and {cluster['zh_idiom_count'] - 3} more")
        print(f"English idioms ({cluster['en_idiom_count']}):")
        for en in cluster["en_idioms"][:3]:
            print(f"  - {en['idiom']} (entities: {en['entities']})")
        if cluster["en_idiom_count"] > 3:
            print(f"  ... and {cluster['en_idiom_count'] - 3} more")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intra-lingual idiom clustering with cross-lingual bridge"
    )
    parser.add_argument(
        "--zh_embeddings", type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/zh/figurative_embeddings",
        help="Chinese embedding base path (without .npz extension)"
    )
    parser.add_argument(
        "--en_embeddings", type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/en/figurative_embeddings",
        help="English embedding base path (without .npz extension)"
    )
    parser.add_argument(
        "--cross_lingual_pairs", type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/cross_lingual_pairs.jsonl",
        help="Path to cross-lingual pairs JSONL file"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/semantic_clusters",
        help="Output directory for cluster files"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help="Minimum cosine similarity threshold for intra-lingual pairs"
    )
    parser.add_argument(
        "--recompute_intra", action="store_true",
        help="Force recomputation of intra-lingual pairs even if cached files exist"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Use parallel processing for computing intra-lingual pairs"
    )
    parser.add_argument(
        "--n_workers", type=int, default=8,
        help="Number of parallel workers (only used with --parallel)"
    )

    args = parser.parse_args()
    main(args)
