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
from sklearn.metrics.pairwise import cosine_similarity


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
    Compute all pairs of idioms within a language that have similarity >= threshold.

    Args:
        metadata: List of idiom metadata dicts
        embeddings: Embeddings array for all figurative meanings
        similarity_threshold: Minimum cosine similarity threshold

    Returns:
        List of tuples (idiom_idx1, idiom_idx2, max_similarity)
    """
    n_idioms = len(metadata)
    pairs = []

    print(f"Computing intra-lingual similarities for {n_idioms} idioms...")

    # Compute similarity in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(n_idioms)):
        i_start = metadata[i]["embedding_start_idx"]
        i_end = metadata[i]["embedding_end_idx"]
        i_embeddings = embeddings[i_start:i_end]

        # Only compare with idioms that come after this one to avoid duplicates
        for j in range(i + 1, n_idioms):
            j_start = metadata[j]["embedding_start_idx"]
            j_end = metadata[j]["embedding_end_idx"]
            j_embeddings = embeddings[j_start:j_end]

            # Compute max similarity across all meaning pairs
            sim_matrix = cosine_similarity(i_embeddings, j_embeddings)
            max_sim = sim_matrix.max()

            if max_sim >= similarity_threshold:
                pairs.append((i, j, float(max_sim)))

    print(f"Found {len(pairs)} pairs above threshold {similarity_threshold}")
    return pairs


def compute_intra_lingual_pairs_efficient(
    metadata: List[Dict],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.7,
    batch_size: int = 1000
) -> List[Tuple[int, int, float]]:
    """
    Efficiently compute all pairs of idioms within a language with similarity >= threshold.

    Uses vectorized operations and batched computation for speed.

    Args:
        metadata: List of idiom metadata dicts
        embeddings: Embeddings array for all figurative meanings
        similarity_threshold: Minimum cosine similarity threshold
        batch_size: Number of candidate pairs to process in each batch

    Returns:
        List of tuples (idiom_idx1, idiom_idx2, max_similarity)
    """
    n_idioms = len(metadata)

    print(f"Computing intra-lingual similarities for {n_idioms} idioms...")

    # For each idiom, get its "representative" embedding (mean of all meanings)
    representative_embeddings = np.zeros((n_idioms, embeddings.shape[1]), dtype=np.float32)
    for i, meta in enumerate(metadata):
        start_idx = meta["embedding_start_idx"]
        end_idx = meta["embedding_end_idx"]
        representative_embeddings[i] = embeddings[start_idx:end_idx].mean(axis=0)

    # Normalize for cosine similarity
    norms = np.linalg.norm(representative_embeddings, axis=1, keepdims=True)
    representative_embeddings = representative_embeddings / (norms + 1e-8)

    # Compute full similarity matrix for representative embeddings
    print("Computing representative similarity matrix...")
    rep_sim_matrix = cosine_similarity(representative_embeddings)

    # Find all candidate pairs using vectorized operations (upper triangle only)
    margin = 0.15
    candidate_threshold = similarity_threshold - margin

    print("Finding candidate pairs (vectorized)...")
    # Get upper triangle indices where similarity >= candidate_threshold
    upper_tri = np.triu(rep_sim_matrix, k=1)
    candidate_i, candidate_j = np.where(upper_tri >= candidate_threshold)
    print(f"Found {len(candidate_i)} candidate pairs to verify")

    if len(candidate_i) == 0:
        return []

    # Precompute embedding ranges for faster access
    embedding_ranges = [(meta["embedding_start_idx"], meta["embedding_end_idx"])
                        for meta in metadata]

    # Process candidates in batches
    print("Computing exact similarities for candidates...")
    pairs = []

    for batch_start in tqdm(range(0, len(candidate_i), batch_size)):
        batch_end = min(batch_start + batch_size, len(candidate_i))
        batch_i = candidate_i[batch_start:batch_end]
        batch_j = candidate_j[batch_start:batch_end]

        for idx in range(len(batch_i)):
            i, j = batch_i[idx], batch_j[idx]
            i_start, i_end = embedding_ranges[i]
            j_start, j_end = embedding_ranges[j]

            # Compute max similarity across all meaning pairs
            # Use dot product since embeddings are already normalized
            sim_matrix = embeddings[i_start:i_end] @ embeddings[j_start:j_end].T
            max_sim = sim_matrix.max()

            if max_sim >= similarity_threshold:
                pairs.append((int(i), int(j), float(max_sim)))

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

    # Compute representative embeddings
    representative_embeddings = np.zeros((n_idioms, embeddings.shape[1]), dtype=np.float32)
    for i, meta in enumerate(metadata):
        start_idx = meta["embedding_start_idx"]
        end_idx = meta["embedding_end_idx"]
        representative_embeddings[i] = embeddings[start_idx:end_idx].mean(axis=0)

    norms = np.linalg.norm(representative_embeddings, axis=1, keepdims=True)
    representative_embeddings = representative_embeddings / (norms + 1e-8)

    print("Computing representative similarity matrix...")
    rep_sim_matrix = cosine_similarity(representative_embeddings)

    margin = 0.15
    candidate_threshold = similarity_threshold - margin

    print("Finding candidate pairs...")
    upper_tri = np.triu(rep_sim_matrix, k=1)
    candidate_i, candidate_j = np.where(upper_tri >= candidate_threshold)
    candidates = list(zip(candidate_i, candidate_j))
    print(f"Found {len(candidates)} candidate pairs to verify")

    if not candidates:
        return []

    embedding_ranges = [(meta["embedding_start_idx"], meta["embedding_end_idx"])
                        for meta in metadata]

    # Split into batches
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]

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


class UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_clusters(self) -> Dict[str, Set[str]]:
        """Get all clusters as a dict of root -> set of members."""
        clusters = defaultdict(set)
        for x in self.parent:
            root = self.find(x)
            clusters[root].add(x)
        return dict(clusters)


def build_combined_clusters(
    zh_metadata: List[Dict],
    zh_intra_pairs: List[Tuple[int, int, float]],
    en_metadata: List[Dict],
    en_intra_pairs: List[Tuple[int, int, float]],
    cross_lingual_pairs: List[Dict]
) -> List[Dict]:
    """
    Build clusters by combining intra-lingual and cross-lingual connections.

    Args:
        zh_metadata: Chinese idiom metadata
        zh_intra_pairs: Chinese intra-lingual pairs (idx1, idx2, sim)
        en_metadata: English idiom metadata
        en_intra_pairs: English intra-lingual pairs (idx1, idx2, sim)
        cross_lingual_pairs: Cross-lingual pairs from the JSONL file

    Returns:
        List of cluster dicts with shared_meaning, zh_idioms, en_idioms
    """
    # Create idiom to index mappings
    zh_idiom_to_idx = {meta["idiom"]: i for i, meta in enumerate(zh_metadata)}
    en_idiom_to_idx = {meta["idiom"]: i for i, meta in enumerate(en_metadata)}

    # Initialize Union-Find with prefixed keys to distinguish languages
    uf = UnionFind()

    # Add all idioms to union-find
    for i in range(len(zh_metadata)):
        uf.find(f"zh_{i}")
    for i in range(len(en_metadata)):
        uf.find(f"en_{i}")

    # Add intra-Chinese edges
    print(f"Adding {len(zh_intra_pairs)} Chinese intra-lingual edges...")
    for idx1, idx2, sim in zh_intra_pairs:
        uf.union(f"zh_{idx1}", f"zh_{idx2}")

    # Add intra-English edges
    print(f"Adding {len(en_intra_pairs)} English intra-lingual edges...")
    for idx1, idx2, sim in en_intra_pairs:
        uf.union(f"en_{idx1}", f"en_{idx2}")

    # Add cross-lingual edges
    print(f"Adding {len(cross_lingual_pairs)} cross-lingual edges...")
    cross_lingual_edges_added = 0
    for pair in cross_lingual_pairs:
        zh_idiom = pair["zh_idiom"]
        en_idiom = pair["en_idiom"]

        if zh_idiom in zh_idiom_to_idx and en_idiom in en_idiom_to_idx:
            zh_idx = zh_idiom_to_idx[zh_idiom]
            en_idx = en_idiom_to_idx[en_idiom]
            uf.union(f"zh_{zh_idx}", f"en_{en_idx}")
            cross_lingual_edges_added += 1

    print(f"Cross-lingual edges successfully added: {cross_lingual_edges_added}")

    # Get clusters
    raw_clusters = uf.get_clusters()
    print(f"Total clusters found: {len(raw_clusters)}")

    # Filter and format clusters that have both Chinese and English idioms
    bilingual_clusters = []
    zh_only_clusters = []
    en_only_clusters = []

    for root, members in raw_clusters.items():
        zh_indices = [int(m[3:]) for m in members if m.startswith("zh_")]
        en_indices = [int(m[3:]) for m in members if m.startswith("en_")]

        if zh_indices and en_indices:
            # Bilingual cluster
            bilingual_clusters.append({
                "zh_indices": zh_indices,
                "en_indices": en_indices
            })
        elif zh_indices and len(zh_indices) > 1:
            # Chinese-only cluster with multiple idioms
            zh_only_clusters.append({"zh_indices": zh_indices})
        elif en_indices and len(en_indices) > 1:
            # English-only cluster with multiple idioms
            en_only_clusters.append({"en_indices": en_indices})

    print(f"Bilingual clusters: {len(bilingual_clusters)}")
    print(f"Chinese-only clusters (2+ idioms): {len(zh_only_clusters)}")
    print(f"English-only clusters (2+ idioms): {len(en_only_clusters)}")

    return bilingual_clusters, zh_only_clusters, en_only_clusters


def extract_shared_meaning(
    zh_idioms_data: List[Dict],
    en_idioms_data: List[Dict],
    cross_lingual_pairs: List[Dict]
) -> str:
    """
    Extract the shared meaning from a cluster.

    Uses the matched meanings from cross-lingual pairs as they represent
    the semantic overlap between languages.

    Args:
        zh_idioms_data: List of Chinese idiom data in the cluster
        en_idioms_data: List of English idiom data in the cluster
        cross_lingual_pairs: All cross-lingual pairs (for finding matched meanings)

    Returns:
        The most representative shared meaning
    """
    # Create sets of idioms in this cluster
    zh_idioms = {d["idiom"] for d in zh_idioms_data}
    en_idioms = {d["idiom"] for d in en_idioms_data}

    # Find all matched meanings from cross-lingual pairs in this cluster
    matched_meanings = []
    for pair in cross_lingual_pairs:
        if pair["zh_idiom"] in zh_idioms and pair["en_idiom"] in en_idioms:
            matched_meanings.append(pair.get("en_matched_meaning", ""))
            matched_meanings.append(pair.get("zh_matched_meaning", ""))

    # If we have matched meanings, use the most common one
    if matched_meanings:
        meaning_counts = defaultdict(int)
        for m in matched_meanings:
            if m:
                meaning_counts[m] += 1
        if meaning_counts:
            return max(meaning_counts.items(), key=lambda x: x[1])[0]

    # Fallback: use the first figurative meaning from English (or Chinese if no English)
    if en_idioms_data and en_idioms_data[0].get("figurative_meanings"):
        return en_idioms_data[0]["figurative_meanings"][0]
    if zh_idioms_data and zh_idioms_data[0].get("figurative_meanings"):
        return zh_idioms_data[0]["figurative_meanings"][0]

    return ""


def format_cluster_output(
    cluster_data: Dict,
    zh_metadata: List[Dict],
    en_metadata: List[Dict],
    cross_lingual_pairs: List[Dict]
) -> Dict:
    """
    Format a cluster for output.

    Args:
        cluster_data: Dict with zh_indices and en_indices
        zh_metadata: Chinese idiom metadata
        en_metadata: English idiom metadata
        cross_lingual_pairs: Cross-lingual pairs for extracting shared meaning

    Returns:
        Formatted cluster dict
    """
    zh_indices = cluster_data.get("zh_indices", [])
    en_indices = cluster_data.get("en_indices", [])

    zh_idioms_data = [zh_metadata[i] for i in zh_indices]
    en_idioms_data = [en_metadata[i] for i in en_indices] if en_indices else []

    # Extract shared meaning
    shared_meaning = extract_shared_meaning(
        zh_idioms_data, en_idioms_data, cross_lingual_pairs
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

    return {
        "shared_meaning": shared_meaning,
        "zh_idiom_count": len(zh_idioms),
        "en_idiom_count": len(en_idioms),
        "zh_idioms": zh_idioms,
        "en_idioms": en_idioms
    }


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
        zh_counts = [len(c.get("zh_indices", [])) for c in zh_only]
        print(f"\nChinese-only clusters: {len(zh_only)}")
        print(f"  Idioms per cluster: min={min(zh_counts)}, max={max(zh_counts)}, avg={sum(zh_counts)/len(zh_counts):.2f}")

    if en_only:
        en_counts = [len(c.get("en_indices", [])) for c in en_only]
        print(f"\nEnglish-only clusters: {len(en_only)}")
        print(f"  Idioms per cluster: min={min(en_counts)}, max={max(en_counts)}, avg={sum(en_counts)/len(en_counts):.2f}")


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
                      else compute_intra_lingual_pairs_efficient)
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
        zh_metadata, zh_intra_pairs,
        en_metadata, en_intra_pairs,
        cross_lingual_pairs
    )

    # Format clusters for output
    print("\nFormatting bilingual clusters...")
    bilingual_clusters = []
    for cluster_data in tqdm(bilingual_raw):
        formatted = format_cluster_output(
            cluster_data, zh_metadata, en_metadata, cross_lingual_pairs
        )
        bilingual_clusters.append(formatted)

    # Sort by total idiom count (largest first)
    bilingual_clusters.sort(key=lambda x: x["zh_idiom_count"] + x["en_idiom_count"], reverse=True)

    # Format Chinese-only clusters
    print("\nFormatting Chinese-only clusters...")
    zh_only_clusters = []
    for cluster_data in tqdm(zh_only_raw):
        formatted = format_cluster_output(
            cluster_data, zh_metadata, en_metadata, cross_lingual_pairs
        )
        zh_only_clusters.append(formatted)
    zh_only_clusters.sort(key=lambda x: x["zh_idiom_count"], reverse=True)

    # Format English-only clusters
    print("\nFormatting English-only clusters...")
    en_only_clusters = []
    for cluster_data in tqdm(en_only_raw):
        formatted = format_cluster_output(
            cluster_data, zh_metadata, en_metadata, cross_lingual_pairs
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
        "--threshold", type=float, default=0.7,
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
