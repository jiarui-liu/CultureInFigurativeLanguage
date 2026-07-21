#!/usr/bin/env python3
"""
Consolidate a culture CPT corpus into plain JSONL shards for training.

The published corpora ship as gzipped JSONL:
    <DATA_ROOT>/<dataset>/data/tagged_*.json.gz
one JSON object per line with (at least) a "text" field, e.g.
    {"text": "<document>\n\n<knowledge block>...", "source", "matched_idioms", ...}

LLaMA-Factory's local loader infers the dataset builder from the file
extension and does NOT recognise the double extension ".json.gz" — so we
decompress and reshard into plain "train_*.jsonl" that its json builder reads
directly. We keep only the fields training needs (text + light provenance) and
shuffle globally across shards. No encryption — culture CPT trains on plaintext.

Datasets prepared with this script (see configs/dataset_info.json):
  hi-proverbs-cpt          -> <DATA_ROOT>/train      (registered as hi_proverbs)
  fineweb-edu-zh-chengyu-cpt -> <DATA_ROOT>/train_zh (registered as zh_chengyu)

Usage:
  # Hindi (default):
  python3 prepare_data.py
  # Chinese:
  python3 prepare_data.py --src_dir <DATA_ROOT>/fineweb-edu-zh-chengyu-cpt/data \
                          --out_dir <DATA_ROOT>/train_zh
  python3 prepare_data.py --verify_only --out_dir <DATA_ROOT>/train
"""

import argparse
import glob
import gzip
import json
import random
import sys
import time
from pathlib import Path

# Root holding the downloaded corpora and the prepared train shards (on lustre,
# reachable regardless of which git checkout this file lives in).
DATA_ROOT = Path("/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")

# Defaults target the Hindi corpus; override with --src_dir/--out_dir for others.
DEFAULT_SRC = DATA_ROOT / "hi-proverbs-cpt" / "data"
DEFAULT_OUT = DATA_ROOT / "train"

# Fields carried through to the training shards. LLaMA-Factory only reads
# "text"; the rest are kept for provenance/debugging and ignored by training.
KEEP_FIELDS = ("text", "source", "matched_idioms")


def iter_docs(gz_path: Path):
    """Yield JSON objects from one gzipped JSONL shard, skipping bad lines."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Residual OCR/encoding noise: skip the odd malformed line
                # rather than abort the whole shard.
                continue


def slim(doc: dict) -> dict | None:
    """Project a source doc onto KEEP_FIELDS; drop docs without usable text."""
    text = doc.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {k: doc[k] for k in KEEP_FIELDS if k in doc}


def build(src_dir: Path, out_dir: Path, docs_per_shard: int, seed: int):
    src_shards = sorted(Path(p) for p in glob.glob(str(src_dir / "tagged_*.json.gz")))
    if not src_shards:
        print(f"ERROR: no tagged_*.json.gz found under {src_dir}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)
    rng.shuffle(src_shards)  # inter-shard shuffle; intra-shard buffer below
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previous build so stale shards never leak into a run.
    for old in out_dir.glob("train_*.jsonl"):
        old.unlink()

    print(f"Reading {len(src_shards)} source shards from {src_dir}")
    print(f"Writing plain JSONL to {out_dir} ({docs_per_shard:,} docs/shard)\n")

    t0 = time.time()
    buf: list[dict] = []
    out_idx = 0
    total_in = total_out = total_chars = 0

    def flush():
        nonlocal out_idx, buf
        if not buf:
            return
        rng.shuffle(buf)
        out_path = out_dir / f"train_{out_idx:05d}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for d in buf:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        out_idx += 1
        buf = []

    for i, shard in enumerate(src_shards):
        for doc in iter_docs(shard):
            total_in += 1
            slimmed = slim(doc)
            if slimmed is None:
                continue
            total_out += 1
            total_chars += len(slimmed["text"])
            buf.append(slimmed)
            if len(buf) >= docs_per_shard:
                flush()
        if (i + 1) % 50 == 0 or i == len(src_shards) - 1:
            print(f"  [{i+1}/{len(src_shards)}] kept {total_out:,}/{total_in:,} docs "
                  f"({time.time() - t0:.0f}s)")
    flush()

    # Coarse token estimate only (chars/2.5); the real count depends on the
    # tokenizer and language. Not used for anything but a sanity print.
    est_tokens = total_chars / 2.5
    print(f"\nDone: {out_idx} shards, {total_out:,} docs "
          f"({total_in - total_out:,} dropped), ~{est_tokens/1e9:.2f}B tokens (est), "
          f"{time.time() - t0:.0f}s")
    print(f"Output: {out_dir}/train_*.jsonl")


def verify(out_dir: Path, n_samples: int = 3):
    shards = sorted(out_dir.glob("train_*.jsonl"))
    if not shards:
        print(f"No train_*.jsonl in {out_dir}. Run without --verify_only first.")
        return
    n_docs = n_chars = 0
    samples: list[dict] = []
    for shard in shards:
        with open(shard, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                n_docs += 1
                n_chars += len(doc["text"])
                if len(samples) < n_samples:
                    samples.append(doc)
    print(f"{len(shards)} shards, {n_docs:,} docs, ~{n_chars/2.5/1e9:.2f}B tokens (est)\n")
    for doc in samples:
        src = doc.get("source", "?")
        idioms = doc.get("matched_idioms", [])
        print(f"  [{src}] idioms={idioms[:3]}")
        print(f"    {doc['text'][:200]}...\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--docs_per_shard", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        verify(args.out_dir)
        return

    build(args.src_dir, args.out_dir, args.docs_per_shard, args.seed)
    verify(args.out_dir)
    print("\nDataset ready. Registered in configs/dataset_info.json.")


if __name__ == "__main__":
    main()
