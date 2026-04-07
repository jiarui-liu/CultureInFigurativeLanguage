#!/usr/bin/env python3
"""
Show document-to-idiom mappings from an infini-gram index.

Usage:
    # Show mappings for specific idioms
    python show_index_mappings.py --index-dir /path/to/index_en --idioms "get better" "in the long run"

    # Load idioms from a file and show all mappings
    python show_index_mappings.py --index-dir /path/to/index_en --idiom-file /path/to/idioms.jsonl

    # Show top N idioms by count
    python show_index_mappings.py --index-dir /path/to/index_en --idiom-file /path/to/idioms.jsonl --top 20
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Set


def load_idioms_from_file(idiom_file: Path) -> Set[str]:
    """Load idioms from a JSONL file."""
    idioms = set()
    with open(idiom_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Handle different formats
                if "output" in data:
                    idiom = data["output"].get("idiom", "")
                else:
                    idiom = data.get("idiom", "")
                if idiom:
                    idioms.add(idiom)
    return idioms


def main():
    parser = argparse.ArgumentParser(
        description="Show document-to-idiom mappings from an infini-gram index"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Path to the infini-gram index directory"
    )
    parser.add_argument(
        "--idioms",
        nargs="+",
        default=None,
        help="Specific idioms to search for"
    )
    parser.add_argument(
        "--idiom-file",
        type=Path,
        default=None,
        help="Path to JSONL file containing idioms"
    )
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace tokenizer to use"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N idioms by count"
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum count to display an idiom"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=3,
        help="Maximum documents to show per idiom"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=60,
        help="Characters of context to show around each idiom match"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output JSON, no progress messages"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.index_dir.exists():
        print(f"Error: Index directory not found: {args.index_dir}")
        return 1

    if args.idioms is None and args.idiom_file is None:
        print("Error: Must specify either --idioms or --idiom-file")
        return 1

    # Import here to avoid slow startup if just checking help
    from infini_gram.engine import InfiniGramEngine
    from transformers import AutoTokenizer

    # Load tokenizer and engine
    if not args.quiet:
        print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        add_bos_token=False,
        add_eos_token=False
    )

    if not args.quiet:
        print(f"Loading index: {args.index_dir}")
    engine = InfiniGramEngine(
        index_dir=str(args.index_dir),
        eos_token_id=tokenizer.eos_token_id
    )

    # Get idioms to search
    if args.idioms:
        idioms = set(args.idioms)
    else:
        if not args.quiet:
            print(f"Loading idioms from: {args.idiom_file}")
        idioms = load_idioms_from_file(args.idiom_file)

    if not args.quiet:
        print(f"Searching for {len(idioms)} idioms...")
        print()

    # Query each idiom
    results = []
    for idiom in idioms:
        input_ids = tokenizer.encode(idiom)
        count_result = engine.count(input_ids=input_ids)
        count = count_result.get("count", 0)

        if count < args.min_count:
            continue

        find_result = engine.find(input_ids=input_ids)

        # Get documents containing this idiom
        docs = []
        for shard_idx, (start, end) in enumerate(find_result.get('segment_by_shard', [])):
            for rank in range(start, min(end, start + args.max_docs)):
                try:
                    doc = engine.get_doc_by_rank(s=shard_idx, rank=rank, max_disp_len=2000)
                    metadata = json.loads(doc['metadata'])
                    text = tokenizer.decode(doc['token_ids'])

                    # Find context around the idiom
                    idiom_lower = idiom.lower()
                    text_lower = text.lower()
                    pos = text_lower.find(idiom_lower)

                    context = ""
                    if pos != -1:
                        start_ctx = max(0, pos - args.context_size)
                        end_ctx = min(len(text), pos + len(idiom) + args.context_size)
                        context = text[start_ctx:end_ctx]

                    docs.append({
                        "source": metadata.get("path", "unknown"),
                        "line": metadata.get("linenum", -1),
                        "context": context,
                        "doc_len": doc.get("doc_len", 0)
                    })
                except Exception as e:
                    if not args.quiet:
                        print(f"  Warning: Error getting doc for '{idiom}': {e}")

        results.append({
            "idiom": idiom,
            "count": count,
            "documents": docs
        })

    # Sort by count descending
    results.sort(key=lambda x: x["count"], reverse=True)

    # Limit to top N if specified
    if args.top:
        results = results[:args.top]

    # Output results
    if not args.quiet:
        print("=" * 70)
        print(f"DOCUMENT-IDIOM MAPPINGS")
        print(f"Index: {args.index_dir}")
        print(f"Found: {len(results)} idioms with count >= {args.min_count}")
        print("=" * 70)

        for r in results:
            print(f"\n{'='*60}")
            print(f"IDIOM: \"{r['idiom']}\"")
            print(f"COUNT: {r['count']}")
            print(f"{'='*60}")

            for doc in r["documents"]:
                print(f"\n  Source: {doc['source']}:{doc['line']}")
                print(f"  Context: ...{doc['context']}...")

    # Save to file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
