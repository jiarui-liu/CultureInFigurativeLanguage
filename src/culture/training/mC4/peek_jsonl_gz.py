#!/usr/bin/env python3
"""
Peek at the first N lines of a .json.gz or .jsonl file.

Usage:
    python peek_jsonl_gz.py path/to/file.json.gz
    python peek_jsonl_gz.py path/to/file.json.gz -n 10
    python peek_jsonl_gz.py path/to/file.jsonl --pretty
"""

import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Print first N lines of a .json.gz or .jsonl file"
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to .json.gz or .jsonl file",
    )
    parser.add_argument(
        "-n", "--lines",
        type=int,
        default=5,
        help="Number of lines to print (default: 5)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation",
    )
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        help="Only print a specific field from each JSON object",
    )
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    # Determine if gzipped
    if args.file.suffix == ".gz":
        open_func = lambda f: gzip.open(f, 'rt', encoding='utf-8')
    else:
        open_func = lambda f: open(f, 'r', encoding='utf-8')

    print(f"File: {args.file}")
    print(f"Showing first {args.lines} lines:\n")
    print("=" * 60)

    with open_func(args.file) as f:
        for i, line in enumerate(f):
            if i >= args.lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                if args.field:
                    value = data.get(args.field, "<field not found>")
                    print(f"[{i+1}] {args.field}: {value[:200]}..." if len(str(value)) > 200 else f"[{i+1}] {args.field}: {value}")
                elif args.pretty:
                    print(f"[{i+1}]")
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                else:
                    # Truncate long text for display
                    if "text" in data and len(data["text"]) > 200:
                        data["text"] = data["text"][:200] + "..."
                    print(f"[{i+1}] {json.dumps(data, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print(f"[{i+1}] (invalid JSON) {line[:100]}...")

            print("-" * 60)

    print(f"\n(Showed {min(i+1, args.lines)} lines)")
    return 0


if __name__ == "__main__":
    exit(main())
