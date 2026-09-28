#!/usr/bin/env python3
"""Build the per-language document pool for the bidirectional study.

One invocation processes one raw source file and writes one pool shard:

    <out_dir>/<shard>.jsonl.gz   {"id", "text", "source", "idioms": [...]}

``idioms`` lists every IdiomAtlas idiom found in the document (the same
matchers as the idiom corpora: normalized Tier-0 Aho-Corasick for Arabic, raw
Aho-Corasick over canonical forms + spelling variants for Hindi, raw
Aho-Corasick over the chengyu inventory for Chinese). The pool is the common
source of the Random arm and the Culture arms, so every arm passes the same
length/quality gates:

* ar: ``quality_ar.reject_reason`` (the gate used for the Arabic idiom corpus),
  300 <= chars <= 25,000.
* zh, hi: 150 <= chars <= 100,000 (the gate of the random controls).

Usage:
  python -m culture.bidirectional.build_pool --lang ar --input <file.parquet> \
      --out_dir $B/pool/ar --shard ar_000
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import ahocorasick

REPO = Path(__file__).resolve().parents[3]
KB = {
    "zh": REPO / "culture/data/idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl",
    "hi": REPO / "culture/data/idioms/hi/idioms_merged_llm_formatted_figurative_only.jsonl",
    "ar": REPO / "culture/data/idioms/ar/idioms_merged_llm_formatted.jsonl",
}
HI_VARIANTS = REPO / "culture/data/hi_idioms/idiom_variants_hi.jsonl"


class RawMatcher:
    def __init__(self, surfaces_to_canonical, min_chars):
        self.A = ahocorasick.Automaton()
        for s, c in surfaces_to_canonical.items():
            if len(s) >= min_chars:
                self.A.add_word(s, c)
        self.A.make_automaton()

    def match(self, text):
        return {c for _, c in self.A.iter(text)}


def make_matcher(lang):
    rows = [json.loads(l) for l in open(KB[lang], encoding="utf-8") if l.strip()]
    if lang == "ar":
        from culture.training.mC4.filter_and_tag_ar import ArabicIdiomMatcher
        return ArabicIdiomMatcher(rows, use_stem=False)
    surf = {r["output"]["idiom"].strip(): r["output"]["idiom"].strip() for r in rows}
    if lang == "hi":
        for l in open(HI_VARIANTS, encoding="utf-8"):
            v = json.loads(l)
            for s in v.get("variants", []):
                surf.setdefault(s.strip(), v["canonical"].strip())
        return RawMatcher(surf, min_chars=6)
    return RawMatcher(surf, min_chars=4)


def iter_raw(path, rg_stride=1, rg_offset=0):
    """Yield document texts from parquet / json.gz / jsonl.gz raw files.

    For parquet, only row groups with ``index % rg_stride == rg_offset`` are read,
    which subsamples very large files in contiguous chunks."""
    p = str(path)
    if p.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(p)
        for rg in range(rg_offset, pf.num_row_groups, rg_stride):
            for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                yield t
    else:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["ar", "zh", "hi"])
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--rg_stride", type=int, default=1)
    ap.add_argument("--rg_offset", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out_dir) / f"{a.shard}.jsonl.gz"
    if out.exists():
        print("exists", out)
        return
    os.makedirs(a.out_dir, exist_ok=True)
    m = make_matcher(a.lang)
    if a.lang == "ar":
        from culture.training.mC4.quality_ar import reject_reason
        lo, hi_ = 300, 25_000
    else:
        reject_reason = None
        lo, hi_ = 150, 100_000
    st = {"scanned": 0, "kept": 0, "with_idiom": 0}
    tmp = str(out) + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fo:
        for path in a.input:
            for text in iter_raw(path, a.rg_stride, a.rg_offset):
                st["scanned"] += 1
                if not text or not (lo <= len(text) <= hi_):
                    continue
                if reject_reason is not None and reject_reason(text):
                    continue
                ids = sorted(m.match(text))
                st["kept"] += 1
                st["with_idiom"] += bool(ids)
                fo.write(json.dumps({"id": f"{a.shard}-{st['kept']}", "text": text,
                                     "source": a.source, "idioms": ids},
                                    ensure_ascii=False) + "\n")
                if a.max_docs and st["kept"] >= a.max_docs:
                    break
            if a.max_docs and st["kept"] >= a.max_docs:
                break
    os.replace(tmp, out)
    json.dump(st, open(Path(a.out_dir) / f"{a.shard}.stats.json", "w"))
    print(a.shard, st, file=sys.stderr)


if __name__ == "__main__":
    main()
