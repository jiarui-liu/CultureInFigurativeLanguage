#!/usr/bin/env python3
"""Build the UNFILTERED Hindi continued-pretraining corpus (ablation control).

Companion to the idiom-tagged corpus (jiviteshjn/hi-proverbs-cpt). This samples
the SAME three web sources but with **no idiom filtering and no augmentation** --
plain random documents -- to the SAME Qwen3.5 token budget, so the only difference
vs. the idiom corpus is the idiom curation itself.

One source per invocation (run three in parallel), e.g.:
  python -m culture.training.continued_pretraining.build_unfiltered_hi \
      --path allenai/c4 --name hi --split train --source_label mc4-hi \
      --token_budget 820000000 --out_dir <DATA>/train_hi_unfiltered \
      --tokenizer /lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B

Sources / budgets to match hi-proverbs-cpt (~1.37B tokens total):
  mc4-hi       allenai/c4            name=hi         split=train   0.82B
  fineweb2-hi  HuggingFaceFW/fineweb-2 name=hin_Deva split=train   0.44B
  indiccorp-hi ai4bharat/IndicCorpV2  (no name)      split=hin_Deva 0.11B
"""

import argparse
import gzip
import hashlib
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.cpt.build_unfiltered")


def _text_of(row):
    if "text" in row and isinstance(row["text"], str):
        return row["text"]
    for v in row.values():                       # fallback: first string field
        if isinstance(v, str) and len(v) > 0:
            return v
    return ""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--hf_repo", default=None,
                   help="For raw-text-file mode: dataset repo id (e.g. ai4bharat/IndicCorpV2).")
    p.add_argument("--hf_text_files", default=None,
                   help="Comma-separated repo-relative .txt paths streamed line-by-line "
                        "(e.g. data/hi-1.txt,data/hi-2.txt). One line = one document.")
    p.add_argument("--data_files", default=None,
                   help="Comma-separated data_files globs for load_dataset (e.g. "
                        "'4_5/000*.parquet,4_5/001*.parquet' for a score-tier subset).")
    p.add_argument("--source_label", required=True)
    p.add_argument("--token_budget", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--min_chars", type=int, default=150)
    p.add_argument("--max_chars", type=int, default=100000)
    p.add_argument("--shard_docs", type=int, default=20000)
    p.add_argument("--shuffle_buffer", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=1000, help="Docs per tokenizer batch.")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Build a generic iterator of document text strings from either an HF dataset
    # (streaming) or raw per-line .txt files on the Hub (IndicCorpV2).
    if args.hf_text_files:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))

        def text_iter():
            for rel in args.hf_text_files.split(","):
                p = f"datasets/{args.hf_repo}/{rel.strip()}"
                logger.info("streaming raw text file %s", p)
                with fs.open(p, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        yield line
        texts = text_iter()
    else:
        from datasets import load_dataset
        df = [g.strip() for g in args.data_files.split(",")] if args.data_files else None
        if df:
            ds = load_dataset(args.path, data_files=df, split=args.split, streaming=True,
                              token=os.environ.get("HF_TOKEN"))
        else:
            ds = load_dataset(args.path, args.name, split=args.split, streaming=True,
                              token=os.environ.get("HF_TOKEN"))
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        texts = (_text_of(row) for row in ds)

    n_tokens = n_docs = n_seen = n_short = n_long = n_dup = 0
    seen = set()
    shard_idx, shard_rows = 0, []
    pending = []  # (text) awaiting batched tokenization

    def flush_shard():
        nonlocal shard_idx, shard_rows
        if not shard_rows:
            return
        path = os.path.join(args.out_dir, f"train_{args.source_label}_{shard_idx:05d}.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for r in shard_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("wrote %s (%d docs, cum_tokens=%d)", path, len(shard_rows), n_tokens)
        shard_idx += 1
        shard_rows = []

    def consume_batch():
        nonlocal n_tokens, n_docs, pending, shard_rows
        if not pending:
            return
        enc = tok([t for t in pending], add_special_tokens=False)
        for t, ids in zip(pending, enc["input_ids"]):
            if n_tokens >= args.token_budget:
                break
            n_tokens += len(ids)
            n_docs += 1
            shard_rows.append({"text": t, "source": args.source_label})
            if len(shard_rows) >= args.shard_docs:
                flush_shard()
        pending = []

    for text in texts:
        if n_tokens >= args.token_budget:
            break
        n_seen += 1
        text = (text or "").strip()
        c = len(text)
        if c < args.min_chars:
            n_short += 1; continue
        if c > args.max_chars:
            n_long += 1; continue
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            n_dup += 1; continue
        seen.add(h)
        pending.append(text)
        if len(pending) >= args.batch:
            consume_batch()
    consume_batch()
    flush_shard()

    report = {
        "source_label": args.source_label, "path": args.path, "name": args.name,
        "token_budget": args.token_budget, "tokens_written": n_tokens,
        "docs_written": n_docs, "docs_seen": n_seen,
        "dropped_short": n_short, "dropped_long": n_long, "dropped_dup": n_dup,
        "shards": shard_idx,
    }
    with open(os.path.join(args.out_dir, f"_report_{args.source_label}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
