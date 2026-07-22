#!/usr/bin/env python3
"""Dimension 1: language-modeling evaluation (perplexity + bits-per-byte).

Computes token-level perplexity (PPL) and tokenizer-agnostic bits-per-byte (BPB)
for one checkpoint over a text corpus, using the sliding-window rolling-NLL in
:meth:`culture.evaluation.scorer.HFModel.rolling_nll`.

Run once per checkpoint (base and CPT) per corpus, then eyeball the deltas:

- **Adaptation** -- a held-out slice of the Hindi CPT corpus (`hi-proverbs-cpt`).
  The CPT model should have *lower* PPL/BPB than the base model here.
- **Retention** -- an English corpus (e.g. WikiText-103). PPL/BPB should be
  roughly unchanged (no catastrophic forgetting of English LM ability).

Lower is better for both metrics.

    # Adaptation (Hindi held-out JSONL with a "text" field):
    python -m culture.evaluation.perplexity \\
        --model_path /path/to/qwen3p5-9b-hi-cpt --run_name cpt \\
        --data_path data/eval/hi/hi_proverbs_heldout.jsonl \\
        --output_dir results/hi/cpt/ppl_hi

    # Retention (WikiText-103 via HuggingFace):
    python -m culture.evaluation.perplexity \\
        --model_path /path/to/qwen3p5-9b-hi-cpt --run_name cpt \\
        --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test \\
        --output_dir results/hi/cpt/ppl_wikitext

CONTAMINATION NOTE: for the adaptation number to be meaningful, the held-out
slice must have been excluded from the CPT training data. Reserve it *before*
training (or use an independent Hindi corpus the model never saw); a slice of the
already-trained corpus is contaminated and will understate the true PPL.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from culture.evaluation.scorer import HFModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.eval.ppl")


def _load_texts(args) -> List[str]:
    """Load documents from a local JSONL/JSON/txt or a HuggingFace dataset."""
    if args.data_path:
        ext = os.path.splitext(args.data_path)[1].lower()
        if ext in (".jsonl", ".json"):
            with open(args.data_path, encoding="utf-8") as f:
                rows = ([json.loads(l) for l in f if l.strip()] if ext == ".jsonl"
                        else json.load(f))
            texts = [r[args.text_field] if isinstance(r, dict) else str(r) for r in rows]
        else:  # plain text: one document per line
            with open(args.data_path, encoding="utf-8") as f:
                texts = [line.rstrip("\n") for line in f]
    elif args.hf_dataset:
        from datasets import load_dataset
        token = os.environ.get("HF_TOKEN")
        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, token=token)
        texts = [r[args.text_field] for r in ds]
    else:
        raise SystemExit("Provide --data_path or --hf_dataset.")

    texts = [t for t in texts if t and t.strip()]
    if args.limit:
        texts = texts[:args.limit]
    logger.info("Loaded %d non-empty documents", len(texts))
    return texts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default=None)

    src = p.add_argument_group("corpus")
    src.add_argument("--data_path", default=None, help="Local jsonl/json/txt corpus.")
    src.add_argument("--hf_dataset", default=None, help="HF dataset name (e.g. wikitext).")
    src.add_argument("--hf_config", default=None, help="HF config (e.g. wikitext-103-raw-v1).")
    src.add_argument("--hf_split", default="test")
    src.add_argument("--text_field", default="text")
    src.add_argument("--limit", type=int, default=None)

    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--stride", type=int, default=None, help="Sliding-window stride (default max_length//2).")
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or os.path.basename(os.path.normpath(args.model_path))

    texts = _load_texts(args)
    model = HFModel(args.model_path, dtype=args.dtype, max_length=args.max_length)

    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    for text in tqdm(texts, desc="perplexity"):
        nll, ntok, nbytes = model.rolling_nll(text, stride=args.stride)
        total_nll += nll
        total_tokens += ntok
        total_bytes += nbytes

    if total_tokens == 0:
        raise SystemExit("No tokens scored -- empty corpus?")

    ppl = math.exp(total_nll / total_tokens)
    bpb = total_nll / math.log(2) / total_bytes  # nats -> bits, per source byte
    result = {
        "run_name": run_name,
        "model_path": args.model_path,
        "corpus": args.data_path or f"{args.hf_dataset}/{args.hf_config}:{args.hf_split}",
        "num_docs": len(texts),
        "num_tokens": total_tokens,
        "num_bytes": total_bytes,
        "ppl": round(ppl, 4),
        "bits_per_byte": round(bpb, 6),
        "avg_nll_nats": round(total_nll / total_tokens, 6),
    }

    out = os.path.join(args.output_dir, "perplexity.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
