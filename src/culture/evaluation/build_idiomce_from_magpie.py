#!/usr/bin/env python3
"""Build the IdiomCE (En->Hi idiomatic translation) eval set from the MAGPIE corpus.

IdiomCE's own eval set was never released. This is the "alternative source (no
generation)" path from docs/plans/eval_benchmarks_download.md §4: instead of
generating sentences with an LLM (see build_idiomce_eval.py), we use *real*
idiom-bearing English sentences from MAGPIE (Haagsma et al., LREC 2020), a
sense-annotated corpus of potentially-idiomatic expressions from the BNC. The
evaluated task is identical: an English sentence containing an idiom -> the model
under test translates it to Hindi -> an OpenAI judge rates idiomatic adequacy
(reference-less, as in the IdiomCE paper, arXiv 2505.21937).

This builder is purely local -- no LLM, no network.

Procedure:
  1. Load the MAGPIE filtered JSONL (confidence >=0.75, binary sense labels).
  2. Keep only idiomatic-usage instances (label 'i') above --min_confidence.
  3. Extract the idiom-bearing sentence: MAGPIE stores a 5-sentence window in
     `context`; the target is context[2], and `offsets` index into it. Validate
     by reconstructing the idiom tokens from those offsets.
  4. Lightly detokenize the BNC-tokenized sentence (fix spaced punctuation).
  5. Drop too-short sentences; dedupe by idiom type (keep highest confidence) and
     by identical sentence text.
  6. Deterministically sample --num_samples idioms.
  7. Write data/eval/hi/idiomce_hi.jsonl in the eval loader's schema
     (reference-less: no `reference` field).

Example:
  python -m culture.evaluation.build_idiomce_from_magpie \\
      --magpie_path data/eval/hi/_magpie/MAGPIE_filtered_split_random.jsonl \\
      --output_path data/eval/hi/idiomce_hi.jsonl --num_samples 400
"""

import argparse
import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.eval.build_idiomce_magpie")

# Stopwords ignored when checking that a sentence carries the idiom's content words.
_STOP = {"the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "for",
         "your", "you", "his", "her", "their", "its", "one", "ones", "be",
         "is", "are", "was", "were", "with", "by", "up", "out", "off", "so",
         "someone", "something", "somebody", "oneself"}


# --------------------------------------------------------------------------- #
# Sentence extraction + cleanup
# --------------------------------------------------------------------------- #
def _content_words(idiom: str) -> List[str]:
    return [w for w in re.split(r"[^a-z]+", idiom.lower())
            if w and w not in _STOP and len(w) > 2]


def _offset_tokens_ok(sent: str, offsets: List[List[int]], idiom: str) -> bool:
    """True if the chars at `offsets` in `sent` reconstruct the idiom's tokens."""
    if not offsets:
        return False
    try:
        pieces = [sent[a:b] for a, b in offsets]
    except (TypeError, ValueError):
        return False
    if not all(p and p.strip() for p in pieces):
        return False
    picked = {p.lower() for p in pieces}
    content = _content_words(idiom)
    if not content:
        return bool(picked)
    # A picked token counts if it prefix-matches a content word (walk/walked etc.).
    hits = sum(1 for w in content
               if any(p.startswith(w[: max(3, len(w) - 2)]) for p in picked))
    return hits / len(content) >= 0.6


def extract_sentence(row: Dict[str, Any]) -> Optional[str]:
    """Return the idiom-bearing sentence from a MAGPIE row, or None if unclear.

    MAGPIE's `context` is a 5-sentence window whose middle element (index 2) holds
    the PIE; `offsets` index into it. We prefer context[2] but fall back to any
    context sentence whose offsets reconstruct the idiom tokens.
    """
    ctx = row.get("context") or []
    offsets = row.get("offsets") or []
    idiom = str(row.get("idiom", ""))
    if not isinstance(ctx, list) or not ctx:
        return None
    if len(ctx) >= 3 and _offset_tokens_ok(ctx[2], offsets, idiom):
        return ctx[2]
    for s in ctx:                                   # fallback scan
        if _offset_tokens_ok(s, offsets, idiom):
            return s
    return None


def detokenize(sent: str) -> str:
    """Light cleanup of BNC-style spaced punctuation. Spacing only -- no rewrites."""
    s = sent
    s = re.sub(r"\s+([,.;:!?)\]])", r"\1", s)       # space before closing punct
    s = re.sub(r"([(\[])\s+", r"\1", s)             # space after opening bracket
    s = re.sub(r"\s+([''])\s*(s|re|ve|ll|d|m|t)\b", r"\1\2", s, flags=re.I)  # 's, n't...
    s = re.sub(r"\bn\s+'\s*t\b", "n't", s, flags=re.I)
    s = re.sub(r"\s+n't\b", "n't", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--magpie_path",
                   default="data/eval/hi/_magpie/MAGPIE_filtered_split_random.jsonl")
    p.add_argument("--output_path", default="data/eval/hi/idiomce_hi.jsonl")
    p.add_argument("--num_samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_confidence", type=float, default=0.9,
                   help="Keep MAGPIE instances with at least this annotation confidence.")
    p.add_argument("--min_words", type=int, default=5, help="Minimum sentence length (words).")
    p.add_argument("--label", default="i", help="MAGPIE sense label to keep ('i'=idiomatic).")
    p.add_argument("--one_per_idiom", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep at most one sentence per idiom type (highest confidence).")
    return p


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    # 1) Load + 2) filter by label/confidence + 3) extract sentence.
    kept: List[Dict[str, Any]] = []
    n_total = n_bad_label = n_low_conf = n_no_target = n_short = 0
    with open(args.magpie_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_total += 1
            row = json.loads(line)
            if str(row.get("label", "")) != args.label:
                n_bad_label += 1
                continue
            conf = float(row.get("confidence", 0.0))
            if conf < args.min_confidence:
                n_low_conf += 1
                continue
            sent = extract_sentence(row)
            if not sent:
                n_no_target += 1
                continue
            sent = detokenize(sent)
            if len(sent.split()) < args.min_words:
                n_short += 1
                continue
            kept.append({
                "idiom": str(row.get("idiom", "")).strip(),
                "source": sent,
                "confidence": conf,
                "magpie_id": row.get("id"),
                "split": row.get("split"),
            })

    logger.info("MAGPIE: %d rows -> %d candidate sentences (dropped: label=%d, "
                "low_conf=%d, no_target=%d, short=%d)",
                n_total, len(kept), n_bad_label, n_low_conf, n_no_target, n_short)

    # 5) Dedupe by idiom type (keep highest confidence) and by identical source.
    n_before = len(kept)
    if args.one_per_idiom:
        best: Dict[str, Dict[str, Any]] = {}
        for r in kept:
            key = r["idiom"].lower()
            if key not in best or r["confidence"] > best[key]["confidence"]:
                best[key] = r
        kept = list(best.values())
    seen_src = set()
    deduped: List[Dict[str, Any]] = []
    for r in kept:
        s = r["source"].lower()
        if s in seen_src:
            continue
        seen_src.add(s)
        deduped.append(r)
    kept = deduped
    n_deduped = n_before - len(kept)

    if not kept:
        raise SystemExit(f"No usable idiom sentences in {args.magpie_path}")

    # 6) Deterministic sample. Sort first so the RNG draw is reproducible regardless
    #    of dict iteration order.
    kept.sort(key=lambda r: (r["idiom"].lower(), str(r["magpie_id"])))
    n_draw = min(len(kept), args.num_samples)
    sample = rng.sample(kept, n_draw)

    # 7) Write output (reference-less: no `reference` field) + report.
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(sample):
            f.write(json.dumps({
                "id": i,
                "source": r["source"],
                "idiom_en": r["idiom"],
                "magpie_id": r["magpie_id"],
                "confidence": r["confidence"],
                "split": r["split"],
                "idiom_detected": True,
            }, ensure_ascii=False) + "\n")

    report = {
        "output_path": args.output_path,
        "requested": args.num_samples,
        "written": len(sample),
        "candidates": n_before,
        "after_dedupe": len(kept),
        "dropped_wrong_label": n_bad_label,
        "dropped_low_conf": n_low_conf,
        "dropped_no_target": n_no_target,
        "dropped_short": n_short,
        "deduped": n_deduped,
        "one_per_idiom": args.one_per_idiom,
        "min_confidence": args.min_confidence,
        "reference_less": True,
    }
    logger.info("IdiomCE (MAGPIE) build report: %s", json.dumps(report, ensure_ascii=False))
    if len(sample) < args.num_samples:
        logger.warning("Wrote %d < requested %d — lower --min_confidence/--min_words "
                       "or disable --one_per_idiom.", len(sample), args.num_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
