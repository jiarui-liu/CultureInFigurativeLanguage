#!/usr/bin/env python3
"""Build the IdiomCE (En->Hi idiomatic translation) eval set from the English idiom KB.

IdiomCE's own eval set was never released, so we construct a functionally-
equivalent one from the project's English idiom knowledge base. The evaluated
task is identical: an English sentence containing an idiom -> the model under test
translates it to Hindi -> an OpenAI judge rates idiomatic adequacy (reference-less,
as in the IdiomCE paper, arXiv 2505.21937).

Procedure (see docs/plans/eval_benchmarks_download.md §4):
  1. Load the English idiom KB (idiom + figurative meaning).
  2. Filter to non-trivial idioms (>=2 words, non-empty figurative meaning); dedupe.
  3. Deterministically sample --num_samples idioms.
  4. Generate ONE natural English sentence per idiom with an LLM (uses the idiom
     figuratively, without explaining it).
  5. Validate the sentence actually contains the idiom (inflection-tolerant) and
     meets a min-length check.
  6. (optional) --add_reference: generate an idiomatic Hindi reference (LLM-made,
     not gold) for reference-guided judging.
  7. Write data/eval/hi/idiomce_hi.jsonl in the eval loader's schema.

Example:
  python -m culture.evaluation.build_idiomce_eval \\
      --idiom_path culture/data/idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl \\
      --output_path data/eval/hi/idiomce_hi.jsonl \\
      --num_samples 500 --model gpt-4o --provider openai
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from culture.models.llm_utils import ChatModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.eval.build_idiomce")

# Minimal stopword set for inflection-tolerant idiom-in-sentence matching.
_STOP = {"the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "for",
         "your", "you", "his", "her", "their", "its", "one", "ones", "be",
         "is", "are", "was", "were", "with", "by", "up", "out", "off"}

GEN_SYSTEM = (
    "You are a fluent native English writer. Given an idiom and its figurative "
    "meaning, write exactly ONE natural, realistic English sentence that USES the "
    "idiom with its figurative meaning in an everyday context. Do NOT define, gloss, "
    "or explain the idiom. Output only the sentence, nothing else."
)
GEN_USER = 'Idiom: "{idiom}"\nFigurative meaning: {figurative}\n\nOne sentence using this idiom naturally:'

REF_SYSTEM = (
    "You are an expert English->Hindi translator. Translate the sentence into "
    "natural, fluent Hindi, rendering any idiom by its FIGURATIVE meaning -- ideally "
    "with a natural Hindi idiom/muhavara -- not word-for-word. Output only the Hindi."
)
REF_USER = "English: {sentence}\nHindi:"


# --------------------------------------------------------------------------- #
# KB loading / filtering
# --------------------------------------------------------------------------- #
def _first(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            return row[k]
    return default


def _as_meaning(val: Any) -> str:
    """Figurative/literal fields may be a string or a list; return a single string."""
    if isinstance(val, list):
        parts = [str(x).strip() for x in val if str(x).strip()]
        return "; ".join(parts)
    return str(val).strip() if val else ""


def load_idioms(path: str) -> List[Dict[str, str]]:
    """Load {idiom, figurative} rows, keeping only non-trivial idioms; dedupe."""
    seen = set()
    out: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idiom = str(_first(row, ["idiom", "phrase", "expression"], "")).strip()
            fig = _as_meaning(_first(row, ["figurative_meanings", "figurative_meaning", "figurative"]))
            if not idiom or not fig:
                continue
            if len(idiom.split()) < 2:            # drop trivial single-word entries
                continue
            key = idiom.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"idiom": idiom, "figurative": fig})
    logger.info("Loaded %d non-trivial, de-duplicated idioms from %s", len(out), path)
    return out


# --------------------------------------------------------------------------- #
# Prompt building / validation
# --------------------------------------------------------------------------- #
def gen_messages(idiom: str, figurative: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": GEN_USER.format(idiom=idiom, figurative=figurative)}]


def ref_messages(sentence: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": REF_SYSTEM},
            {"role": "user", "content": REF_USER.format(sentence=sentence)}]


def clean_sentence(raw: Optional[str]) -> str:
    """Strip quotes / list markers / 'Sentence:' prefixes from an LLM completion."""
    if not raw:
        return ""
    text = raw.strip().splitlines()[0].strip()          # first line only
    text = re.sub(r'^(sentence|output|hindi|answer)\s*[:\-]\s*', "", text, flags=re.I)
    return text.strip().strip('"').strip("'").strip()


def sentence_contains_idiom(idiom: str, sentence: str) -> bool:
    """Lenient, inflection-tolerant check that the sentence uses the idiom."""
    s = sentence.lower()
    if idiom.lower() in s:
        return True
    content = [w for w in re.split(r"[^a-z]+", idiom.lower()) if w and w not in _STOP and len(w) > 2]
    if not content:
        return True
    # count content words present as a prefix match (handles walk/walked, chicken/chickens)
    matched = sum(1 for w in content if re.search(r"\b" + re.escape(w[: max(3, len(w) - 2)]), s))
    return matched / len(content) >= 0.6


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--idiom_path",
                   default="culture/data/idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl")
    p.add_argument("--output_path", default="data/eval/hi/idiomce_hi.jsonl")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--provider", default="openai")
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--add_reference", action="store_true",
                   help="Also generate an (LLM-made, non-gold) Hindi reference per item.")
    p.add_argument("--keep_all", action="store_true",
                   help="Keep items even if the generated sentence fails the idiom check.")
    p.add_argument("--min_words", type=int, default=4, help="Minimum sentence length.")
    p.add_argument("--oversample", type=float, default=1.3,
                   help="Sample this multiple of --num_samples to offset validation drops.")
    return p


def main():
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    idioms = load_idioms(args.idiom_path)
    if not idioms:
        raise SystemExit(f"No usable idioms in {args.idiom_path}")

    # Oversample to absorb validation failures, capped at the KB size.
    n_draw = min(len(idioms), int(args.num_samples * args.oversample))
    sample = rng.sample(idioms, n_draw)
    logger.info("Sampled %d idioms (target %d after validation)", n_draw, args.num_samples)

    model = ChatModel(model=args.model, provider=args.provider)

    # 1) Generate one English sentence per idiom.
    gen_reqs = [(i, gen_messages(it["idiom"], it["figurative"])) for i, it in enumerate(sample)]
    gen_out = model.batch_generate_with_indices_sync(gen_reqs, batch_size=args.batch_size, temperature=0.7)
    sentences: Dict[int, str] = {}
    for idx, resp, err in gen_out:
        if err is not None:
            logger.warning("Generation error on idiom %s: %s", idx, err)
        sentences[idx] = clean_sentence(resp)

    # 2) Validate and assemble.
    records: List[Dict[str, Any]] = []
    n_fail_len = n_fail_idiom = 0
    for i, it in enumerate(sample):
        if len(records) >= args.num_samples:
            break
        sent = sentences.get(i, "")
        if len(sent.split()) < args.min_words:
            n_fail_len += 1
            continue
        ok = sentence_contains_idiom(it["idiom"], sent)
        if not ok and not args.keep_all:
            n_fail_idiom += 1
            continue
        records.append({
            "id": len(records),
            "source": sent,
            "idiom_en": it["idiom"],
            "figurative_meaning": it["figurative"],
            "idiom_detected": ok,
        })

    # 3) Optional LLM-generated Hindi reference.
    if args.add_reference and records:
        ref_reqs = [(r["id"], ref_messages(r["source"])) for r in records]
        ref_out = model.batch_generate_with_indices_sync(ref_reqs, batch_size=args.batch_size, temperature=0.0)
        refs = {idx: clean_sentence(resp) for idx, resp, err in ref_out}
        for r in records:
            r["reference"] = refs.get(r["id"], "")
            r["reference_is_llm_generated"] = True

    # 4) Write output + report.
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "output_path": args.output_path,
        "requested": args.num_samples,
        "written": len(records),
        "sampled": n_draw,
        "dropped_too_short": n_fail_len,
        "dropped_idiom_not_detected": n_fail_idiom,
        "with_reference": bool(args.add_reference),
        "model": args.model,
    }
    logger.info("IdiomCE build report: %s", json.dumps(report, ensure_ascii=False))
    if len(records) < args.num_samples:
        logger.warning("Wrote %d < requested %d — raise --oversample or --num_samples.",
                       len(records), args.num_samples)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
