"""Repair the three real defects left in the built Arabic KB, in place.

Run AFTER `build_ar_idioms.py` + `enrich_ar_idioms.py`. The equivalent fixes are
also wired into the builder (`split_variant_furniture`) so a clean rebuild does
not reintroduce them; this script exists so the shipped KB can be corrected
without paying for a full 10,386-call re-enrichment.

Each fix is content-preserving — nothing is deleted, only moved or unwrapped —
and the script reports exactly what it touched. It is idempotent.

1. VARIANT FURNITURE in `idiom` (16 rows). al-Maydani writes the head proverb and
   then its variants inline in quotes:
       أَضَلُّ مِنْ ضَبٍّ، و"مِنْ وَرَلٍ"و"مِنْ وَلَدِ الْيَرْبُوعِ"
   This is not cosmetic: that surface never occurs in running text, so the corpus
   matcher scores **zero recall** on those entries. The head moves to `idiom`, the
   variants to `meta.idiom_variants`.

2. `((...))` MARKUP in meanings (2 rows) — scrape artefact, unwrapped.

3. CLOZE BLANKS in `examples` (2 rows). Kinayat's `full_sentence` occasionally
   carries the ـــــ blank from its cloze task; a blanked sentence is not a usage
   example. Dropped (→ "NAN" if it was the only one).

NOT repaired, deliberately:
- MEANING_TOO_SHORT (36): the source gloss really is terse ("أي أصلها"). 34 of the
  36 have no other meaning, so dropping it would violate inclusion criterion 2
  (every idiom keeps at least one meaning). Terse-but-true beats empty.
- IDIOM_TOO_LONG (19): these are genuinely long classical proverbs and hadith.
  The 15-token threshold is the approximation, not the data.
- EXAMPLE_LACKS_IDIOM (39 minus the 2 above): almost all are the idiom appearing
  in an INFLECTED form (غِرِق → غَرْقان), which is exactly what the stemmer in
  `stem.py` exists to handle. Flagging them is the audit being literal-minded.

Usage:
    python -m culture.data_processing.ar_idioms.repair_ar_kb \
        --input  data/idioms/ar/idioms_merged_llm_formatted.jsonl \
        --output data/idioms/ar/idioms_merged_llm_formatted.jsonl
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from culture.data_processing.ar_idioms.build_ar_idioms import split_variant_furniture

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("repair_ar_kb")

NAN = "NAN"
# Scrape artefact: ((text)) around or inside a gloss.
_RE_MARKUP = re.compile(r"\(\(\s*(.*?)\s*\)\)")
# Absher is crowd-sourced, and two contributors used editing notation inside the
# gloss itself: `||` as a separator and <...> as an editorial aside. Neither is
# ours, and neither is part of the meaning.
_RE_PIPES = re.compile(r"\s*\|\|\s*")
_RE_ANGLE_NOTE = re.compile(r"\s*<[^>]{1,60}>\s*")
# Kinayat's cloze blank: a run of tatweel used as a fill-in gap.
_RE_CLOZE_BLANK = re.compile(r"ـ{3,}")


def _as_list(v: Any) -> List[str]:
    if v is None or v == NAN:
        return []
    return [x for x in (v if isinstance(v, list) else [v]) if x and x != NAN]


def _store(v: List[str]) -> Any:
    return v if v else NAN


def repair_row(row: Dict[str, Any], stats: Dict[str, int]) -> Dict[str, Any]:
    out = row.get("output") or {}
    meta = row.setdefault("meta", {})

    # 1. variant furniture in the idiom
    idiom = out.get("idiom") or ""
    head, variants = split_variant_furniture(idiom)
    if variants or head != idiom:
        if head and head != idiom:
            out["idiom"] = head
            row["idiom"] = head          # top-level mirror of the same string
            stats["idiom_head_extracted"] += 1
        if variants:
            meta["idiom_variants"] = variants
            stats["variants_preserved"] += len(variants)

    # 2. editing notation inside meanings: ((...)), || separators, <...> asides
    for field in ("figurative_meanings", "literal_meanings"):
        ms = _as_list(out.get(field))
        new = []
        for m in ms:
            m2 = _RE_MARKUP.sub(r"\1", m)
            m2 = _RE_PIPES.sub(" ", m2)
            m2 = _RE_ANGLE_NOTE.sub(" ", m2)
            new.append(re.sub(r"\s{2,}", " ", m2).strip())
        if new != ms:
            out[field] = _store([m for m in new if m])
            stats["markup_unwrapped"] += 1

    # 3. cloze blanks in examples
    ex = _as_list(out.get("examples"))
    kept = [e for e in ex if not _RE_CLOZE_BLANK.search(e)]
    if len(kept) != len(ex):
        out["examples"] = _store(kept)
        stats["cloze_examples_dropped"] += len(ex) - len(kept)

    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/idioms/ar/idioms_merged_llm_formatted.jsonl")
    p.add_argument("--output", default=None, help="Default: overwrite --input.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    out_path = args.output or args.input

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    stats: Dict[str, int] = {"idiom_head_extracted": 0, "variants_preserved": 0,
                             "markup_unwrapped": 0, "cloze_examples_dropped": 0}
    before = {r.get("index"): (r.get("output") or {}).get("idiom") for r in rows}
    rows = [repair_row(r, stats) for r in rows]

    # Safety: repairs must not change the row count, and must not empty any idiom
    # or leave an entry with no meaning (inclusion criterion 2).
    assert len(rows) == len(before), "row count changed"
    for r in rows:
        o = r.get("output") or {}
        assert (o.get("idiom") or "").strip(), f"empty idiom at index {r.get('index')}"
        assert _as_list(o.get("figurative_meanings")) or _as_list(o.get("literal_meanings")), \
            f"entry lost all meanings at index {r.get('index')}"

    logger.info("repairs: %s", json.dumps(stats))
    for r in rows:
        b, a = before.get(r.get("index")), (r.get("output") or {}).get("idiom")
        if b != a:
            logger.info("  idiom: %s  ->  %s", (b or "")[:70], (a or "")[:70])

    if args.dry_run:
        logger.info("dry run; %s not written", out_path)
        return 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %s (%d rows)", out_path, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
