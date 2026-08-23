#!/usr/bin/env python3
"""Repair structural defects in a merged idiom KB.

Found by ``ar_idioms/audit_idioms.py`` on the shipped Chinese KB:

===========================  =====  ======================================
defect                        count  cause
===========================  =====  ======================================
nested lists in a field       5,031  the LLM emitted ``[["a","b"]]`` and
 (16.1% of zh; 0% of en)             ``parse_llm_output`` stored the object
                                     verbatim with no schema validation
``output: null``              3 zh   idiom missing from the LLM response
                              9 en
U+FFFD mojibake in a gloss    6 zh   upstream encoding damage
===========================  =====  ======================================

What this script does:

* **flattens** nested lists (recursively), de-duplicates while preserving order,
  and drops empty strings — fully deterministic and safe.
* **reports** ``output: null`` rows and mojibake rows. Those are *not* silently
  "fixed": a null row has no content to recover, and the character destroyed by
  mojibake cannot be reconstructed without guessing. Use ``--drop-null-output``
  to remove null rows explicitly.

A ``.bak`` copy is written unless ``--output`` points elsewhere.

Usage::

    python repair_idiom_kb.py --input data/idioms/zh/idioms_merged_llm_formatted.jsonl
    python repair_idiom_kb.py --input ... --drop-null-output
    python repair_idiom_kb.py --self-test
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.repair_idiom_kb")

FIELDS = ("entities", "literal_meanings", "figurative_meanings", "examples")
NAN = "NAN"


def flatten(value: Any) -> List[str]:
    """Recursively flatten a possibly-nested container into a list of strings.

    Order-preserving de-duplication; empty/``NAN`` items are dropped. A bare
    string becomes a single-element list. ``dict`` values are flattened over
    their values so a stray object cannot silently become ``"{'a': 1}"``.
    """
    out: List[str] = []

    def walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, (list, tuple, set)):
            for item in v:
                walk(item)
        elif isinstance(v, dict):
            for item in v.values():
                walk(item)
        else:
            s = str(v).strip()
            if s and s != NAN:
                out.append(s)

    walk(value)
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def repair_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (repaired_row, list_of_actions)."""
    actions: List[str] = []
    out = row.get("output")
    if not isinstance(out, dict):
        return row, ["NULL_OUTPUT"] if out is None else ["BAD_OUTPUT_TYPE"]

    for f in FIELDS:
        if f not in out:
            continue
        v = out[f]
        if v == NAN or v is None:
            continue
        was_nested = isinstance(v, (list, tuple)) and any(
            isinstance(x, (list, tuple, dict, set)) for x in v
        )
        flat = flatten(v)
        if was_nested:
            actions.append(f"FLATTENED:{f}")
        elif isinstance(v, list) and flat != [str(x).strip() for x in v if str(x).strip()]:
            actions.append(f"CLEANED:{f}")
        out[f] = flat
    if any("�" in s for f in FIELDS for s in flatten(out.get(f))):
        actions.append("MOJIBAKE")
    return row, actions


def repair(rows: List[Dict[str, Any]], drop_null: bool = False
           ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}
    for row in rows:
        fixed, actions = repair_row(row)
        for a in actions:
            key = a.split(":")[0] if ":" in a else a
            stats[key] = stats.get(key, 0) + 1
            if ":" in a:
                stats[a] = stats.get(a, 0) + 1
        if drop_null and "NULL_OUTPUT" in actions:
            stats["DROPPED_NULL_ROWS"] = stats.get("DROPPED_NULL_ROWS", 0) + 1
            continue
        kept.append(fixed)
    return kept, stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input")
    p.add_argument("--output", default=None, help="Default: repair in place (+ .bak).")
    p.add_argument("--drop-null-output", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    if not args.input:
        raise SystemExit("--input is required (or --self-test)")

    src = Path(args.input)
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    logger.info("Read %d rows from %s", len(rows), src)

    fixed, stats = repair(rows, drop_null=args.drop_null_output)

    print("\nrepair actions:")
    for k, v in sorted(stats.items()):
        print(f"  {k:<34} {v}")
    print(f"\nrows in={len(rows)}  out={len(fixed)}")

    if args.dry_run:
        logger.info("--dry-run: nothing written")
        return 0

    dst = Path(args.output) if args.output else src
    if dst == src:
        bak = src.with_suffix(src.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(src, bak)
            logger.info("Backed up original to %s", bak)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for r in fixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(fixed), dst)
    return 0


def self_test() -> int:
    # flatten
    assert flatten([["a", "b"], "c"]) == ["a", "b", "c"]
    assert flatten("x") == ["x"]
    assert flatten([[["deep"]]]) == ["deep"]
    assert flatten(["a", "a", "b"]) == ["a", "b"]          # dedup, order kept
    assert flatten(["", "  ", NAN, None, "ok"]) == ["ok"]  # empties dropped
    assert flatten(None) == [] and flatten([]) == []
    assert flatten({"k": ["v1", "v2"]}) == ["v1", "v2"]    # stray dict

    # the real zh shape
    row = {"idiom": "一丝不挂", "index": 9,
           "output": {"idiom": "一丝不挂", "entities": [],
                      "literal_meanings": [["指人裸体。", "后指人赤身裸体。"]],
                      "figurative_meanings": ["形容一点儿东西也不带"]}}
    fixed, actions = repair_row(row)
    assert "FLATTENED:literal_meanings" in actions
    assert fixed["output"]["literal_meanings"] == ["指人裸体。", "后指人赤身裸体。"]
    assert fixed["output"]["figurative_meanings"] == ["形容一点儿东西也不带"]

    # null output is reported, never invented
    _, actions = repair_row({"idiom": "官卑职小", "index": 9385, "output": None})
    assert actions == ["NULL_OUTPUT"]

    # mojibake is reported, not guessed at
    _, actions = repair_row({"idiom": "x", "output": {"figurative_meanings": ["同“一�一笑”。"]}})
    assert "MOJIBAKE" in actions

    # NAN is preserved (it is our explicit "no value" marker, not a defect)
    fixed, _ = repair_row({"idiom": "x", "output": {"examples": NAN,
                                                    "literal_meanings": ["ok"]}})
    assert fixed["output"]["examples"] == NAN

    # drop-null behaviour
    rows = [{"idiom": "a", "output": {"literal_meanings": [["x"]]}},
            {"idiom": "b", "output": None}]
    kept, stats = repair(rows, drop_null=True)
    assert len(kept) == 1 and stats["DROPPED_NULL_ROWS"] == 1
    kept, _ = repair(rows, drop_null=False)
    assert len(kept) == 2

    print("all repair_idiom_kb.py self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
