#!/usr/bin/env python3
"""Entry-by-entry quality audit for a merged idiom KB.

Two passes, both covering **every** entry:

1. **Deterministic pass (this file, no LLM).** ~15 checks run over 100% of rows:
   malformed structure, wrong script, circular definitions, placeholder/truncated
   meanings, leftover quote furniture or markup, mojibake, post-normalization
   duplicates, implausible length, examples that do not contain their own idiom,
   and explanations copy-pasted across many idioms. Cheap and exhaustive — this
   catches the mechanical defects that an LLM reviewer would waste tokens on.

2. **LLM semantic pass (``--emit-shards``).** Everything the deterministic pass
   cannot judge — "is this actually an idiom?", "does the explanation explain
   *this* expression?", "is the usage sentence natural and correct?" — is written
   out as reviewable shards for subagents, one JSON file per shard.

Language-agnostic: ``--lang ar`` uses the Arabic normalizer for dedup/example
checks; ``zh``/``en`` fall back to casefolding, so the tool can be validated
against the existing Chinese/English KBs before the Arabic one exists.

Usage::

    # full deterministic audit
    python audit_idioms.py --input data/idioms/ar/idioms_merged_llm_formatted.jsonl \\
        --lang ar --report data/idioms/ar/audit_report.json

    # write 20 shards for LLM review by subagents
    python audit_idioms.py --input ... --lang ar --emit-shards 20 \\
        --shard-dir data/idioms/ar/audit_shards

    python audit_idioms.py --self-test
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.ar_idioms.audit")

NAN = "NAN"

# --------------------------------------------------------------------------- #
# Thresholds (tune here; every one is reported so effects are visible)
# --------------------------------------------------------------------------- #
# Per-language: Chinese is far denser than Arabic, so a 5-char zh gloss is normal
# while a 5-char Arabic one is truncated. Validated against the real zh KB, where a
# flat threshold of 6 produced ~2.8k false positives on legitimate glosses.
MIN_MEANING_CHARS = {"ar": 10, "en": 8, "zh": 4}
DEFAULT_MIN_MEANING_CHARS = 6
MIN_IDIOM_TOKENS = 2         # a 1-token "idiom" is a word, not an expression
MAX_IDIOM_TOKENS = 15        # longer than this is a sentence/paragraph, not an idiom
SHARED_MEANING_MIN = 5       # same explanation on >=N idioms -> advisory only

# Advisory codes are surfaced but do NOT count a row as defective: genuinely
# synonymous idioms legitimately share a gloss (validated on the zh KB, where
# "形容人多拥挤。" is correctly attached to 10 different chengyu).
ADVISORY_CODES = {"SHARED_MEANING"}

_RE_TOKEN = re.compile(r"\S+")
_RE_FURNITURE = re.compile(r'[«»“”„‟\[\]{}]|^["\']|["\']$')
_RE_MARKUP = re.compile(r"<[^>]{1,40}>|&[a-z]{2,8};|&#\d{2,5};|\|\||\{\{|\}\}")
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_PLACEHOLDER = re.compile(r"^(?:[.\-–—_…\s?؟]+|n/?a|nan|none|null|tbd|xxx+)$", re.I)
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
_RE_CJK = re.compile(r"[㐀-䶿一-鿿]")
# Phrases that betray an LLM wrote the "explanation" (criterion 1 backstop).
# Must be an assistant *disclaimer*, not ordinary prose: a bare "I cannot" is a
# perfectly good gloss ("(I) can't beat that." -> "I cannot do better than that."),
# and matching it produced 16 false positives on the real English KB.
_RE_AI_TELL = re.compile(
    r"as an ai\b"
    r"|\bai (?:language )?model\b"
    r"|\blanguage model\b"
    r"|i(?:'m| am) (?:an ai|unable to (?:assist|provide|help))"
    r"|i cannot (?:assist|provide|help|generate|comply|fulfill)"
    r"|بصفتي نموذج|كنموذج ذكاء|لا أستطيع تقديم|لا يمكنني مساعدت",
    re.I,
)

SCRIPT_RE = {"ar": _RE_ARABIC, "zh": _RE_CJK, "en": _RE_LATIN}


def _norm(text: str, lang: str) -> str:
    """Normalizer used for dedup and example containment."""
    if lang == "ar":
        from culture.data_processing.ar_idioms.normalize import normalize_ar
        return normalize_ar(text)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _norm_idiom(idiom: str, lang: str) -> str:
    """Like :func:`_norm` but also drops quote furniture.

    Furniture is reported separately as its own issue; for *comparison* purposes
    (dedup, circularity, example containment) it must not mask a real match.
    """
    if lang == "ar":
        from culture.data_processing.ar_idioms.normalize import (
            normalize_ar,
            strip_quote_furniture,
        )
        return normalize_ar(strip_quote_furniture(idiom))
    return _norm(_RE_FURNITURE.sub("", idiom), lang)


# --------------------------------------------------------------------------- #
# Row access — tolerant of both the {"output": {...}} shape and a flat shape
# --------------------------------------------------------------------------- #
FIELDS = ("entities", "literal_meanings", "figurative_meanings", "examples")


def get_out(row: Dict[str, Any]) -> Dict[str, Any]:
    out = row.get("output")
    return out if isinstance(out, dict) else row


def field_list(row: Dict[str, Any], name: str) -> List[str]:
    """Return a field as a list of strings; ``"NAN"``/missing -> []."""
    v = get_out(row).get(name)
    if v is None or v == NAN:
        return []
    if isinstance(v, str):
        return [] if v == NAN else [v]
    return [str(x) for x in v if x is not None and str(x) != NAN]


def meanings(row: Dict[str, Any]) -> List[str]:
    return field_list(row, "literal_meanings") + field_list(row, "figurative_meanings")


# --------------------------------------------------------------------------- #
# Per-row checks
# --------------------------------------------------------------------------- #
def check_row(row: Dict[str, Any], lang: str) -> List[Tuple[str, str]]:
    """Return a list of (issue_code, detail) for one entry."""
    issues: List[Tuple[str, str]] = []
    out = get_out(row)
    idiom = str(row.get("idiom") or out.get("idiom") or "")

    if not isinstance(out, dict):
        return [("STRUCTURE", "output is not an object")]
    if not idiom.strip():
        return [("EMPTY_IDIOM", "")]

    # --- the idiom string itself ---
    if _RE_CONTROL.search(idiom):
        issues.append(("CONTROL_CHARS", repr(idiom[:40])))
    if "�" in idiom:
        issues.append(("MOJIBAKE", repr(idiom[:40])))
    if _RE_FURNITURE.search(idiom):
        issues.append(("FURNITURE", idiom[:40]))
    if _RE_MARKUP.search(idiom):
        issues.append(("MARKUP", idiom[:40]))

    script = SCRIPT_RE.get(lang)
    if script and not script.search(idiom):
        issues.append(("WRONG_SCRIPT", idiom[:40]))
    if lang == "ar" and _RE_LATIN.search(idiom):
        issues.append(("LATIN_IN_ARABIC", idiom[:40]))

    ntok = len(_RE_TOKEN.findall(idiom))
    if ntok < MIN_IDIOM_TOKENS and lang != "zh":   # zh chengyu are single tokens
        issues.append(("IDIOM_TOO_SHORT", f"{ntok} token(s): {idiom[:40]}"))
    if ntok > MAX_IDIOM_TOKENS:
        issues.append(("IDIOM_TOO_LONG", f"{ntok} tokens"))

    # --- structural: a field must be a flat list of strings ---
    for fname in FIELDS:
        v = out.get(fname)
        if isinstance(v, (list, tuple)) and any(isinstance(x, (list, tuple, dict))
                                                for x in v):
            issues.append(("NESTED_LIST", f"{fname} contains a nested container"))

    # --- meanings ---
    ms = meanings(row)
    if not ms:
        issues.append(("NO_MEANING", ""))          # violates inclusion criterion 2
    for m in ms:
        ms_ = m.strip()
        if _RE_PLACEHOLDER.match(ms_):
            issues.append(("PLACEHOLDER_MEANING", ms_[:40]))
        elif len(ms_) < MIN_MEANING_CHARS.get(lang, DEFAULT_MIN_MEANING_CHARS):
            issues.append(("MEANING_TOO_SHORT", ms_[:40]))
        if _norm(ms_, lang) == _norm_idiom(idiom, lang):
            issues.append(("CIRCULAR_MEANING", ms_[:40]))
        if _RE_AI_TELL.search(ms_):
            issues.append(("AI_TELL", ms_[:60]))
        if "�" in ms_:
            issues.append(("MOJIBAKE_MEANING", ms_[:40]))
        if _RE_MARKUP.search(ms_):
            issues.append(("MARKUP_MEANING", ms_[:40]))

    # --- examples must actually contain their idiom ---
    nidiom = _norm_idiom(idiom, lang)
    for ex in field_list(row, "examples"):
        if nidiom and nidiom not in _norm(ex, lang):
            issues.append(("EXAMPLE_LACKS_IDIOM", ex[:60]))

    return issues


def audit(rows: List[Dict[str, Any]], lang: str) -> Dict[str, Any]:
    """Run every check over every row; return a structured report."""
    issues_by_row: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    counts: Counter = Counter()
    by_norm: Dict[str, List[int]] = defaultdict(list)
    meaning_users: Dict[str, List[int]] = defaultdict(list)
    idioms: Dict[int, str] = {}

    for i, row in enumerate(rows):
        idiom = str(row.get("idiom") or get_out(row).get("idiom") or "")
        idioms[i] = idiom
        for code, detail in check_row(row, lang):
            issues_by_row[i].append((code, detail))
            counts[code] += 1
        key = _norm_idiom(idiom, lang)
        if key:
            by_norm[key].append(i)
        for m in meanings(row):
            meaning_users[_norm(m, lang)].append(i)

    # --- cross-row checks, attributed back to the offending rows ---
    dups = {k: v for k, v in by_norm.items() if len(v) > 1}
    for key, idxs in dups.items():
        for i in idxs[1:]:                    # first occurrence is the keeper
            issues_by_row[i].append(("DUPLICATE_AFTER_NORMALIZATION",
                                     f"same as index {idxs[0]}: {key[:40]}"))
            counts["DUPLICATE_AFTER_NORMALIZATION"] += 1

    shared = {k: v for k, v in meaning_users.items()
              if len(v) >= SHARED_MEANING_MIN and k}
    for key, idxs in shared.items():
        for i in idxs:
            issues_by_row[i].append(("SHARED_MEANING",
                                     f"explanation reused on {len(idxs)} idioms"))
            counts["SHARED_MEANING"] += 1

    per_row = [
        {"index": i, "idiom": idioms[i],
         "issues": [{"code": c, "detail": d} for c, d in sorted(set(v))],
         "defective": any(c not in ADVISORY_CODES for c, _ in v)}
        for i, v in sorted(issues_by_row.items()) if v
    ]

    n = len(rows)
    defective = sum(1 for r in per_row if r["defective"])
    clean = n - defective
    return {
        "total": n,
        "clean": clean,
        "defective": defective,
        "advisory_only": len(per_row) - defective,
        "clean_pct": round(100.0 * clean / n, 2) if n else 0.0,
        "issue_counts": dict(counts.most_common()),
        "duplicate_groups": [
            {"key": k, "indices": v[:10], "n": len(v)}
            for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1]))[:20]
        ],
        "shared_meaning_groups": [
            {"meaning": k[:80], "n": len(v), "indices": v[:10]}
            for k, v in sorted(shared.items(), key=lambda kv: -len(kv[1]))[:20]
        ],
        "flagged_rows": per_row,
    }


# --------------------------------------------------------------------------- #
# LLM review shards
# --------------------------------------------------------------------------- #
REVIEW_INSTRUCTIONS = (
    "You are auditing an Arabic idiom/proverb knowledge base entry by entry. "
    "For EACH entry judge, using only the text given:\n"
    "  1. is_idiom: is this a genuine idiom/proverb (not a random sentence, a "
    "single word, or a dictionary artefact)?\n"
    "  2. meaning_correct: does the explanation actually explain THIS expression, "
    "and is it a figurative/literal gloss rather than an unrelated note?\n"
    "  3. example_ok: if an example is present, is it natural Arabic that uses the "
    "idiom correctly? (null if no example)\n"
    "  4. looks_ai_generated: does the wording read as machine-generated?\n"
    "Return ONLY a JSON list, one object per entry: "
    '{"index": int, "is_idiom": bool, "meaning_correct": bool, '
    '"example_ok": bool|null, "looks_ai_generated": bool, "note": "<=15 words"}. '
    "Do not invent Arabic text. Judge conservatively; flag only clear problems."
)


def emit_shards(rows: List[Dict[str, Any]], n_shards: int, shard_dir: Path,
                lang: str) -> List[Path]:
    """Write review-ready shards covering every row."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for s in range(n_shards):
        chunk = [
            {"index": i,
             "idiom": str(rows[i].get("idiom") or get_out(rows[i]).get("idiom") or ""),
             "literal_meanings": field_list(rows[i], "literal_meanings") or NAN,
             "figurative_meanings": field_list(rows[i], "figurative_meanings") or NAN,
             "examples": field_list(rows[i], "examples") or NAN}
            for i in range(s, len(rows), n_shards)
        ]
        p = shard_dir / f"shard_{s:03d}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"lang": lang, "shard": s, "n_shards": n_shards,
                       "count": len(chunk), "instructions": REVIEW_INSTRUCTIONS,
                       "entries": chunk}, f, ensure_ascii=False, indent=2)
        paths.append(p)
    logger.info("Wrote %d shards to %s (%d entries total)",
                len(paths), shard_dir, len(rows))
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input")
    p.add_argument("--lang", default="ar", choices=["ar", "zh", "en"])
    p.add_argument("--report", default=None)
    p.add_argument("--max-flagged", type=int, default=500,
                   help="Cap flagged rows written into the report JSON.")
    p.add_argument("--emit-shards", type=int, default=0,
                   help="Write N LLM-review shards covering every entry.")
    p.add_argument("--shard-dir", default=None)
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    if not args.input:
        raise SystemExit("--input is required (or use --self-test)")

    rows = read_jsonl(Path(args.input))
    logger.info("Auditing %d entries from %s (lang=%s)", len(rows), args.input, args.lang)
    rep = audit(rows, args.lang)

    print(f"\ntotal={rep['total']}  clean={rep['clean']} ({rep['clean_pct']}%)  "
          f"defective={rep['defective']}  advisory_only={rep['advisory_only']}")
    print("\nissue counts:")
    for code, c in rep["issue_counts"].items():
        if c:
            print(f"  {code:<32} {c}")

    if args.emit_shards:
        emit_shards(rows, args.emit_shards,
                    Path(args.shard_dir or (Path(args.input).parent / "audit_shards")),
                    args.lang)

    if args.report:
        trimmed = dict(rep)
        trimmed["flagged_rows"] = rep["flagged_rows"][: args.max_flagged]
        trimmed["flagged_rows_truncated"] = len(rep["flagged_rows"]) > args.max_flagged
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)
        logger.info("Wrote report to %s", args.report)
    return 0


def self_test() -> int:
    rows = [
        # 0 clean
        {"idiom": "اهل مكه ادري بشعابها",
         "output": {"idiom": "اهل مكه ادري بشعابها", "entities": NAN,
                    "literal_meanings": NAN,
                    "figurative_meanings": ["يضرب لمن هو اعلم بشان نفسه"],
                    "examples": NAN}},
        # 1 circular + leftover guillemets
        {"idiom": "«مثل عربي»",
         "output": {"idiom": "«مثل عربي»", "entities": NAN, "literal_meanings": NAN,
                    "figurative_meanings": ["مثل عربي"], "examples": NAN}},
        # 2 no meaning at all (violates criterion 2)
        {"idiom": "قول ماثور",
         "output": {"idiom": "قول ماثور", "entities": NAN, "literal_meanings": NAN,
                    "figurative_meanings": NAN, "examples": NAN}},
        # 3 example that does not contain its idiom
        {"idiom": "خرج من ايده",
         "output": {"idiom": "خرج من ايده", "entities": NAN, "literal_meanings": NAN,
                    "figurative_meanings": ["كنايه عن فقدان السيطره"],
                    "examples": ["جمله لا علاقه لها بالتعبير"]}},
        # 4 duplicate of 0 modulo diacritics/hamza
        {"idiom": "أهْلُ مَكَّةَ أدْرَى بِشِعَابِهَا",
         "output": {"idiom": "أهْلُ مَكَّةَ أدْرَى بِشِعَابِهَا", "entities": NAN,
                    "literal_meanings": NAN,
                    "figurative_meanings": ["شرح اخر"], "examples": NAN}},
        # 5 placeholder meaning + Latin contamination
        {"idiom": "proverb مثل",
         "output": {"idiom": "proverb مثل", "entities": NAN, "literal_meanings": NAN,
                    "figurative_meanings": ["..."], "examples": NAN}},
    ]
    rep = audit(rows, "ar")
    codes = rep["issue_counts"]

    assert codes.get("FURNITURE"), "guillemets not detected"
    assert codes.get("CIRCULAR_MEANING"), "circular definition not detected"
    assert codes.get("NO_MEANING"), "missing-meaning row not detected"
    assert codes.get("EXAMPLE_LACKS_IDIOM"), "bad example not detected"
    assert codes.get("PLACEHOLDER_MEANING"), "placeholder not detected"
    assert codes.get("LATIN_IN_ARABIC"), "Latin contamination not detected"
    # rows 0 and 4 are the same proverb once normalized
    assert codes.get("DUPLICATE_AFTER_NORMALIZATION") == 1, codes
    assert rep["clean"] == 1 and rep["total"] == 6, rep["clean"]

    # example containment must succeed when the idiom really is present
    ok = [{"idiom": "خرج من ايده",
           "output": {"idiom": "خرج من ايده", "figurative_meanings": ["كنايه"],
                      "examples": ["الموضوع خرج من ايده خلاص"]}}]
    assert not audit(ok, "ar")["issue_counts"].get("EXAMPLE_LACKS_IDIOM")

    # shard emission covers every row exactly once
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        paths = emit_shards(rows, 3, Path(td), "ar")
        seen = []
        for p in paths:
            seen += [e["index"] for e in json.loads(p.read_text())["entries"]]
        assert sorted(seen) == list(range(len(rows))), "shards must cover every row once"

    print("all audit_idioms.py self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
