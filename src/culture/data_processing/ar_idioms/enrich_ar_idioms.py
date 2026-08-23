#!/usr/bin/env python3
"""Enrich the Arabic idiom KB with `entities` and `literal_meanings`.

WHY THIS IS NEEDED
------------------
Every downstream analysis the Chinese pipeline performs (entity frequency, entity
embedding clustering, cross-lingual same-entity/same-meaning studies) keys off
``entities``, and several also use ``literal_meanings``. The Arabic KB has **0%
of both** — no upstream Arabic source supplies them (see the resource survey).
So the analyses cannot run at all until these two fields exist.

PROVENANCE
----------
These two fields are **LLM-generated** and every enriched row records that in
``meta.field_provenance``. This does not weaken the project's inclusion criterion
that sources be human-authored: that rule governs which upstream *datasets* we
ingest, and the idiom strings plus their ``figurative_meanings`` remain verbatim
human lexicography. Derived fields are labelled, never laundered.

VALIDATION
----------
An LLM asked for "entities" will happily invent them. Every returned entity is
therefore checked to actually occur in the idiom (after Arabic normalization, so
that clitics/diacritics do not cause false rejections); anything else is dropped
and counted. Literal meanings are length- and language-checked.

Resumable: results are appended to a JSONL cache keyed by the normalized idiom,
so a re-run only pays for what is missing.

Usage::

    export EMBEDDING_API_KEY="$APE_API_KEY"     # not needed here, but keeps envs uniform
    python enrich_ar_idioms.py \\
        --input  data/idioms/ar/idioms_merged_llm_formatted.jsonl \\
        --output data/idioms/ar/idioms_merged_llm_formatted.jsonl \\
        --cache  data/idioms/ar/enrich_cache.jsonl

    python enrich_ar_idioms.py --self-test
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# The MetaGen helper lives in the sibling meta-autoresearch repo.
_AUTORESEARCH = Path.home() / "local/git-repos/meta-autoresearch/code"
if _AUTORESEARCH.is_dir():
    sys.path.insert(0, str(_AUTORESEARCH))

from culture.data_processing.ar_idioms.normalize import normalize_ar  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.ar_idioms.enrich")

NAN = "NAN"

SYSTEM = (
    "You are an Arabic lexicographer. You are given an Arabic idiom/proverb and its "
    "figurative meaning (written by a human lexicographer). Produce two things:\n"
    "1. entities: the key CONTENT words that actually appear in the idiom — concrete "
    "nouns, named entities, animals, body parts, objects. Copy them VERBATIM from the "
    "idiom text. Do NOT invent words that are not in the idiom. Exclude particles, "
    "pronouns and generic verbs. Usually 1-4 items; use [] if none.\n"
    "2. literal_meaning: the word-for-word, non-figurative reading of the idiom in "
    "Modern Standard Arabic — what the words literally describe, ignoring the "
    "figurative sense. One sentence. If the idiom is already literal, say so briefly.\n"
    'Respond with ONLY a JSON object: {"entities": [...], "literal_meaning": "..."}'
)
USER_TMPL = "Idiom: {idiom}\nFigurative meaning: {fig}\n\nJSON:"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
_RE_AR_TOKEN = re.compile(r"[ء-ي]+")
# The model often prefixes the gloss with "the literal meaning:" — redundant.
_RE_LIT_PREFIX = re.compile(r"^\s*(?:المعنى\s+الحرفي|المعنى\s+الحرفيّ|حرفيًا|حرفياً)\s*[:：]?\s*")


def entity_in_idiom(entity: str, idiom: str) -> bool:
    """True if `entity` genuinely occurs in `idiom`.

    Compared after Arabic normalization so that diacritics, hamza carriers and a
    leading clitic on the idiom side do not cause a false rejection. A bare
    substring test on the raw strings would reject most valid entities.
    """
    e, i = normalize_ar(entity), normalize_ar(idiom)
    if not e or not i:
        return False
    if e in i:
        return True
    # Token-level: every token of the entity must appear as a substring of some
    # idiom token (handles the idiom carrying و/ال/ب on that word).
    etoks = _RE_AR_TOKEN.findall(e)
    itoks = _RE_AR_TOKEN.findall(i)
    if not etoks:
        return False
    return all(any(t in it for it in itoks) for t in etoks)


def validate(obj: Any, idiom: str) -> Tuple[List[str], str, Dict[str, int]]:
    """Coerce one model response into (entities, literal_meaning, drop_counts)."""
    drops = {"not_in_idiom": 0, "empty": 0}
    if not isinstance(obj, dict):
        return [], "", drops

    ents: List[str] = []
    for e in obj.get("entities") or []:
        e = str(e).strip()
        if not e:
            drops["empty"] += 1
            continue
        if not entity_in_idiom(e, idiom):
            drops["not_in_idiom"] += 1        # hallucinated — drop it
            continue
        if e not in ents:
            ents.append(e)

    lit = _RE_LIT_PREFIX.sub("", str(obj.get("literal_meaning") or "").strip()).strip()
    # Must be Arabic and a real sentence, not an echo of the idiom.
    if len(lit) < 8 or not _RE_AR_TOKEN.search(lit):
        lit = ""
    elif normalize_ar(lit) == normalize_ar(idiom):
        lit = ""
    return ents, lit, drops


def parse_json(raw: str) -> Any:
    """Tolerate code fences / surrounding prose around the JSON object."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else raw)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Enrichment
# --------------------------------------------------------------------------- #
def enrich_one(idiom: str, fig: str, model: Optional[str], max_tokens: int) -> Dict[str, Any]:
    from autoresearch.utils.llm import chat
    kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
    if model:
        kwargs["model"] = model
    raw = chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": USER_TMPL.format(idiom=idiom, fig=fig[:600])}],
        **kwargs,
    )
    return {"raw": raw}


def run(rows: List[Dict[str, Any]], cache: Dict[str, Dict[str, Any]],
        workers: int, model: Optional[str], max_tokens: int,
        cache_path: Optional[Path], limit: Optional[int]) -> Dict[str, int]:
    """Fill the cache for every row that is missing one. Thread-safe append."""
    todo = []
    for r in rows:
        key = normalize_ar(r["output"]["idiom"])
        if key and key not in cache:
            todo.append((key, r))
    if limit:
        todo = todo[:limit]
    logger.info("%d rows need enrichment (%d already cached)", len(todo), len(cache))
    if not todo:
        return {"called": 0, "failed": 0}

    lock = threading.Lock()
    fh = open(cache_path, "a", encoding="utf-8") if cache_path else None
    stats = {"called": 0, "failed": 0}

    def work(item):
        key, r = item
        o = r["output"]
        fig = o["figurative_meanings"]
        fig = fig[0] if isinstance(fig, list) and fig else ""
        try:
            res = enrich_one(o["idiom"], fig, model, max_tokens)
        except Exception as e:  # noqa: BLE001
            return key, {"error": f"{type(e).__name__}: {e}"[:200]}
        return key, res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            key, res = fut.result()
            with lock:
                cache[key] = res
                stats["called"] += 1
                if res.get("error"):
                    stats["failed"] += 1
                if fh:
                    fh.write(json.dumps({"key": key, **res}, ensure_ascii=False) + "\n")
                    fh.flush()
            if n % 250 == 0:
                logger.info("  %d/%d (%d failed)", n, len(todo), stats["failed"])
    if fh:
        fh.close()
    return stats


def apply_cache(rows: List[Dict[str, Any]], cache: Dict[str, Dict[str, Any]]
                ) -> Dict[str, int]:
    """Write validated entities/literal_meanings into the rows."""
    st = {"entities_filled": 0, "literal_filled": 0,
          "dropped_not_in_idiom": 0, "unparseable": 0, "no_cache": 0}
    for r in rows:
        o = r["output"]
        key = normalize_ar(o["idiom"])
        hit = cache.get(key)
        if not hit or hit.get("error"):
            st["no_cache"] += 1
            continue
        obj = parse_json(hit.get("raw", ""))
        if obj is None:
            st["unparseable"] += 1
            continue
        ents, lit, drops = validate(obj, o["idiom"])
        st["dropped_not_in_idiom"] += drops["not_in_idiom"]

        prov = r.setdefault("meta", {}).setdefault("field_provenance", {})
        if ents:
            o["entities"] = ents
            prov["entities"] = "llm_generated_validated"
            st["entities_filled"] += 1
        if lit:
            o["literal_meanings"] = [lit]
            prov["literal_meanings"] = "llm_generated"
            st["literal_filled"] += 1
    return st


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/idioms/ar/idioms_merged_llm_formatted.jsonl")
    p.add_argument("--output", default=None, help="Default: in place.")
    p.add_argument("--cache", default="data/idioms/ar/enrich_cache.jsonl")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--model", default=None, help="Override METAGEN_MODEL.")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--limit", type=int, default=None, help="Cap API calls (smoke test).")
    p.add_argument("--apply-only", action="store_true",
                   help="Skip API calls; just apply an existing cache.")
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    logger.info("Loaded %d rows from %s", len(rows), args.input)

    cache: Dict[str, Dict[str, Any]] = {}
    cache_path = Path(args.cache)
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                cache[d.pop("key")] = d
        logger.info("Loaded %d cached responses", len(cache))

    if not args.apply_only:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        st = run(rows, cache, args.workers, args.model, args.max_tokens,
                 cache_path, args.limit)
        logger.info("API: %s", st)

    st = apply_cache(rows, cache)
    logger.info("Applied: %s", st)

    out = Path(args.output or args.input)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    logger.info("Wrote %d rows to %s | entities %.1f%% | literal %.1f%%", n, out,
                100 * st["entities_filled"] / n, 100 * st["literal_filled"] / n)
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


def self_test() -> int:
    idiom = "أهل مكة أدرى بشعابها"
    # entity present verbatim
    assert entity_in_idiom("مكة", idiom)
    assert entity_in_idiom("شعابها", idiom)
    # present modulo diacritics/hamza on the idiom side
    assert entity_in_idiom("مكه", idiom)
    # hallucinated entity rejected
    assert not entity_in_idiom("الجمل", idiom)
    assert not entity_in_idiom("", idiom)

    # clitic on the idiom side must not cause a false rejection
    assert entity_in_idiom("دهر", "اكل عليه الدهر وشرب")

    ents, lit, drops = validate(
        {"entities": ["مكة", "الجمل", "شعابها", ""],
         "literal_meaning": "سكان مكة يعرفون طرق جبالها ومسالكها أكثر من غيرهم"},
        idiom)
    assert ents == ["مكة", "شعابها"], ents          # الجمل dropped, "" dropped
    assert drops["not_in_idiom"] == 1 and drops["empty"] == 1
    assert lit.startswith("سكان")
    # redundant "المعنى الحرفي:" prefix is stripped
    _, lit3, _ = validate({"literal_meaning": "المعنى الحرفي: سكان مكة أعرف بجبالها"}, idiom)
    assert lit3 == "سكان مكة أعرف بجبالها", lit3

    # literal meaning that merely echoes the idiom is rejected
    _, lit2, _ = validate({"entities": [], "literal_meaning": idiom}, idiom)
    assert lit2 == ""
    # too short / non-Arabic rejected
    assert validate({"literal_meaning": "short"}, idiom)[1] == ""
    assert validate({"literal_meaning": "a literal english gloss here"}, idiom)[1] == ""

    # tolerant JSON parsing
    assert parse_json('```json\n{"entities": ["a"]}\n```')["entities"] == ["a"]
    assert parse_json("garbage") is None
    assert parse_json("") is None

    # apply_cache writes provenance and skips errored/uncached rows
    rows = [{"output": {"idiom": idiom, "entities": NAN, "literal_meanings": NAN,
                        "figurative_meanings": ["يضرب لمن هو أعلم بشأنه"]}, "meta": {}},
            {"output": {"idiom": "مثل آخر هنا", "entities": NAN,
                        "literal_meanings": NAN,
                        "figurative_meanings": ["شرح"]}, "meta": {}}]
    cache = {normalize_ar(idiom): {"raw": json.dumps(
        {"entities": ["مكة"], "literal_meaning": "سكان مكة أعرف بطرق جبالها"},
        ensure_ascii=False)}}
    st = apply_cache(rows, cache)
    assert rows[0]["output"]["entities"] == ["مكة"]
    assert rows[0]["meta"]["field_provenance"]["entities"] == "llm_generated_validated"
    assert rows[0]["meta"]["field_provenance"]["literal_meanings"] == "llm_generated"
    assert rows[1]["output"]["entities"] == NAN and st["no_cache"] == 1

    print("all enrich_ar_idioms.py self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
