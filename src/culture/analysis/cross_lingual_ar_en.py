#!/usr/bin/env python3
"""Cross-lingual Arabic↔English idiom analysis.

The Arabic counterpart of the two cross-lingual scripts in this directory:

============  ==================================================  ======================================
sub-command    question                                            zh/en counterpart
============  ==================================================  ======================================
``pairs``      **Same meaning, different entity** — which Arabic   ``cross_lingual_same_meaning_diff_entity.py``
               and English idioms mean the same thing, and what
               imagery does each culture reach for?
``entities``   **Same entity, different meaning** — where both     ``cross_lingual_same_entity_diff_meaning.py``
               cultures build idioms on the same object, do they
               mean the same thing?
============  ==================================================  ======================================

Method notes
------------
* Both sides are embedded with the SAME model (MetaGen ``text-embedding-3-small``
  @512d). The stored ``figurative_embeddings.npz`` for zh/en came from
  Qwen3-Embedding and lives in a different vector space, so it is deliberately
  **not** reused — mixing spaces would silently produce garbage similarities.
* Arabic entities are translated to English with the MetaGen chat model before
  matching, mirroring what the zh pipeline does with its translation step.
* All embeddings and translations are disk-cached, so re-runs cost nothing.

Usage::

    export EMBEDDING_API_KEY="$APE_API_KEY"
    python cross_lingual_ar_en.py pairs    --outdir data/idioms/ar/analysis
    python cross_lingual_ar_en.py entities --outdir data/idioms/ar/analysis
    python cross_lingual_ar_en.py --self-test
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_AUTORESEARCH = Path.home() / "local/git-repos/meta-autoresearch/code"
if _AUTORESEARCH.is_dir():
    sys.path.insert(0, str(_AUTORESEARCH))

from culture.analysis.analyze_ar_idioms import (  # noqa: E402
    NAN, embed_cached, fld, l2norm, load_rows, write_json, write_jsonl,
)
from culture.data_processing.ar_idioms.normalize import normalize_ar  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.analysis.ar_en")


# Arabic proverb dictionaries (al-Maydani, Taymur) write the entry as
# ETYMOLOGY/ANECDOTE first and only then the actual usage gloss, introduced by
# "يُضرب في/لمن ..." ("it is said of / applied to ..."). So a raw
# `figurative_meanings` string averages 191 chars — 3x the English side (64) and
# 10x the Chinese side (18) — and most of that length is a story about which
# Companion said what, which is NOT what the English gloss means. Embedding the
# whole passage buries the comparable signal: at the Chinese pipeline's own
# threshold of 0.70 it yields 21 ar-en pairs against zh-en's 37,045.
#
# Trimming to the يضرب clause recovers a gloss-length string (median 54 chars)
# for the 38.2% of meanings that have one (49.3% of the passages over 150 chars).
YUDRAB_RE = re.compile(r"[يت]ُ?ضْ?رَ?ب")


def usage_gloss(meaning: str, min_len: int = 10) -> str:
    """Trim an Arabic dictionary entry to its "يُضرب ..." usage clause.

    Returns `meaning` unchanged when there is no such clause, or when the clause
    is too short to embed meaningfully.
    """
    m = YUDRAB_RE.search(meaning)
    if not m:
        return meaning
    clause = meaning[m.start():].strip()
    return clause if len(clause) >= min_len else meaning


def en_fld(row: Dict[str, Any], name: str) -> List[str]:
    """Field accessor for the English KB (same {output:{...}} shape)."""
    v = (row.get("output") or {}).get(name)      # rows may have "output": null
    if not v or v == NAN:
        return []
    return list(v) if isinstance(v, list) else [v]


# --------------------------------------------------------------------------- #
# A5 — same meaning, different entity
# --------------------------------------------------------------------------- #
def cmd_pairs(args) -> Dict[str, Any]:
    ar = load_rows(args.ar_input)
    en = load_rows(args.en_input)
    if args.max_en:
        en = en[:args.max_en]
    outdir = Path(args.outdir)
    logger.info("ar=%d  en=%d", len(ar), len(en))

    # One embedding per (idiom, meaning). Arabic side uses the ENGLISH gloss when
    # one exists (796 human translations) and the Arabic gloss otherwise — the
    # embedding model aligns cross-lingually, but an English-vs-English comparison
    # is measurably tighter where we can get it for free.
    ar = [r for r in ar if r.get("output")]
    en = [r for r in en if r.get("output")]
    ar_items: List[Tuple[int, str]] = []
    n_trimmed = 0
    for i, r in enumerate(ar):
        en_ms = en_fld(r, "figurative_meanings_en")
        # English glosses are already gloss-shaped; only the Arabic exegesis needs trimming.
        ms = en_ms or [usage_gloss(m) if args.usage_gloss else m
                       for m in fld(r, "figurative_meanings")]
        if not en_ms and args.usage_gloss:
            n_trimmed += sum(1 for a, b in zip(ms, fld(r, "figurative_meanings")) if a != b)
        for m in ms[:2]:
            if len(m) >= 10:
                ar_items.append((i, m[:400]))
    if args.usage_gloss:
        logger.info("trimmed %d Arabic meanings to their يضرب usage clause", n_trimmed)

    # Translate the Arabic side to English so the comparison is English<->English
    # (see translate_meanings for the measurement that forces this).
    if args.translate:
        need = [m for _, m in ar_items if not m.isascii()]
        if args.max_translate:
            need = need[:args.max_translate]
            logger.info("translation capped at %d meanings (--max_translate)",
                        args.max_translate)
        tr = translate_meanings(need, outdir / "cache" / "meaning_translations_ar_en.jsonl",
                                model=args.model)
        translated = [(i, tr.get(m) or m, m) for i, m in ar_items]
        n_ok = sum(1 for i, new, old in translated if new != old)
        logger.info("using %d translated Arabic meanings (%d left untranslated)",
                    n_ok, len(translated) - n_ok)
        ar_items = [(i, m) for i, m, _ in translated if len(m) >= 10]
        ar_source = {(i, m): src for i, m, src in translated if len(m) >= 10}
    else:
        ar_source = {}
    en_items: List[Tuple[int, str]] = []
    for j, r in enumerate(en):
        for m in en_fld(r, "figurative_meanings")[:2]:
            if len(m) >= 10:
                en_items.append((j, m[:400]))
    logger.info("meanings: ar=%d  en=%d", len(ar_items), len(en_items))

    A = l2norm(embed_cached([m for _, m in ar_items],
                            outdir / "cache" / "ar_meaning_embeddings.jsonl"))
    E = l2norm(embed_cached([m for _, m in en_items],
                            outdir / "cache" / "en_meaning_embeddings.jsonl"))

    pairs = []
    B = 512
    for s in range(0, len(A), B):
        sims = A[s:s + B] @ E.T                       # (b, |en|)
        for bi in range(sims.shape[0]):
            gi = s + bi
            row = sims[bi]
            top = np.argpartition(-row, min(args.top_k, len(row) - 1))[:args.top_k]
            for j in top[np.argsort(-row[top])]:
                sc = float(row[j])
                if sc < args.threshold:
                    break
                ai, am = ar_items[gi]
                ej, em = en_items[j]
                pairs.append({
                    "ar_idiom": ar[ai]["output"]["idiom"],
                    # `ar_matched_meaning` is the text that was actually embedded; when
                    # --translate is on that is an English paraphrase, and the Arabic it
                    # came from is kept in `ar_matched_meaning_source`.
                    "ar_matched_meaning": am,
                    "ar_matched_meaning_source": ar_source.get((ai, am), am),
                    "ar_figurative_meanings": fld(ar[ai], "figurative_meanings"),
                    "ar_entities": fld(ar[ai], "entities"),
                    "ar_literal_meanings": fld(ar[ai], "literal_meanings"),
                    "ar_variety": ar[ai]["output"].get("variety", NAN),
                    "ar_register": ar[ai]["output"].get("register", NAN),
                    "en_idiom": en[ej]["output"]["idiom"],
                    "en_matched_meaning": em,
                    "en_figurative_meanings": en_fld(en[ej], "figurative_meanings"),
                    "en_entities": en_fld(en[ej], "entities"),
                    "similarity": round(sc, 4),
                })
        if (s // B) % 10 == 0:
            logger.info("  matched %d/%d arabic meanings", min(s + B, len(A)), len(A))

    pairs.sort(key=lambda p: -p["similarity"])
    write_jsonl(pairs, outdir / "cross_lingual_pairs_ar_en.jsonl")

    # Entity-overlap statistics — the actual research question.
    both = diff = overlap = only_ar = only_en = neither = 0
    for p in pairs:
        a, e = set(p["ar_entities"]), set(p["en_entities"])
        if a and e:
            both += 1
            # Arabic entities are Arabic strings and English ones English, so a
            # raw set intersection is almost always empty; this counts the
            # *availability* of the comparison, while `entities` (A6) does the
            # translated comparison properly.
            if a & e:
                overlap += 1
            else:
                diff += 1
        elif a:
            only_ar += 1
        elif e:
            only_en += 1
        else:
            neither += 1
    stats = {"total_pairs": len(pairs), "pairs_both_have_entities": both,
             "pairs_with_different_entities": diff,
             "pairs_with_overlapping_entities": overlap,
             "pairs_only_ar_has_entities": only_ar,
             "pairs_only_en_has_entities": only_en,
             "pairs_neither_has_entities": neither,
             "threshold": args.threshold,
             "unique_ar_idioms": len({p["ar_idiom"] for p in pairs}),
             "unique_en_idioms": len({p["en_idiom"] for p in pairs})}
    write_json(stats, outdir / "cross_lingual_stats_ar_en.json")

    print(f"\n{len(pairs)} ar-en pairs @ sim>={args.threshold}")
    for p in pairs[:8]:
        print(f"  {p['similarity']:.3f}  {p['ar_idiom'][:38]:<40} <-> {p['en_idiom'][:34]}")
        print(f"          ar_ents={p['ar_entities']}  en_ents={p['en_entities']}")
    return stats


# --------------------------------------------------------------------------- #
# A6 — same entity, different meaning
# --------------------------------------------------------------------------- #
def translate_entities(ents: List[str], cache_path: Path, batch: int = 40,
                       model: Optional[str] = None) -> Dict[str, str]:
    """Arabic entity -> English lemma, via the MetaGen chat model (disk-cached)."""
    cache: Dict[str, str] = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                cache[d["ar"]] = d["en"]
    missing = [e for e in dict.fromkeys(ents) if e not in cache]
    if missing:
        from autoresearch.utils.llm import chat
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            for i in range(0, len(missing), batch):
                chunk = missing[i:i + batch]
                prompt = (
                    "Translate each Arabic word to its single most common English "
                    "noun/lemma. Reply with ONLY a JSON object mapping each input "
                    "verbatim to one lowercase English word.\n\n"
                    + json.dumps(chunk, ensure_ascii=False))
                kw = {"max_tokens": 1200}
                if model:
                    kw["model"] = model
                try:
                    raw = chat([{"role": "user", "content": prompt}], **kw)
                    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
                    obj = json.loads(m.group(0)) if m else {}
                except Exception as e:  # noqa: BLE001
                    logger.warning("translate batch failed: %s", e)
                    obj = {}
                for a in chunk:
                    en = str(obj.get(a, "")).strip().lower()
                    cache[a] = en
                    f.write(json.dumps({"ar": a, "en": en}, ensure_ascii=False) + "\n")
                f.flush()
                logger.info("  translated %d/%d", min(i + batch, len(missing)), len(missing))
    return cache


def translate_meanings(meanings: List[str], cache_path: Path, batch: int = 20,
                       model: Optional[str] = None) -> Dict[str, str]:
    """Arabic figurative-meaning gloss -> one-line English paraphrase (disk-cached).

    Needed because the embedding space does not align Arabic with English at the
    sentence level. Measured on this KB, top-1 similarity against 4,538 English
    glosses:

        Arabic-language meanings (n=9,596)   mean 0.354  p99 0.465  **max 0.607**
        human English glosses    (n=1,590)   mean 0.502  p99 0.695  max 0.825

    Not one Arabic-language meaning reaches the Chinese pipeline's 0.70 threshold
    — and Chinese *does* reach it directly (zh<->en max 0.879, 37,045 pairs). The
    content is not the problem: the same Arabic entries whose meaning happens to
    carry a human English gloss match fine. So the Arabic side is translated to
    English first and matched English<->English, which is exactly the trick that
    makes the A6 entity analysis work.
    """
    cache: Dict[str, str] = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                cache[d["ar"]] = d["en"]
        logger.info("meaning-translation cache: %d", len(cache))
    missing = [m for m in dict.fromkeys(meanings) if m not in cache]
    if missing:
        from autoresearch.utils.llm import chat
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            for i in range(0, len(missing), batch):
                chunk = missing[i:i + batch]
                prompt = (
                    "Each item below is the meaning of an Arabic proverb or idiom, as "
                    "written in an Arabic dictionary. For each one, give a SHORT English "
                    "paraphrase of what the proverb MEANS (one clause, under 15 words). "
                    "Ignore etymology, anecdotes and who said it. Reply with ONLY a JSON "
                    "array of strings, same length and order as the input.\n\n"
                    + json.dumps(chunk, ensure_ascii=False))
                kw = {"max_tokens": 2000}
                if model:
                    kw["model"] = model
                try:
                    raw = chat([{"role": "user", "content": prompt}], **kw)
                    m = re.search(r"\[.*\]", raw or "", re.DOTALL)
                    arr = json.loads(m.group(0)) if m else []
                except Exception as e:  # noqa: BLE001 — a dropped batch is recoverable
                    logger.warning("translate batch failed: %s", str(e)[:120])
                    arr = []
                if len(arr) != len(chunk):
                    arr = arr + [""] * (len(chunk) - len(arr))
                for a, e in zip(chunk, arr):
                    v = str(e).strip()
                    cache[a] = v
                    f.write(json.dumps({"ar": a, "en": v}, ensure_ascii=False) + "\n")
                f.flush()
                logger.info("  translated %d/%d meanings", min(i + batch, len(missing)),
                            len(missing))
    return cache


def cmd_entities(args) -> Dict[str, Any]:
    ar = load_rows(args.ar_input)
    en = load_rows(args.en_input)
    if args.max_en:
        en = en[:args.max_en]
    outdir = Path(args.outdir)

    ar = [r for r in ar if r.get("output")]
    en = [r for r in en if r.get("output")]
    ar_ent: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(ar):
        for e in fld(r, "entities"):
            k = re.sub(r"^ال", "", normalize_ar(e))
            if len(k) >= 2:
                ar_ent[k].append(i)
    en_ent: Dict[str, List[int]] = defaultdict(list)
    for j, r in enumerate(en):
        for e in en_fld(r, "entities"):
            k = e.strip().lower()
            if len(k) >= 2:
                en_ent[k].append(j)

    top_ar = [e for e, v in sorted(ar_ent.items(), key=lambda kv: -len(kv[1]))
              [:args.top_entities]]
    write_json(top_ar, outdir / "top_entities_ar.json")
    write_json([e for e, v in sorted(en_ent.items(), key=lambda kv: -len(kv[1]))[:200]],
               outdir / "top_entities_en.json")

    tr = translate_entities(top_ar, outdir / "cache" / "entity_translations_ar_en.jsonl",
                            model=args.model)

    shared = []
    for a in top_ar:
        e = tr.get(a, "")
        if not e or e not in en_ent:
            continue
        ai, ej = ar_ent[a], en_ent[e]
        if len(ai) < args.min_count or len(ej) < args.min_count:
            continue
        shared.append({
            "ar_entity": a, "en_entity": e,
            "ar_idiom_count": len(ai), "en_idiom_count": len(ej),
            "ar_idioms": [{"idiom": ar[i]["output"]["idiom"],
                           "figurative_meanings": fld(ar[i], "figurative_meanings")[:1],
                           "variety": ar[i]["output"].get("variety", NAN)}
                          for i in ai[:args.max_examples]],
            "en_idioms": [{"idiom": en[j]["output"]["idiom"],
                           "figurative_meanings": en_fld(en[j], "figurative_meanings")[:1]}
                          for j in ej[:args.max_examples]],
        })
    shared.sort(key=lambda x: -(x["ar_idiom_count"] + x["en_idiom_count"]))
    write_jsonl(shared, outdir / "shared_entities_ar_en.jsonl")

    stats = {"ar_unique_entities": len(ar_ent), "en_unique_entities": len(en_ent),
             "ar_entities_considered": len(top_ar),
             "translated": sum(1 for a in top_ar if tr.get(a)),
             "shared_with_english": len(shared)}
    write_json(stats, outdir / "shared_entity_stats_ar_en.json")

    print(f"\n{len(shared)} entities shared between Arabic and English idiom stocks")
    for s in shared[:12]:
        print(f"  {s['ar_entity']:<12} = {s['en_entity']:<12} "
              f"ar:{s['ar_idiom_count']:>4}  en:{s['en_idiom_count']:>4}")
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="pairs",
                   choices=["pairs", "entities", "all"])
    p.add_argument("--ar_input", default="data/idioms/ar/idioms_merged_llm_formatted.jsonl")
    p.add_argument("--en_input", default="data/idioms/en/idioms_merged_llm_formatted.jsonl")
    p.add_argument("--outdir", default="data/idioms/ar/analysis")
    # 0.70 matches the Chinese pipeline (data/idioms/cross_lingual_pairs.jsonl,
    # 37,045 pairs, max observed similarity 0.879), so the two are comparable.
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--no-usage-gloss", dest="usage_gloss", action="store_false",
                   help="Embed the FULL Arabic dictionary entry instead of trimming it to "
                        "the يضرب usage clause. Off by default because the untrimmed "
                        "etymological commentary buries the comparable signal.")
    p.set_defaults(usage_gloss=True)
    p.add_argument("--no-translate", dest="translate", action="store_false",
                   help="Match raw Arabic against English instead of translating the "
                        "Arabic side first. Measured to yield almost nothing: no "
                        "Arabic-language meaning exceeds 0.607 similarity to any English "
                        "gloss, against a 0.70 threshold.")
    p.set_defaults(translate=True)
    p.add_argument("--max_translate", type=int, default=None,
                   help="Cap how many Arabic meanings get translated (API cost).")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_en", type=int, default=None, help="Cap English rows (cost).")
    p.add_argument("--top_entities", type=int, default=150)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--max_examples", type=int, default=8)
    p.add_argument("--model", default=None)
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    cmds = ["pairs", "entities"] if args.command == "all" else [args.command]
    for c in cmds:
        logger.info("=== %s ===", c)
        (cmd_pairs if c == "pairs" else cmd_entities)(args)
    return 0


def self_test() -> int:
    # en_fld honours the NAN convention and both shapes
    assert en_fld({"output": {"x": NAN}}, "x") == []
    assert en_fld({"output": {"x": ["a"]}}, "x") == ["a"]
    assert en_fld({"output": {"x": "a"}}, "x") == ["a"]
    assert en_fld({"output": {}}, "x") == []
    assert en_fld({"output": None}, "x") == []       # explicit null output

    # entity key normalization must merge ال-prefixed and diacritized forms
    k = lambda e: re.sub(r"^ال", "", normalize_ar(e))          # noqa: E731
    assert k("الحمي") == k("حمي") == "حمي"
    assert k("مَكَّة") == k("مكة")

    # cosine bookkeeping used by cmd_pairs
    A = l2norm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    E = l2norm(np.array([[1.0, 0.0]], dtype=np.float32))
    sims = A @ E.T
    assert abs(sims[0, 0] - 1.0) < 1e-6 and abs(sims[1, 0]) < 1e-6

    print("all cross_lingual_ar_en.py self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
