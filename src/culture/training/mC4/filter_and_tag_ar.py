#!/usr/bin/env python3
"""Filter an Arabic web corpus for idiom-bearing documents and tag them.

Produces the same artefact the Hindi/Chinese CPT corpora ship as::

    <out>/tagged_00000.json.gz      # one JSON object per line
    {"text": "<document>\\n\\n<knowledge block>", "source": ...,
     "matched_idioms": [...], "url": ..., "doc_index": N}

which ``continued_pretraining/prepare_data.py`` then reshards for LLaMA-Factory.

WHY ARABIC NEEDS ITS OWN SCRIPT
-------------------------------
``download_and_filter_mc4.py`` matches raw substrings. For Arabic that recovers
almost nothing, because dictionaries are vocalized and web text is not. Measured
on 60k FineWeb-2 ``arb_Arab`` documents with the 7,730-entry inventory:

======================================  =====  ==============
matching                                 docs   distinct idioms
======================================  =====  ==============
raw citation form (the old behaviour)       10               8
+ normalization (Tier 0)                   477             308
+ stem matching (Tier 2, contiguous)      ~633            ~382
======================================  =====  ==============

So this script runs **two passes** and takes the union:

* **Tier 0 — normalized Aho-Corasick.** Dediacritize, fold hamza/alif-maksura/
  teh-marbuta, strip tatweel + bidi, collapse whitespace, on BOTH the patterns and
  the document. Substring semantics are kept deliberately: a proclitic (و/ف/ب/ال)
  on the first word or an enclitic pronoun on the last word only adds characters
  outside the match, and enforcing token boundaries would destroy 21.4% of real
  hits. Patterns shorter than ``MIN_PATTERN_CHARS`` are dropped — that is where
  virtually all false positives come from.
* **Tier 2 — contiguous stem-sequence match. OFF BY DEFAULT.** It catches interior
  inflection Tier 0 cannot (``عينه فيه`` vs ``عينها فيها``) and adds **+19.7%**
  documents — but on real FineWeb-2 text its additions were measured to be
  overwhelmingly FALSE POSITIVES. Inspecting the stem-only hits: ``كل شيء بأوان``
  fired on an article about sexual dysfunction and ``حبلك على غاربك`` on a Tunisian
  history page and a question about school problems — in none of them does the
  idiom occur. Light stemming collapses frequent words to short stems that then
  coincide by chance in long documents. Enable with ``--use_stem`` only if you also
  raise ``MIN_STEM_TOKENS`` and accept the precision cost.

Measured on 4,000 streamed FineWeb-2 ``arb_Arab`` documents with the 10,386-entry
inventory: raw substring **0** docs, Tier 0 **142** docs (3.55%), Tier 0+2 **170**.
Tier-0 precision on a manual read of sampled contexts was 5/5 genuine idiom uses.

Frequency capping mirrors the published recipe: at most ``--max-docs-per-idiom``
documents per idiom, so high-frequency expressions cannot swamp rare ones.

Usage::

    # smoke test on a few thousand streamed docs
    python filter_and_tag_ar.py --limit 5000 --out /tmp/ar_tagged

    # full run
    python filter_and_tag_ar.py --out $DATA_ROOT/ar-amthal-cpt/data

    python filter_and_tag_ar.py --self-test
"""

import argparse
import gzip
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from culture.data_processing.ar_idioms.normalize import (  # noqa: E402
    MIN_PATTERN_CHARS,
    normalize_ar,
    normalize_idiom_for_matching,
)
from culture.data_processing.ar_idioms.stem import StemMatcher  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.ar.filter_tag")

NAN = "NAN"


# --------------------------------------------------------------------------- #
# Idiom inventory
# --------------------------------------------------------------------------- #
def load_idioms(path: str) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    logger.info("loaded %d idioms from %s", len(rows), path)
    return rows


def _lst(o: Dict[str, Any], k: str) -> List[str]:
    v = o.get(k, NAN)
    if v == NAN or not v:
        return []
    return list(v) if isinstance(v, list) else [v]


class ArabicIdiomMatcher:
    """Tier 0 (normalized substring) ∪ Tier 2 (contiguous stem) matching."""

    def __init__(self, idioms: List[Dict[str, Any]], use_stem: bool = True,
                 min_pattern_chars: int = MIN_PATTERN_CHARS):
        self.by_surface: Dict[str, Dict[str, Any]] = {}
        self.skipped_short = 0
        patterns: Dict[str, str] = {}
        for r in idioms:
            o = r["output"]
            surface = o["idiom"]
            key = normalize_idiom_for_matching(surface)
            self.by_surface[surface] = o
            if not key:
                continue
            if len(key) < min_pattern_chars:
                self.skipped_short += 1
                continue
            patterns.setdefault(key, surface)

        self._patterns = patterns
        self._automaton = None
        try:
            import ahocorasick
            A = ahocorasick.Automaton()
            for k, v in patterns.items():
                A.add_word(k, v)
            A.make_automaton()
            self._automaton = A
        except ImportError:
            logger.warning("pyahocorasick missing — falling back to a slow scan")

        self.stem = StemMatcher([r["output"]["idiom"] for r in idioms]) if use_stem else None
        logger.info("matcher: %d normalized patterns (%d too short), stem patterns=%s",
                    len(patterns), self.skipped_short,
                    len(self.stem) if self.stem else "off")

    def match(self, text: str) -> Set[str]:
        """Return the set of matched idiom SURFACE forms."""
        hay = normalize_ar(text)
        if not hay:
            return set()
        if self._automaton is not None:
            hits = {v for _, v in self._automaton.iter(hay)}
        else:
            hits = {v for k, v in self._patterns.items() if k in hay}
        if self.stem is not None:
            hits |= self.stem.find(text)
        return hits


# --------------------------------------------------------------------------- #
# Knowledge block
# --------------------------------------------------------------------------- #
KB_HEADER = "المعاني الاصطلاحية للتعابير الواردة في النص:"   # "Idiomatic meanings of the expressions above:"


def knowledge_block(surfaces: Iterable[str], by_surface: Dict[str, Dict[str, Any]],
                    max_meanings: int = 2) -> str:
    """Render the appended knowledge block, in Arabic, for the matched idioms."""
    lines: List[str] = []
    for s in surfaces:
        o = by_surface.get(s)
        if not o:
            continue
        parts = [f"- {s}"]
        fig = _lst(o, "figurative_meanings")[:max_meanings]
        lit = _lst(o, "literal_meanings")[:1]
        if fig:
            parts.append("  المعنى المجازي: " + " | ".join(fig))
        if lit:
            parts.append("  المعنى الحرفي: " + lit[0])
        ents = _lst(o, "entities")
        if ents:
            parts.append("  العناصر: " + "، ".join(ents))
        reg = o.get("region") or o.get("variety_region")
        if reg and reg != NAN:
            parts.append("  اللهجة: " + "، ".join(reg if isinstance(reg, list) else [reg]))
        lines.append("\n".join(parts))
    if not lines:
        return ""
    return KB_HEADER + "\n" + "\n".join(lines)


def tag_document(text: str, surfaces: List[str],
                 by_surface: Dict[str, Dict[str, Any]]) -> str:
    kb = knowledge_block(surfaces, by_surface)
    return f"{text}\n\n{kb}" if kb else text


# --------------------------------------------------------------------------- #
# Corpus streaming
# --------------------------------------------------------------------------- #
def stream_corpus(dataset: str, config: Optional[str], split: str,
                  text_field: str, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True,
                      token=os.environ.get("HF_TOKEN"))
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        t = row.get(text_field) or ""
        if t:
            yield {"text": t, "url": row.get("url", ""), "doc_index": i}


class ShardWriter:
    """Rolling gzip-JSONL writer producing tagged_NNNNN.json.gz."""

    def __init__(self, outdir: Path, docs_per_shard: int = 50_000):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.docs_per_shard = docs_per_shard
        self.n = 0
        self.shard = 0
        self.fh = None

    def _roll(self):
        if self.fh:
            self.fh.close()
        p = self.outdir / f"tagged_{self.shard:05d}.json.gz"
        self.fh = gzip.open(p, "wt", encoding="utf-8")
        logger.info("writing %s", p)
        self.shard += 1

    def write(self, obj: Dict[str, Any]):
        if self.fh is None or self.n % self.docs_per_shard == 0:
            self._roll()
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self):
        if self.fh:
            self.fh.close()


# --------------------------------------------------------------------------- #
# Main filtering loop
# --------------------------------------------------------------------------- #
def run(args) -> Dict[str, Any]:
    idioms = load_idioms(args.idioms)
    matcher = ArabicIdiomMatcher(idioms, use_stem=args.use_stem)
    writer = ShardWriter(Path(args.out), args.docs_per_shard)

    per_idiom = Counter()
    stats = Counter()
    kept_chars = 0
    for doc in stream_corpus(args.dataset, args.config, args.split,
                             args.text_field, args.limit):
        stats["scanned"] += 1
        if len(doc["text"]) < args.min_doc_chars:
            stats["too_short"] += 1
            continue
        hits = matcher.match(doc["text"])
        if not hits:
            continue
        stats["matched"] += 1

        # Frequency capping: keep the doc only for idioms still under the cap.
        keep = [s for s in hits if per_idiom[s] < args.max_docs_per_idiom]
        if not keep:
            stats["capped_out"] += 1
            continue
        for s in keep:
            per_idiom[s] += 1

        text = tag_document(doc["text"], keep, matcher.by_surface)
        kept_chars += len(text)
        writer.write({"text": text, "source": f"{args.dataset}:{args.config or ''}",
                      "url": doc.get("url", ""), "doc_index": doc["doc_index"],
                      "matched_idioms": sorted(keep)})
        stats["written"] += 1
        if stats["scanned"] % args.log_every == 0:
            logger.info("scanned=%d matched=%d written=%d idioms_hit=%d",
                        stats["scanned"], stats["matched"], stats["written"],
                        len(per_idiom))
    writer.close()

    out = {
        **dict(stats),
        "unique_idioms_hit": len(per_idiom),
        "inventory_size": len(idioms),
        "coverage_pct": round(100 * len(per_idiom) / max(len(idioms), 1), 2),
        "match_rate_pct": round(100 * stats["matched"] / max(stats["scanned"], 1), 3),
        "kept_chars": kept_chars,
        "top_idioms": per_idiom.most_common(20),
    }
    rep = Path(args.out) / "filter_report.json"
    with open(rep, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("report -> %s", rep)
    print(json.dumps({k: v for k, v in out.items() if k != "top_idioms"},
                     ensure_ascii=False, indent=2))
    return out


# --------------------------------------------------------------------------- #
# Precision / recall probe
# --------------------------------------------------------------------------- #
def cmd_measure(args) -> Dict[str, Any]:
    """Compare Tier 0 vs Tier 0+2 on a streamed sample, and dump hits to inspect.

    There is no gold-labelled Arabic corpus for this task, so we report the
    *relative* recall of each tier plus a sample of matched contexts for manual
    precision judgement — which is exactly how the tier design was validated.
    """
    idioms = load_idioms(args.idioms)
    t0 = ArabicIdiomMatcher(idioms, use_stem=False)
    t02 = ArabicIdiomMatcher(idioms, use_stem=True)

    n = 0
    raw_hits = t0_docs = t02_docs = 0
    tier0_examples: List[Dict[str, str]] = []
    stem_only_examples: List[Dict[str, str]] = []
    idiom_doc_freq: Counter = Counter()
    raw_patterns = [r["output"]["idiom"] for r in idioms][:2000]

    def context(text: str, surface: str, width: int = 90) -> str:
        """Window around the ACTUAL match, so precision can be judged.

        Showing the document head instead (the earlier bug) makes every example
        unreadable for this purpose — you cannot tell a true hit from a false one.
        """
        key = normalize_idiom_for_matching(surface)
        hay = normalize_ar(text)
        i = hay.find(key) if key else -1
        if i < 0:
            return "…(matched by stem, not substring)… " + text[:2 * width]
        # map back approximately: normalization is near length-preserving
        lo, hi = max(0, i - width), min(len(hay), i + len(key) + width)
        return ("…" if lo else "") + hay[lo:hi] + ("…" if hi < len(hay) else "")

    for doc in stream_corpus(args.dataset, args.config, args.split,
                             args.text_field, args.limit):
        n += 1
        text = doc["text"]
        if any(p in text for p in raw_patterns):                # cheap raw baseline
            raw_hits += 1
        h0 = t0.match(text)
        h2 = t02.match(text)
        if h0:
            t0_docs += 1
        if h2:
            t02_docs += 1
        for s_ in h2:
            idiom_doc_freq[s_] += 1
        if h0 and len(tier0_examples) < args.n_examples:
            s_ = sorted(h0, key=len, reverse=True)[0]
            tier0_examples.append({"idiom": s_, "context": context(text, s_)})
        extra = h2 - h0
        if extra and len(stem_only_examples) < args.n_examples:
            s_ = sorted(extra, key=len, reverse=True)[0]
            stem_only_examples.append({"idiom": s_, "context": context(text, s_)})

    res = {"scanned": n, "raw_substring_docs": raw_hits,
           "tier0_docs": t0_docs, "tier0_plus_stem_docs": t02_docs,
           "tier0_match_rate_pct": round(100 * t0_docs / max(n, 1), 2),
           "tier0_gain_vs_raw": round(t0_docs / max(raw_hits, 1), 1),
           "stem_gain_vs_tier0_pct": round(100 * (t02_docs - t0_docs) / max(t0_docs, 1), 1),
           # An idiom firing on a large share of ALL documents is almost certainly a
           # common fixed phrase rather than a discriminative idiom -> over-matching.
           "over_matching_idioms": [
               {"idiom": k, "docs": v, "pct_of_corpus": round(100 * v / max(n, 1), 2)}
               for k, v in idiom_doc_freq.most_common(15)],
           "tier0_examples": tier0_examples,
           "stem_only_examples": stem_only_examples}
    print(json.dumps(res, ensure_ascii=False, indent=2)[:3000])
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with open(Path(args.out) / "match_measurement.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    return res


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="filter", choices=["filter", "measure"])
    p.add_argument("--idioms", default="data/idioms/ar/idioms_merged_llm_formatted.jsonl")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    p.add_argument("--config", default="arb_Arab")
    p.add_argument("--split", default="train")
    p.add_argument("--text_field", default="text")
    p.add_argument("--out", default="/tmp/ar_tagged")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min_doc_chars", type=int, default=200)
    p.add_argument("--max_docs_per_idiom", type=int, default=10_000)
    p.add_argument("--docs_per_shard", type=int, default=50_000)
    p.add_argument("--log_every", type=int, default=20_000)
    # Tier 2 is opt-IN: its extra hits measured as false positives on real text.
    p.add_argument("--use_stem", action="store_true",
                   help="Also run Tier-2 stem matching (+19.7%% docs, worse precision).")
    p.add_argument("--n_examples", type=int, default=10)
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    return 0 if (cmd_measure(args) if args.command == "measure" else run(args)) else 0


def self_test() -> int:
    idioms = [
        {"output": {"idiom": "أهل مكة أدرى بشعابها",
                    "entities": ["مكة", "شعابها"],
                    "literal_meanings": ["سكان مكة أعرف بطرق جبالها"],
                    "figurative_meanings": ["يضرب لمن هو أعلم بشأن نفسه"],
                    "variety_region": ["Classical Arabic"]}},
        {"output": {"idiom": "اكل عليه الدهر وشرب", "entities": ["الدهر"],
                    "literal_meanings": NAN,
                    "figurative_meanings": ["كناية عن الشيء القديم البالي"],
                    "variety_region": NAN}},
        {"output": {"idiom": "خبر ابيض", "entities": NAN, "literal_meanings": NAN,
                    "figurative_meanings": ["كناية عن الخبر السيئ"],
                    "variety_region": ["Egyptian"]}},
    ]
    m = ArabicIdiomMatcher(idioms, use_stem=True)
    # the 8-char "خبر ابيض" is below MIN_PATTERN_CHARS -> excluded from Tier 0
    assert m.skipped_short == 1

    # Tier 0: vocalized dictionary form found in unvocalized web text
    hits = m.match("يقولون أهل مكة أدرى بشعابها دائما وهذا صحيح")
    assert "أهل مكة أدرى بشعابها" in hits, hits

    # Tier 0 tolerates a proclitic glued to the first word (substring semantics)
    assert m.match("وأهل مكة أدرى بشعابها")

    # Tier 2: interior inflection (عليه -> عليها) that Tier 0 cannot catch
    t0 = ArabicIdiomMatcher(idioms, use_stem=False)
    infl = "المقتضيات القانونيه التي اكل عليها الدهر وشرب"
    assert "اكل عليه الدهر وشرب" not in t0.match(infl)
    assert "اكل عليه الدهر وشرب" in m.match(infl)

    # no false positive on unrelated Arabic
    assert not m.match("هذا نص عربي عادي لا يحتوي على اي تعبير اصطلاحي معروف")

    # knowledge block content + Arabic labels
    kb = knowledge_block(["أهل مكة أدرى بشعابها"], m.by_surface)
    assert KB_HEADER in kb
    assert "المعنى المجازي: يضرب لمن هو أعلم بشأن نفسه" in kb
    assert "المعنى الحرفي: سكان مكة أعرف بطرق جبالها" in kb
    assert "العناصر: مكة، شعابها" in kb
    assert "اللهجة: Classical Arabic" in kb
    # NAN fields must not leak into the block
    kb2 = knowledge_block(["اكل عليه الدهر وشرب"], m.by_surface)
    assert "NAN" not in kb2 and "المعنى الحرفي" not in kb2

    # tagging appends, never destroys the document
    doc = "نص المستند هنا"
    out = tag_document(doc, ["أهل مكة أدرى بشعابها"], m.by_surface)
    assert out.startswith(doc) and KB_HEADER in out
    assert tag_document(doc, [], m.by_surface) == doc      # no hits -> unchanged

    # shard writer round-trips gzip JSONL
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        w = ShardWriter(Path(td), docs_per_shard=2)
        for i in range(3):
            w.write({"text": f"d{i}", "matched_idioms": []})
        w.close()
        shards = sorted(Path(td).glob("tagged_*.json.gz"))
        assert len(shards) == 2, shards            # 2 docs + 1 doc
        rows = []
        for s in shards:
            with gzip.open(s, "rt", encoding="utf-8") as f:
                rows += [json.loads(l) for l in f if l.strip()]
        assert [r["text"] for r in rows] == ["d0", "d1", "d2"]

    print("all filter_and_tag_ar.py self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
