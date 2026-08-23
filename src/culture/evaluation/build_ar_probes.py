"""Build the two Arabic Dimension-1/2 perplexity probes (PPL / BPB).

  ar_fineweb2_heldout.jsonl  IN-DOMAIN. FineWeb-2 `arb_Arab` **official test split**.
  ar_wiki_heldout.jsonl      OUT-OF-DOMAIN. Arabic Wikipedia, title-decontaminated.

Both are ``{"text": ...}`` JSONL, the format ``perplexity.py`` expects.

Why these two, and why this is better than the Chinese pair
-----------------------------------------------------------
``build_zh_probes.py`` pairs a Wikipedia probe with an in-domain slice taken
from the *training* shards — that in-domain probe is contaminated by
construction, which is why it is labelled CONTAMINATED there. Arabic does not
need that compromise: FineWeb-2 ships an official ``test`` split for
``arb_Arab`` that is disjoint from the ``train`` split our CPT corpus is built
from (``filter_and_tag_ar.py --split train``). So the in-domain probe here is a
genuine held-out set, and the two probes answer different questions:

- FineWeb-2 test  -> did CPT improve Arabic LM on the training distribution?
- Wikipedia       -> did it do so without register collapse / forgetting?

Wikipedia decontamination (measured, not assumed)
-------------------------------------------------
Arabic Wikipedia is *inside* FineWeb-2: 95 of 42,080 docs in the test shard come
from ``*.wikipedia.org`` (71 from ``ar.wikipedia.org``), i.e. 0.23 % of docs and
0.32 % of bytes. Extrapolated over ~55 M ``arb_Arab`` docs that is roughly
43 k-93 k Arabic Wikipedia articles, ~4-8 % of the 1.22 M-article dump. So the
Wikipedia probe is NOT automatically clean.

``--exclude_urls`` fixes this exactly: point it at the shards that will actually
be trained on, and every ``ar.wikipedia.org/wiki/<title>`` found there is
excluded from the probe. For a *filtered* CPT arm (~1-3 % of FineWeb-2 docs
survive the idiom filter) residual overlap is ~0.01 % and this is a formality;
for an *unfiltered* arm it is mandatory. Run it either way — it is cheap.

Also dropped, because they inflate BPB variance without being Arabic prose:
disambiguation pages (``(توضيح)`` in the title) and docs under ``--min_chars``.

Usage
-----
    python src/culture/evaluation/build_ar_probes.py \
        --out_dir data/eval/ar \
        --exclude_urls 'data/train_ar/*.json.gz'
"""

import argparse
import glob
import gzip
import io
import json
import logging
import os
import random
import re
import urllib.parse
from typing import Iterator, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_ar_probes")

FINEWEB2_TEST_URL = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/"
                     "resolve/main/data/arb_Arab/test/000_00000.parquet")
WIKI_SHARD_URL = ("https://huggingface.co/datasets/wikimedia/wikipedia/"
                  "resolve/main/20231101.ar/train-{i:05d}-of-00007.parquet")
N_WIKI_SHARDS = 7

DISAMBIGUATION_MARKER = "(توضيح)"          # "(disambiguation)"
WIKI_URL_RE = re.compile(r"https?://ar\.wikipedia\.org/wiki/([^\s\"'<>]+)")


# --------------------------------------------------------------------------- #
def _curl(url: str, dest: str) -> str:
    """Download via curl. HF's python client stalls at 0 bytes on Xet-backed
    files behind this proxy; plain HTTPS to `resolve/main` works."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        logger.info("cached: %s", dest)
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    import subprocess
    logger.info("downloading %s", url)
    subprocess.run(["curl", "-sfL", "--max-time", "1800", url, "-o", dest], check=True)
    return dest


def _iter_shard_lines(pattern: str) -> Iterator[dict]:
    """Yield JSON objects from .jsonl / .jsonl.gz / .json.gz shards matching a glob."""
    for path in sorted(glob.glob(pattern)):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _wiki_title(url: str) -> Optional[str]:
    """``https://ar.wikipedia.org/wiki/%D8%B3%D9%86%D8%A9`` -> ``سنة``."""
    m = WIKI_URL_RE.match(url or "")
    if not m:
        return None
    return urllib.parse.unquote(m.group(1)).replace("_", " ").strip()


def collect_excluded_titles(patterns: List[str]) -> Set[str]:
    """Arabic-Wikipedia titles reachable from the URLs in the training shards."""
    titles: Set[str] = set()
    n_docs = 0
    for pattern in patterns:
        for row in _iter_shard_lines(pattern):
            n_docs += 1
            for key in ("url", "URL", "source_url", "id"):
                t = _wiki_title(str(row.get(key) or ""))
                if t:
                    titles.add(t)
                    break
    logger.info("scanned %d training docs -> %d ar.wikipedia titles to exclude",
                n_docs, len(titles))
    return titles


# --------------------------------------------------------------------------- #
def build_fineweb2(out_path: str, cache_dir: str, n_docs: int,
                   min_chars: int, max_chars: int, seed: int) -> int:
    import pandas as pd
    path = _curl(FINEWEB2_TEST_URL, os.path.join(cache_dir, "fineweb2_arb_test.parquet"))
    df = pd.read_parquet(path, columns=["text", "url"])
    logger.info("FineWeb-2 arb_Arab test shard: %d docs", len(df))
    texts = [t for t in df["text"].tolist()
             if isinstance(t, str) and min_chars <= len(t) <= max_chars]
    rng = random.Random(seed)
    rng.shuffle(texts)
    sel = texts[:n_docs]
    _write(out_path, sel)
    logger.info("ar_fineweb2_heldout (IN-DOMAIN, officially held out): %d docs, %d chars",
                len(sel), sum(len(t) for t in sel))
    return len(sel)


def build_wikipedia(out_path: str, cache_dir: str, n_docs: int, min_chars: int,
                    max_chars: int, seed: int, excluded: Set[str],
                    n_shards: int) -> int:
    import pandas as pd
    kept, n_seen, n_excl, n_disamb = [], 0, 0, 0
    for i in range(n_shards):
        path = _curl(WIKI_SHARD_URL.format(i=i), os.path.join(cache_dir, f"wiki_ar_{i}.parquet"))
        df = pd.read_parquet(path, columns=["title", "text"])
        for title, text in zip(df["title"].tolist(), df["text"].tolist()):
            n_seen += 1
            if not isinstance(text, str) or not (min_chars <= len(text) <= max_chars):
                continue
            title = (title or "").strip()
            if DISAMBIGUATION_MARKER in title:
                n_disamb += 1
                continue
            if title in excluded:
                n_excl += 1
                continue
            kept.append(text)
        logger.info("shard %d: pool=%d (excluded=%d disambig=%d)", i, len(kept), n_excl, n_disamb)
        if len(kept) >= n_docs * 20:      # plenty to sample from; stop early
            break
    rng = random.Random(seed)
    rng.shuffle(kept)
    sel = kept[:n_docs]
    _write(out_path, sel)
    logger.info("ar_wiki_heldout (OOD, decontaminated): %d docs, %d chars "
                "[seen=%d, title-excluded=%d, disambiguation=%d]",
                len(sel), sum(len(t) for t in sel), n_seen, n_excl, n_disamb)
    return len(sel)


def _write(path: str, texts: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="data/eval/ar")
    p.add_argument("--cache_dir", default="data/eval/ar/raw/probes")
    p.add_argument("--n_docs", type=int, default=2000,
                   help="Docs per probe (matches build_zh_probes.py).")
    p.add_argument("--min_chars", type=int, default=1000,
                   help="Wikipedia stubs below this are noise; 1000 also drops "
                        "the (توضيح) leftovers that slip the title filter.")
    p.add_argument("--max_chars", type=int, default=8000)
    p.add_argument("--exclude_urls", action="append", default=[],
                   help="Glob of TRAINING shards (jsonl/.gz) whose ar.wikipedia URLs "
                        "should be excluded from the Wikipedia probe. Repeatable.")
    p.add_argument("--wiki_shards", type=int, default=N_WIKI_SHARDS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", choices=["fineweb2", "wikipedia"], default=None)
    args = p.parse_args()

    excluded: Set[str] = set()
    if args.exclude_urls:
        excluded = collect_excluded_titles(args.exclude_urls)
    else:
        logger.warning("No --exclude_urls given: the Wikipedia probe is NOT "
                       "decontaminated against the training shards. Arabic Wikipedia "
                       "is ~0.23%% of FineWeb-2 docs, so pass the training globs if "
                       "you are evaluating an unfiltered CPT arm.")

    if args.only != "wikipedia":
        build_fineweb2(os.path.join(args.out_dir, "ar_fineweb2_heldout.jsonl"),
                       args.cache_dir, args.n_docs, args.min_chars, args.max_chars, args.seed)
    if args.only != "fineweb2":
        build_wikipedia(os.path.join(args.out_dir, "ar_wiki_heldout.jsonl"),
                        args.cache_dir, args.n_docs, args.min_chars, args.max_chars,
                        args.seed, excluded, args.wiki_shards)


if __name__ == "__main__":
    main()
