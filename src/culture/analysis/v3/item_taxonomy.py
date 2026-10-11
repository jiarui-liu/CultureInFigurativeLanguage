#!/usr/bin/env python3
"""Label every benchmark item with the six-way culture-layer taxonomy, keyed by qid.

§5.4 of the paper classifies a 400-item *sample* per source and reports source-level
distributions. That measures what each benchmark is made of, but not the claim the
distributions are used to support -- that each kind of training text teaches the layer it
states. The decisive test is *within* a benchmark: if the explanation is right, the gain of
Idiom-CPT over Random-CPT should be larger on the symbolic/evaluative items of a culture
benchmark than on its factual ones, and the gain of Culture-CPT larger on the factual and
material-practice items. That test needs a label per `qid`, which the §5.4 cache (keyed on
the prompt hash, never storing the qid) cannot provide.

This script labels whole benchmarks and writes `{task: {qid: category}}`.

    PYTHONPATH=src:src/culture/analysis/v2 python item_taxonomy.py [--cap 4000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import local_llm  # noqa: E402  (src/culture/analysis/v2 on PYTHONPATH)

ITEMS = os.environ.get(
    "CULTURE_ITEMS_DIR",
    "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/items",
)
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3",
)

# Verbatim from culture.analysis.v2.culture_layer_taxonomy so the per-item labels are
# comparable with the source-level distributions already in the paper.
CATEGORIES = {
    "symbolic_evaluative": (
        "What something stands for, connotes, or symbolizes; praise, blame, or moral "
        "judgement of a person, trait, or behaviour; whether a quality is admired or despised."
    ),
    "social_norm_relation": (
        "How people are expected to behave toward one another: obligations, etiquette, "
        "hierarchy, kinship and social roles, politeness formulas."
    ),
    "material_practice": (
        "Concrete customs and daily life: food, dress, festivals, rituals, ceremonies, "
        "household routines, games, occupations as practised."
    ),
    "factual_knowledge": (
        "Facts about named things: history, geography, institutions, religious doctrine, "
        "named people, books, films, songs, places."
    ),
    "language_form": (
        "Properties of language itself: spelling, grammar, dialect vocabulary, register, "
        "which wording is correct or idiomatic."
    ),
    "generic_pragmatic": (
        "General life advice, physical or psychological common sense that is not specific "
        "to any one culture."
    ),
}
CAT_KEYS = list(CATEGORIES)

PROMPT = """You are analysing what kind of cultural knowledge a test item requires.

Categories:
{cats}

Item (benchmark question):
{text}

Which single category best describes the knowledge this item encodes or tests? Consider what a
person would have to know to answer or to understand it. Reply with exactly one category name
from the list and nothing else."""

# Everything the 9B/2B grid is scored on, minus the tasks whose items are not questions
# about culture at all (English retention, perplexity probes).
TASKS = [
    # culture
    "arabculture", "alyah", "dzirieval", "arabic_cultural_qa", "global_piqa_ar",
    "ccpm", "global_piqa_zh", "global_piqa_hi", "global_piqa",
    # regional knowledge
    "arabicmmlu", "milu", "cmmlu", "parambench_hi_culture", "parambench_hi_other",
    # idiom / figurative
    "kinayat_meaning", "ar_figurative", "mabl", "chengyu_bench",
    "idiomatlas_mc_ar_seen", "idiomatlas_mc_ar_unseen",
    "idiomatlas_mc_hi_seen", "idiomatlas_mc_hi_unseen",
    "idiomatlas_mc_zh_seen", "idiomatlas_mc_zh_unseen",
    # symbolism
    "symbolism_v2_ar_letter", "symbolism_v2_hi_letter", "symbolism_v2_zh_letter",
]


def _clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def render(o):
    opts = [_clean(x) for x in (o.get("options") or []) if _clean(x)][:6]
    body = "\n".join(f"  - {x}" for x in opts)
    return f"Question: {_clean(o['stem'])[:900]}\nOptions:\n{body}"[:1600]


def parse_cat(resp):
    low = (resp or "").strip().lower()
    for k in CAT_KEYS:
        if k in low:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=4000,
                    help="max items per task (random, seed 0); 0 = all")
    ap.add_argument("--subsample", type=int, default=0,
                    help="after --cap, keep only the first N items of each task; used for the "
                         "second annotator so it relabels a subset of exactly the same items")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--model", default="primary", choices=["primary", "second", "third", "fourth"])
    ap.add_argument("--out", default="item_taxonomy.json")
    args = ap.parse_args()

    model = {"primary": local_llm.PRIMARY, "second": local_llm.SECOND,
             "third": local_llm.THIRD, "fourth": local_llm.FOURTH}[args.model]
    want = [t for t in TASKS if not args.tasks or t in args.tasks.split(",")]

    cats_txt = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    prompts, meta = [], []
    rng = random.Random(0)
    for t in want:
        p = f"{ITEMS}/{t}.jsonl"
        if not os.path.exists(p):
            print(f"[skip] {t}: no dump")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if args.cap and len(rows) > args.cap:
            rows = rng.sample(rows, args.cap)
        if args.subsample:
            rows = rows[: args.subsample]
        print(f"[load] {t}: {len(rows)}")
        for o in rows:
            prompts.append(PROMPT.format(cats=cats_txt, text=render(o)))
            meta.append((t, o["qid"], o["lang"]))

    print(f"[llm] {len(prompts)} calls with {local_llm.NAMES.get(model, model)}")
    resp = local_llm.generate(prompts, f"item_taxonomy_{args.model}", model=model, max_tokens=16)

    labels = defaultdict(dict)
    dist = defaultdict(Counter)
    unparsed = Counter()
    for (t, qid, lang), r in zip(meta, resp):
        c = parse_cat(r)
        if c is None:
            unparsed[t] += 1
            continue
        labels[t][qid] = c
        dist[t][c] += 1

    out = {
        "model": local_llm.NAMES.get(model, model),
        "cap": args.cap,
        "labels": labels,
        "distribution": {t: dict(c) for t, c in dist.items()},
        "n": {t: len(v) for t, v in labels.items()},
        "unparsed": dict(unparsed),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[write] {path}")
    for t in want:
        if t in dist:
            tot = sum(dist[t].values())
            top = ", ".join(f"{k} {v / tot:.0%}" for k, v in dist[t].most_common(3))
            print(f"  {t:28s} n={tot:6d}  {top}")


if __name__ == "__main__":
    main()
