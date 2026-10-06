"""What layer of culture do idioms encode, and what layer do culture benchmarks test?

The paper argues that transfer between idioms and culture benchmarks is narrow because the two
probe different things: idioms carry a *symbolic and evaluative* layer (what an entity stands
for, who is admired or despised), while culture benchmarks mostly test *facts and practices*
(what is eaten at a festival, which film a line comes from). That claim is currently asserted,
not measured. This script measures it.

Every item -- an idiom's figurative meaning, or a culture-benchmark question -- is classified by
an LLM into one shared taxonomy, with a second model from a different family on a subsample for
agreement. The output is a source x category distribution plus agreement stats.

    PYTHONPATH=src:src/culture/analysis/v2 python culture_layer_taxonomy.py [--per_source N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import common
import local_llm

EVAL = os.environ.get(
    "CULTURE_EVAL_DIR",
    "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval",
)
DATA_ROOT = os.environ.get(
    "CULTURE_DATA_ROOT",
    "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data",
)
NOTES_DIR = os.environ.get(
    "CULTURE_NOTES_DIR",
    "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/full/ar/notes",
)

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

PROMPT = """You are analysing what kind of cultural knowledge a test item or an idiom requires.

Categories:
{cats}

Item ({kind}):
{text}

Which single category best describes the knowledge this item encodes or tests? Consider what a
person would have to know to answer or to understand it. Reply with exactly one category name
from the list and nothing else."""


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


# --------------------------------------------------------------------------- sources


def src_idioms(lang: str, n: int, rng: random.Random):
    rows = [r for r in common.load_kb(lang) if r["fig"]]
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        out.append(f"Idiom: {r['idiom']}\nFigurative meaning: {_clean(r['fig'][0])}")
    return out


def _mcq(stem: str, opts) -> str:
    opts = [_clean(o) for o in opts if _clean(o)]
    body = "\n".join(f"  - {o}" for o in opts[:6])
    return f"Question: {_clean(stem)}\nOptions:\n{body}"


def src_arabculture(n, rng):
    import glob

    import pandas as pd

    out = []
    for p in sorted(glob.glob(f"{EVAL}/ar/raw/arabculture/*.parquet")):
        d = pd.read_parquet(p)
        for _, r in d.iterrows():
            opts = list(r["options"]["text"])
            out.append(_mcq(r["first_statement"], opts))
    rng.shuffle(out)
    return out[:n]


def src_alyah(n, rng):
    import pandas as pd

    d = pd.read_parquet(f"{EVAL}/ar/raw/alyah.parquet")
    out = [
        _mcq(r["query"], [r["option_1"], r["option_2"], r["option_3"], r["option_4"]])
        for _, r in d.iterrows()
    ]
    rng.shuffle(out)
    return out[:n]


def src_dzirieval(n, rng):
    out = []
    for line in open(f"{EVAL}/ar/raw/dzirieval.jsonl", encoding="utf-8"):
        o = json.loads(line)
        out.append(_mcq(o["question"], list((o.get("choices") or {}).values())))
    rng.shuffle(out)
    return out[:n]


def src_acqa(n, rng):
    out = []
    for line in open(f"{EVAL}/ar/raw/acqa_test.jsonl", encoding="utf-8"):
        o = json.loads(line)
        if o.get("dialect") == "english":  # same items, translated
            continue
        out.append(_mcq(o["question"], [o.get(k) for k in "ABCD"]))
    rng.shuffle(out)
    return out[:n]


def src_milu(n, rng):
    out = []
    for line in open(f"{EVAL}/hi/milu_hi_test.jsonl", encoding="utf-8"):
        o = json.loads(line)
        out.append(_mcq(o["question"], [o.get(f"option{i}") for i in range(1, 5)]))
    rng.shuffle(out)
    return out[:n]


def src_ccpm(n, rng):
    out = []
    for line in open(f"{EVAL}/zh/CCPM/test_public.jsonl", encoding="utf-8"):
        o = json.loads(line)
        out.append(_mcq(o["translation"], o.get("choices") or []))
    rng.shuffle(out)
    return out[:n]


def src_cultural_notes(n, rng):
    """The LLM-written notes appended in the Culture+notes arm (Arabic).

    Included because the arm did not teach idioms: if the notes are facts and practices
    rather than symbolic content, that is the direct explanation.
    """
    import glob
    import gzip

    out = []
    for p in sorted(glob.glob(f"{NOTES_DIR}/notes_*.jsonl.gz"))[:2]:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 20000:
                    break
                for note in (json.loads(line).get("notes") or []):
                    note = _clean(note).lstrip("- ").strip()
                    if len(note) > 25:
                        out.append(f"Cultural note: {note}")
    rng.shuffle(out)
    return out[:n]


def _gpiqa_tsv(path, out):
    import csv as _csv
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            stem = r.get("prompt")
            opts = [r.get("solution0"), r.get("solution1")]
            if stem and any(opts):
                out.append(_mcq(stem, opts))


def src_gpiqa_hi(n, rng):
    out = []
    for p in (f"{EVAL}/hi/_gpiqa/data/nonparallel_hin_deva.tsv",
              f"{EVAL}/hi/global_piqa_hi.tsv"):
        if os.path.exists(p):
            _gpiqa_tsv(p, out)
            break
    rng.shuffle(out)
    return out[:n]


def src_gpiqa_ar(n, rng):
    import glob
    out = []
    for p in sorted(glob.glob(f"{EVAL}/ar/raw/global_piqa/nonparallel_*.tsv")):
        _gpiqa_tsv(p, out)
    rng.shuffle(out)
    return out[:n]


CULTURE_CORPUS = {
    "ar": f"{DATA_ROOT}/train_ar_culture",
    "hi": f"{DATA_ROOT}/train_hi_culture",
    "zh": f"{DATA_ROOT}/train_zh_culture",
}


def src_culture_corpus(lang, n, rng):
    """Documents of the culture-CPT corpus itself -- the data the Culture arm trains on.

    Unlike the other sources these are whole web documents, so we classify a leading
    excerpt and ask what the text mainly conveys.
    """
    import glob

    files = sorted(glob.glob(f"{CULTURE_CORPUS[lang]}/*.jsonl"))[:4]
    out = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 4000:
                    break
                try:
                    t = _clean(json.loads(line).get("text", ""))
                except Exception:
                    continue
                if len(t) > 300:
                    out.append(f"Document excerpt: {t[:1200]}")
    rng.shuffle(out)
    return out[:n]


SOURCES = {
    "Idioms (en)": ("idiom", lambda n, r: src_idioms("en", n, r)),
    "Idioms (zh)": ("idiom", lambda n, r: src_idioms("zh", n, r)),
    "Idioms (hi)": ("idiom", lambda n, r: src_idioms("hi", n, r)),
    "Idioms (ar)": ("idiom", lambda n, r: src_idioms("ar", n, r)),
    "ArabCulture": ("benchmark question", src_arabculture),
    "Alyah": ("benchmark question", src_alyah),
    "DziriEval": ("benchmark question", src_dzirieval),
    "ArabicCulturalQA": ("benchmark question", src_acqa),
    "Global-PIQA (hi)": ("benchmark question", src_gpiqa_hi),
    "Global-PIQA (ar)": ("benchmark question", src_gpiqa_ar),
    "MILU (hi)": ("benchmark question", src_milu),
    "CCPM (zh)": ("benchmark question", src_ccpm),
    "Cultural notes (ar)": ("cultural note", src_cultural_notes),
    "Culture corpus (ar)": ("web document", lambda n, r: src_culture_corpus("ar", n, r)),
    "Culture corpus (hi)": ("web document", lambda n, r: src_culture_corpus("hi", n, r)),
    "Culture corpus (zh)": ("web document", lambda n, r: src_culture_corpus("zh", n, r)),
}


def parse_cat(resp: str):
    low = (resp or "").strip().lower()
    for k in CAT_KEYS:
        if k in low:
            return k
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_source", type=int, default=400)
    ap.add_argument("--agree_n", type=int, default=120)
    ap.add_argument("--out", default="culture_layer_taxonomy.json")
    # vLLM cannot initialise a second model in one process, so the primary and
    # second-annotator passes run as separate invocations; the on-disk cache makes
    # the already-computed pass free on the second run.
    ap.add_argument("--skip_second", action="store_true")
    # Which model plays the second annotator. "third" is aya-expanse-8b (8B, different
    # family); "fourth" is gemma-4-26B-A4B-it, which is both a different family and
    # comparable in size to the primary, so disagreement cannot be blamed on capacity.
    ap.add_argument("--annotator", choices=["third", "fourth"], default="third")
    args = ap.parse_args()

    rng = random.Random(0)
    cats_txt = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())

    texts, meta = [], []
    for name, (kind, fn) in SOURCES.items():
        try:
            items = fn(args.per_source, rng)
        except Exception as e:  # a missing benchmark file must not kill the whole run
            print(f"[skip] {name}: {e}")
            continue
        print(f"[load] {name}: {len(items)}")
        for t in items:
            texts.append(PROMPT.format(cats=cats_txt, kind=kind, text=t[:1600]))
            meta.append({"source": name, "kind": kind})

    print(f"[llm] {len(texts)} primary calls")
    prim = local_llm.generate(texts, "culture_layer_primary", max_tokens=16)

    dist = defaultdict(Counter)
    unparsed = Counter()
    for m, r in zip(meta, prim):
        c = parse_cat(r)
        if c is None:
            unparsed[m["source"]] += 1
            continue
        dist[m["source"]][c] += 1
        m["cat"] = c

    # Second annotator, different family, on a stratified subsample.
    by_src = defaultdict(list)
    for i, m in enumerate(meta):
        if "cat" in m:
            by_src[m["source"]].append(i)
    agree, pairs_ab = [], []
    per_source_second = defaultdict(list)
    if not args.skip_second:
        sub = []
        per = max(1, args.agree_n // max(1, len(by_src)))
        for s, idxs in by_src.items():
            sub.extend(rng.sample(idxs, min(per, len(idxs))))
        ann_model = (local_llm.THIRD if args.annotator == "third"
                     else local_llm.FOURTH)
        ann_tag = ("culture_layer_second" if args.annotator == "third"
                   else "culture_layer_second_gemma")
        print(f"[llm] {len(sub)} agreement calls "
              f"({local_llm.NAMES.get(ann_model, ann_model)})")
        sec = local_llm.generate(
            [texts[i] for i in sub], ann_tag, model=ann_model, max_tokens=16,
        )
        pairs_ab = [(meta[i]["cat"], parse_cat(r))
                    for i, r in zip(sub, sec) if parse_cat(r)]
        agree = [a == b for a, b in pairs_ab]
        # Per-source: does the second family reproduce the same symbolic share?
        for i, r in zip(sub, sec):
            c2 = parse_cat(r)
            if c2:
                per_source_second[meta[i]["source"]].append((meta[i]["cat"], c2))

    # Collapse to the paper's two-way contrast.
    SYMBOLIC = {"symbolic_evaluative", "social_norm_relation"}
    summary = {}
    for s, c in dist.items():
        tot = sum(c.values())
        summary[s] = {
            "n": tot,
            "shares": {k: round(c.get(k, 0) / tot, 4) for k in CAT_KEYS},
            "counts": {k: c.get(k, 0) for k in CAT_KEYS},
            "symbolic_or_normative": round(
                sum(c.get(k, 0) for k in SYMBOLIC) / tot, 4
            ),
        }

    # Cohen's kappa: raw agreement over six imbalanced classes is hard to read.
    kappa = None
    if not args.skip_second and len(pairs_ab) > 5:
        cats_seen = sorted({c for pr in pairs_ab for c in pr})
        n = len(pairs_ab)
        po = sum(a == b for a, b in pairs_ab) / n
        pa = {c: sum(1 for a, _ in pairs_ab if a == c) / n for c in cats_seen}
        pb = {c: sum(1 for _, b in pairs_ab if b == c) / n for c in cats_seen}
        pe = sum(pa[c] * pb[c] for c in cats_seen)
        kappa = round((po - pe) / (1 - pe), 4) if pe < 1 else None

    out = {
        "method": __doc__,
        "categories": CATEGORIES,
        "per_source": args.per_source,
        "primary_model": local_llm.NAMES.get(local_llm.PRIMARY, local_llm.PRIMARY),
        "second_model": local_llm.NAMES.get(
            local_llm.THIRD if args.annotator == "third" else local_llm.FOURTH,
            "second"),
        "agreement_n": len(agree),
        "agreement_raw": round(sum(agree) / len(agree), 4) if agree else None,
        "agreement_cohen_kappa": kappa,
        "agreement_by_source": {
            s2: {
                "n": len(v),
                "raw_agreement": round(sum(a == b for a, b in v) / len(v), 4),
                "symbolic_share_primary": round(
                    sum(1 for a, _ in v if a == "symbolic_evaluative") / len(v), 4),
                "symbolic_share_second": round(
                    sum(1 for _, b in v if b == "symbolic_evaluative") / len(v), 4),
            }
            for s2, v in sorted(per_source_second.items()) if v
        },
        "unparsed": dict(unparsed),
        "by_source": summary,
    }
    # Keepeach annotator separate: a later run with a different model must not
    # overwrite the earlier one.
    prev_path = os.path.join(common.OUT, args.out)
    prev = {}
    if os.path.exists(prev_path):
        try:
            prev = json.load(open(prev_path, encoding="utf-8"))
        except Exception:
            prev = {}
    models = dict(prev.get("agreement_models") or {})
    if not args.skip_second and out.get("agreement_n"):
        models[out["second_model"]] = {
            "n": out["agreement_n"],
            "raw_agreement": out["agreement_raw"],
            "cohen_kappa": out["agreement_cohen_kappa"],
            "by_source": out["agreement_by_source"],
        }
    out["agreement_models"] = models
    if args.skip_second and prev:
        # a primary-only rerun should not drop agreement already measured
        for k in ("agreement_n", "agreement_raw", "agreement_cohen_kappa",
                  "agreement_by_source", "second_model"):
            if prev.get(k) and not out.get(k):
                out[k] = prev[k]
    p = common.dump(out, args.out)
    print("wrote", p)
    for s, v in summary.items():
        print(f"  {s:22s} n={v['n']:4d}  symbolic/normative={v['symbolic_or_normative']:.2f}")


if __name__ == "__main__":
    main()
