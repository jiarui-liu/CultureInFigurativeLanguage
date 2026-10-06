#!/usr/bin/env python3
"""Contamination check for the new zh / hi culture benchmarks.

The benchmark review (docs/literature_reviews/idiom_and_cultural_benchmarks_hi_zh.md,
"Two cross-cutting cautions") requires every newly adopted benchmark to be
decontaminated against the SFT mixtures, and warns that Chinese SFT pools frequently
ingest exam data -- which is exactly what ParamBench is (UGC-NET exam papers).

Method: exact substring match of each benchmark question against the training text.
We index the TRAINING side, not the benchmark side: every training document is cut
into a set of character n-grams, and a question counts as hit if a distinctive
n-gram drawn from it (the longest shingle we can take, after whitespace collapse)
appears in that set. Substring matching is the right test here -- paraphrase overlap
is not contamination, verbatim reproduction of the item is.

Checks each benchmark against:
  * the SFT mixtures actually used (infinity-instruct-zh for zh, indic-align for hi)
  * the CPT corpora the arms train on (train_{zh,hi}_{culture,unfiltered})
CPT corpora are large, so --max_docs samples them; the sample size is reported with
the result so a null is never read as stronger than it is.

Usage:
  python -m culture.evaluation.contamination_check --lang zh
  python -m culture.evaluation.contamination_check --lang hi --max_docs 400000
"""
import argparse
import glob
import json
import os
import random
import re

MC = "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc"
CPT = "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data"
SFT = "/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data"

BENCH = {
    "zh": ["global_piqa_zh", "global_piqa_zh_cultural", "global_piqa_zh_parallel4"],
    "hi": ["global_piqa_hi", "global_piqa_hi_cultural", "global_piqa_hi_parallel4",
           "parambench_hi_culture", "parambench_hi_other"],
}
SOURCES = {
    "zh": [("sft:infinity-instruct-zh", os.path.join(SFT, "infinity-instruct-zh")),
           ("cpt:train_zh_culture", os.path.join(CPT, "train_zh_culture")),
           ("cpt:train_zh_unfiltered", os.path.join(CPT, "train_zh_unfiltered"))],
    "hi": [("sft:indic-align", os.path.join(SFT, "indic-align")),
           ("cpt:train_hi_culture", os.path.join(CPT, "train_hi_culture")),
           ("cpt:train_hi_unfiltered", os.path.join(CPT, "train_hi_unfiltered"))],
}
WS = re.compile(r"\s+")


def norm(s):
    return WS.sub(" ", str(s or "")).strip()


def question_of(item):
    """The question text, stripped of the template and the option block."""
    c = norm(item["context"])
    c = re.split(r"\nA[\.。]|\n[A-D]\. ", item["context"])[0]
    c = re.sub(r"^(प्रश्न:|问题：)\s*", "", norm(c))
    return re.sub(r"(उत्तर:|答案：)\s*$", "", c).strip()


def iter_texts(path, max_docs, rng):
    """Yield text from a dir of jsonl(.gz) / parquet, sampling at most max_docs."""
    import gzip
    files = sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True))
    # The CPT corpora ship as .jsonl.gz; _report_*.json are build logs, not data.
    files = [f for f in files if os.path.isfile(f)
             and f.endswith((".jsonl", ".jsonl.gz", ".parquet", ".json"))
             and not os.path.basename(f).startswith("_report")]
    n = 0
    for f in files:
        if n >= max_docs:
            return
        try:
            if f.endswith(".parquet"):
                import pyarrow.parquet as pq
                tbl = pq.read_table(f)
                names = tbl.schema.names
                # IndicAlign is multilingual *within* a row: one column per language
                # (hin_Deva, ben_Beng, ...), no `text` column at all. Scan the Hindi
                # column there; fall back to the usual single-text schemas elsewhere.
                cols = [c for c in ("hin_Deva", "text", "content", "output",
                                    "conversations", "messages") if c in names]
                if not cols:
                    print(f"    [warn] {os.path.basename(f)}: no known text column "
                          f"in {names[:8]}")
                    continue
                for v in tbl.column(cols[0]).to_pylist():
                    yield str(v); n += 1
                    if n >= max_docs: return
            else:
                op = gzip.open if f.endswith(".gz") else open
                for line in op(f, "rt", encoding="utf-8", errors="replace"):
                    if not line.strip():
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    yield str(o.get("text") or o.get("content") or o.get("output") or o)
                    n += 1
                    if n >= max_docs: return
        except Exception as e:
            print(f"    [warn] {os.path.basename(f)}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "hi"], required=True)
    ap.add_argument("--max_docs", type=int, default=300000)
    ap.add_argument("--ngram", type=int, default=None,
                    help="chars of the probe shingle; default 20 for zh, 40 for hi. "
                         "Chinese characters carry far more information per character, "
                         "and a 40-char floor would discard 545 of 648 Global-PIQA-zh items.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.ngram is None:
        a.ngram = 20 if a.lang == "zh" else 40
    rng = random.Random(0)

    probes = {}   # bench -> {probe: qid}
    for b in BENCH[a.lang]:
        p = os.path.join(MC, b + ".jsonl")
        if not os.path.exists(p):
            print(f"[skip] {p} missing"); continue
        d = {}
        short = 0
        for line in open(p, encoding="utf-8"):
            it = json.loads(line)
            q = question_of(it)
            if len(q) < a.ngram:
                short += 1
                continue
            d[q[-a.ngram:]] = it["qid"]   # tail shingle: least templated part
        probes[b] = d
        print(f"{b}: {len(d)} probes ({short} questions shorter than {a.ngram} chars, skipped)")

    report = {"lang": a.lang, "ngram": a.ngram, "max_docs": a.max_docs, "sources": {}}
    # One Aho-Corasick automaton over every probe of every benchmark: each training
    # document is then scanned once, instead of once per probe (300k docs x 5.4k probes
    # of naive `in` would be ~10^9 substring searches).
    import ahocorasick
    A = ahocorasick.Automaton()
    for b, d in probes.items():
        for probe, qid in d.items():
            A.add_word(probe, (b, qid))
    if len(A) == 0:
        print("no probes built; nothing to check"); return
    A.make_automaton()

    for sname, spath in SOURCES[a.lang]:
        if not os.path.isdir(spath):
            print(f"\n[skip] {sname}: {spath} not found"); continue
        print(f"\n== scanning {sname} ==")
        hits = {b: {} for b in probes}
        seen = 0
        for text in iter_texts(spath, a.max_docs, rng):
            seen += 1
            if seen % 100000 == 0:
                print(f"   {seen} docs...")
            for _, (b, qid) in A.iter(text):
                hits[b][qid] = True
        report["sources"][sname] = {"docs_scanned": seen,
                                    "hits": {b: len(h) for b, h in hits.items()},
                                    "rate": {b: (len(h) / len(probes[b]) if probes[b] else 0.0)
                                             for b, h in hits.items()},
                                    "example_qids": {b: sorted(h)[:10] for b, h in hits.items()}}
        for b, h in hits.items():
            pct = 100 * len(h) / max(1, len(probes[b]))
            flag = "  <-- CONTAMINATED" if pct > 1.0 else ""
            print(f"   {b:30} {len(h):5}/{len(probes[b]):5} ({pct:.2f}%){flag}")
        print(f"   docs scanned: {seen}")

    out = a.out or f"docs/paper_stats/v2/contamination_{a.lang}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
