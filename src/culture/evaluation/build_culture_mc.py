#!/usr/bin/env python3
"""Build the new Chinese / Hindi culture MC tasks as ``jsonl:`` eval files.

The paper's Chinese culture column is CCPM alone and its Hindi one is a 100-item
Global-PIQA slice. This script produces the files that fix that, in the pre-built
``load_jsonl_mc`` format (qid / context / options / gold / meta), so no change to
``run_eval`` is needed -- the same mechanism IdiomAtlas-MC and the symbolism probe use.

Templates are IMPORTED from ``culture.evaluation.tasks`` rather than copied, so the
expanded Hindi set stays scoreable against the existing 100-item numbers.

What it builds, and why these and not the others (all measured, see --audit):
  global_piqa_{zh,hi}            the `unsampled_nonparallel` pools the repo ships
                                 alongside the 100-item samples: hi 1,406 (vs 100 in
                                 use) and zh 648 (a Chinese slice we never used).
  global_piqa_{zh,hi}_cultural   the subset with approx_cultural_score == 1 (hi 447,
                                 zh 237). NOTE this flag is an LLM approximation, not
                                 human annotation -- a filter, not a gold label.
  global_piqa_zh_parallel        103 items identical to the hi and ar parallel sets:
                                 a culture-agnostic physics control, and the only
                                 genuinely item-matched cross-language comparison we have.
  parambench_hi_culture          ParamBench restricted to Normal MCQ x culture-bearing
                                 subjects. 100% Devanagari. The compound question types
                                 (Match the List, Assertion and Reason, Sequence) are
                                 excluded: log-likelihood scoring of them is not
                                 meaningful. parambench_hi_other is the non-culture
                                 contrast from the same exam pool.

Rejected after inspecting the data (not built):
  DRISHTIKON    4,286 Hindi rows, but 4,019 (93.8%) textually reference their image.
  CulturalBench 59 China / 46 India items, and the questions are English.
  WenMind       only 917 of 4,875 rows are MCQ; 117 in the relevant domain.
  SANSKRITI     the released file is `Merged_Dataset_english_SANSKRITI.csv` -- English.

Usage:
  python -m culture.evaluation.build_culture_mc --out_dir <mc dir>
  python -m culture.evaluation.build_culture_mc --audit      # print counts, write nothing
  # babel: --eval_dir /data/group_data/r3lit_culture_pretrain/culture/bidir/eval_data \
  #        --parambench <eval_dir>/hi/parambench/ParamBench.parquet
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

from culture.evaluation.tasks import GLOBAL_PIQA_TEMPLATE, MILU_TEMPLATE

csv.field_size_limit(10 ** 9)

STAGE = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/bench_stage"
EVAL = "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval"
LETTERS = ["A", "B", "C", "D", "E", "F"]

# Chinese counterpart of GLOBAL_PIQA_TEMPLATE (which is Hindi); same shape.
GLOBAL_PIQA_TEMPLATE_ZH = "问题：{goal}\n答案："

# ParamBench subjects that test culture rather than academic/social science. Counts
# from the released parquet. Excluded: Education, Sociology, Psychology, Law,
# Economics, Current Affairs, Philosophy, Political Science, Defence and Strategic
# Studies -- graduate exam subjects that are not Indian-culture-specific.
PARAMBENCH_CULTURE = [
    "Indian Culture", "Comparative Study of Religions", "History", "Archaeology",
    "Anthropology", "Drama and theatre", "RABINDRA SANGEET", "Karnatak Music",
    "Percussion Instruments", "Music", "Yoga", "Tribal and Regional Language_Literature",
]


def tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write(path, items, audit):
    print(f"  {'(audit) ' if audit else ''}{os.path.basename(path):34} n={len(items)}")
    if audit or not items:
        return
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def piqa_items(rows, tmpl, tag, n_opt=2):
    """Global-PIQA rows -> MC items. Options are scored as continuations, exactly as
    tasks.load_global_piqa does, so the expanded sets stay comparable to the old ones."""
    out = []
    for i, r in enumerate(rows):
        opts = [r.get(f"solution{j}") for j in range(n_opt)]
        if any(o is None or not str(o).strip() for o in opts):
            continue
        try:
            gold = int(r["label"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= gold < n_opt:
            continue
        out.append({"qid": r.get("example_id") or f"{tag}/{i}",
                    "context": tmpl.format(goal=str(r["prompt"]).strip()),
                    "options": [" " + str(o).strip() for o in opts],
                    "gold": gold,
                    "meta": {"cultural": r.get("approx_cultural_score"),
                             "language": r.get("language")}})
    return out


def parambench_items(rows, tag):
    """ParamBench -> MILU-style letter task (same template and letter options as
    tasks.load_milu), so it reads as a companion to MILU rather than a new format."""
    out = []
    for r in rows:
        opts = [r.get(f"option_{c}") for c in "abcd"]
        if any(o is None or not str(o).strip() for o in opts):
            continue
        ans = str(r.get("correct_answer") or "").strip().upper()
        if ans not in LETTERS[:4]:
            continue
        block = "\n".join(f"{LETTERS[j]}. {str(o).strip()}" for j, o in enumerate(opts))
        out.append({"qid": r.get("unique_question_id") or f"{tag}/{len(out)}",
                    "context": MILU_TEMPLATE.format(question=str(r["question_text"]).strip(),
                                                    options_block=block),
                    "options": [" " + LETTERS[j] for j in range(4)],
                    "gold": LETTERS.index(ans),
                    "meta": {"subject": r.get("subject"),
                             "question_type": r.get("question_type")}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None, help="default: <eval_dir>/mc")
    ap.add_argument("--eval_dir", default=EVAL,
                    help="holds {zh,hi}/global_piqa/*.tsv (default: the H100-server path)")
    ap.add_argument("--parambench", default=os.path.join(STAGE, "parambench", "ParamBench.parquet"),
                    help="path to ParamBench.parquet (default: the H100-server path)")
    ap.add_argument("--audit", action="store_true", help="count only, write nothing")
    a = ap.parse_args()
    a.out_dir = a.out_dir or os.path.join(a.eval_dir, "mc")
    os.makedirs(a.out_dir, exist_ok=True)
    o = lambda n: os.path.join(a.out_dir, n + ".jsonl")

    print("== Global-PIQA ==")
    for lang, code, tmpl in [("zh", "cmn_hans", GLOBAL_PIQA_TEMPLATE_ZH),
                             ("hi", "hin_deva", GLOBAL_PIQA_TEMPLATE)]:
        gp = os.path.join(a.eval_dir, lang, "global_piqa")
        full = os.path.join(gp, f"unsampled_nonparallel_{code}.tsv")
        if not os.path.exists(full):
            print(f"  [skip] {full} missing"); continue
        rows = tsv(full)
        items = piqa_items(rows, tmpl, f"gpiqa_{lang}")
        write(o(f"global_piqa_{lang}"), items, a.audit)
        cult = [it for it in items if str(it["meta"].get("cultural")) == "1"]
        write(o(f"global_piqa_{lang}_cultural"), cult, a.audit)
        par = os.path.join(gp, f"parallel_{code}.tsv")
        if os.path.exists(par):
            write(o(f"global_piqa_{lang}_parallel4"),
                  piqa_items(tsv(par), tmpl, f"gpiqa_{lang}_par", n_opt=4), a.audit)

    print("== ParamBench ==")
    pb = a.parambench
    if not os.path.exists(pb):
        print(f"  [skip] {pb} missing"); return
    import pyarrow.parquet as pq
    rows = pq.read_table(pb).to_pylist()
    dev = sum(1 for r in rows if re.search(r"[ऀ-ॿ]", r.get("question_text") or ""))
    print(f"  source: {len(rows)} rows, {100 * dev / len(rows):.1f}% Devanagari")
    normal = [r for r in rows if r.get("question_type") == "Normal MCQ"]
    cult = [r for r in normal if r.get("subject") in PARAMBENCH_CULTURE]
    other = [r for r in normal if r.get("subject") not in PARAMBENCH_CULTURE]
    write(o("parambench_hi_culture"), parambench_items(cult, "pb_cult"), a.audit)
    write(o("parambench_hi_other"), parambench_items(other, "pb_other"), a.audit)
    c = collections.Counter(r["subject"] for r in cult)
    print("  culture subjects:", dict(c.most_common()))


if __name__ == "__main__":
    sys.exit(main())
