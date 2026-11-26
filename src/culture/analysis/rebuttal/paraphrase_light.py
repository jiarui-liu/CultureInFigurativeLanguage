#!/usr/bin/env python3
"""IdiomAtlas-MC seen split with every option lightly reworded.

A second rewording control with a milder rewrite than paraphrase_mc.py: each option keeps its
dictionary register and fixed gloss frame, and only part of the content words and the clause order
change, so that no option is a verbatim copy of a training meaning block. Every option (gold and
distractors) is rewritten under the same rules; gold index and option order are unchanged.

  build : merge rewritten option strings (one JSONL per chunk with {"id", "para"}) into
          <mc_para2>/idiomatlas_mc_{L}_seen_para2.jsonl, copy the original seen file next to it,
          and report chrF between rewrites and originals
  stats : original vs reworded contrasts from per-item records (paired bootstrap)
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from culture.analysis.rebuttal.paraphrase_mc import B, MC, RUN, EVAL, boot, chrf, recs

PARA2 = B / "eval_data/mc_para2"
EVALP2 = B / "eval9b_para2"


def build(a):
    PARA2.mkdir(parents=True, exist_ok=True)
    src = Path(a.src)
    report = {}
    for lang in a.langs:
        ids = json.load(open(src / f"{lang}_ids.json", encoding="utf-8"))
        para = {}
        for fn in sorted((src / "out").glob(f"{lang}_*.jsonl")):
            for line in open(fn, encoding="utf-8"):
                r = json.loads(line)
                para[r["id"]] = r["para"].strip()
        items = [json.loads(l) for l in open(MC / f"idiomatlas_mc_{lang}_seen.jsonl", encoding="utf-8")]
        missing, scores, gold_scores, same = 0, [], [], 0
        with open(PARA2 / f"idiomatlas_mc_{lang}_seen_para2.jsonl", "w", encoding="utf-8") as f:
            for x in items:
                opts, cs = [], []
                for o in x["options"]:
                    p = para.get(ids.get(o))
                    if not p:
                        missing += 1
                        p = o.strip()
                    same += p == o.strip()
                    c = chrf(p, o.strip())
                    opts.append(" " + p)
                    cs.append(round(c, 1))
                scores += cs
                gold_scores.append(cs[x["gold"]])
                y = dict(x, options=opts, meta={**x["meta"], "orig_options": x["options"], "para2_chrf": cs})
                f.write(json.dumps(y, ensure_ascii=False) + "\n")
        shutil.copy(MC / f"idiomatlas_mc_{lang}_seen.jsonl", PARA2 / f"idiomatlas_mc_{lang}_seen.jsonl")
        report[lang] = {"items": len(items), "unique_options": len(ids), "rewritten": len(para),
                        "missing": missing, "identical": same,
                        "chrF_median_all": round(float(np.median(scores)), 1),
                        "chrF_median_gold": round(float(np.median(gold_scores)), 1),
                        "share_gold_chrF_gt_80": round(float(np.mean(np.array(gold_scores) > 80)), 3)}
        print(lang, json.dumps(report[lang]), flush=True)
    json.dump(report, open(PARA2 / "build_report.json", "w"), indent=1)


def stats(a):
    rng = np.random.default_rng(11)
    out = {}
    for lang in a.langs:
        t = f"idiomatlas_mc_{lang}_seen"
        O = {r: recs(EVAL, lang, r, t) for r in RUN}
        P = {r: recs(EVALP2, lang, r, t + "_para2") for r in RUN}
        O = {r: v for r, v in O.items() if v}
        P = {r: v for r, v in P.items() if v}
        if not {"unfiltered", "cpt"} <= set(P):
            continue
        q = sorted(set.intersection(*[set(v) for v in list(O.values()) + list(P.values())]))
        row = {"n": len(q),
               "acc_orig": {RUN[r]: round(100 * np.mean([O[r][i] for i in q]), 1) for r in O},
               "acc_para2": {RUN[r]: round(100 * np.mean([P[r][i] for i in q]), 1) for r in P}}
        for x, y in [("cpt", "unfiltered"), ("cpt", "untagged"), ("untagged", "unfiltered"), ("cpt", "base")]:
            if x in P and y in P and x in O and y in O:
                do = np.array([O[x][i] - O[y][i] for i in q], float)
                dp = np.array([P[x][i] - P[y][i] for i in q], float)
                row[f"{RUN[x]} - {RUN[y]}"] = {"orig": boot(do, rng), "para2": boot(dp, rng)}
        out[f"{lang}_seen"] = row
        print(lang, json.dumps(row, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats"])
    ap.add_argument("--src", help="directory with <lang>_ids.json and out/<lang>_*.jsonl")
    ap.add_argument("--langs", nargs="+", default=["ar", "hi", "zh"])
    ap.add_argument("--out", default="docs/paper_stats/rebuttal/paraphrase_light.json")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else stats(a)


if __name__ == "__main__":
    main()
