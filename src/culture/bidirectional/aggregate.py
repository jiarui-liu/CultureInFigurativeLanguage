#!/usr/bin/env python3
"""Aggregate evaluation outputs of the bidirectional study.

Input layout: <eval_root>/<lang>/<arm>[_s<seed>]/<task>.json (harness output with
per-item records). For every language, task and arm, computes accuracy and the
paired contrast against Random (seed-matched when seeds exist; otherwise seed 42),
with a paired-bootstrap 95% CI and an exact McNemar p-value, then Holm-corrects
within benchmark groups. Also builds the transfer matrix: mean gain over Random per
(training condition x benchmark group), averaged over the tasks of the group and the
languages, with a bootstrap CI that resamples items within every task.

Usage:
  python -m culture.bidirectional.aggregate --eval_root $B/eval2b --out docs/paper_stats/v2/bidir_2b.json
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

from culture.evaluation.compute_cis import load_run, paired

GROUP = {
    "kinayat_meaning": "idiom_meaning", "chengyu_bench": "idiom_meaning", "chengyu_bench_app": "idiom_meaning",
    "idiomatlas_mc_ar_seen": "idiom_meaning", "idiomatlas_mc_hi_seen": "idiom_meaning",
    "idiomatlas_mc_zh_seen": "idiom_meaning",
    "idiomatlas_mc_ar_unseen": "idiom_unseen", "idiomatlas_mc_hi_unseen": "idiom_unseen",
    "idiomatlas_mc_zh_unseen": "idiom_unseen",
    "ar_figurative": "figurative", "mabl": "figurative",
    "kinayat_cloze": "cloze", "chid": "cloze",
    "symbolism_v2_ar_letter": "symbolism", "symbolism_v2_hi_letter": "symbolism",
    "symbolism_v2_zh_letter": "symbolism",
    "alyah": "culture", "dzirieval": "culture", "arabculture": "culture", "arabic_cultural_qa": "culture",
    "global_piqa_ar": "culture", "global_piqa": "culture", "ccpm": "culture",
    "arabicmmlu": "regional", "milu": "regional", "cmmlu": "regional",
    "global_piqa_ar_parallel": "control",
}
ARMS = ["idiom_tagged", "idiom_untagged", "culture", "culture_notes"]
# Two-option tasks with fixed labels: accuracy mostly tracks each model's label prior, so
# correctness is recomputed after median-centering the option log-probability difference.
CALIBRATE = {"chengyu_bench"}


def calibrated(run_dir, task):
    recs = json.load(open(os.path.join(run_dir, task + ".json")))["records"]
    lp = [json.loads(r["logprobs_norm"]) if isinstance(r["logprobs_norm"], str) else r["logprobs_norm"]
          for r in recs]
    if any(len(x) != 2 for x in lp):
        return None
    d = np.array([x[1] - x[0] for x in lp])
    pred = (d - np.median(d) > 0).astype(int)
    return {r["qid"]: bool(p == int(r["gold"])) for r, p in zip(recs, pred)}


def holm(ps):
    m = len(ps)
    order = np.argsort(ps)
    adj, run = np.empty(m), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--prefix", default="", help='run-name prefix of the starting checkpoint ("i_" = post-trained)')
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    res = {"per_task": [], "matrix": {}}
    items = collections.defaultdict(dict)  # (lang, arm, seed) -> task -> {qid: correct}
    for d in sorted(glob.glob(os.path.join(a.eval_root, "*", "*"))):
        lang, run = d.split("/")[-2], d.split("/")[-1]
        if a.prefix:
            if not run.startswith(a.prefix):
                continue
            run = run[len(a.prefix):]
        elif run.startswith("i_"):
            continue
        arm, seed = (run.rsplit("_s", 1) + ["42"])[:2] if "_s" in run else (run, "42")
        items[(lang, arm, seed)] = load_run(d)
        for t in CALIBRATE & set(items[(lang, arm, seed)]):
            c = calibrated(d, t)
            if c is not None:
                items[(lang, arm, seed)][t] = c
    langs = sorted({k[0] for k in items})
    # per-task contrasts vs random (pooled over seeds present for both arms)
    diffs = collections.defaultdict(list)  # (arm, group) -> list of (lang, task, a_vec, r_vec)
    for lang in langs:
        seeds_r = {k[2] for k in items if k[0] == lang and k[1] == "random"}
        tasks = sorted(set().union(*[set(v) for k, v in items.items() if k[0] == lang]))
        for arm in ["base"] + ARMS:
            for t in tasks:
                A, R = [], []
                seeds = sorted({k[2] for k in items if k[0] == lang and k[1] == arm})
                for s in seeds:
                    ra = items[(lang, arm, s)].get(t)
                    rr = items.get((lang, "random", s if s in seeds_r else "42"), {}).get(t)
                    if not ra or not rr:
                        continue
                    q = [x for x in ra if x in rr]
                    A += [ra[x] for x in q]
                    R += [rr[x] for x in q]
                if not A:
                    continue
                d_, lo, hi, aw, bw, p = paired(A, R, rng)
                row = {"lang": lang, "task": t, "group": GROUP.get(t, "other"), "arm": arm, "n": len(A),
                       "n_seeds": len(seeds), "acc": float(np.mean(A)), "acc_random": float(np.mean(R)),
                       "delta": float(d_), "ci95": [lo, hi], "p": float(p)}
                res["per_task"].append(row)
                if arm != "base":
                    diffs[(arm, row["group"])].append((np.asarray(A, float), np.asarray(R, float)))
    # Holm within (lang, arm, group)
    keyf = lambda r: (r["lang"], r["arm"], r["group"])
    for k in {keyf(r) for r in res["per_task"]}:
        idx = [i for i, r in enumerate(res["per_task"]) if keyf(r) == k]
        for i, adj in zip(idx, holm([res["per_task"][i]["p"] for i in idx])):
            res["per_task"][i]["p_holm_group"] = adj
    # transfer matrix: mean over tasks of the group; bootstrap resamples items within each task
    for (arm, g), lst in diffs.items():
        point = float(np.mean([x.mean() - y.mean() for x, y in lst]))
        boots = np.zeros(a.B)
        for x, y in lst:
            idx = rng.integers(0, len(x), size=(a.B, len(x)))
            boots += x[idx].mean(1) - y[idx].mean(1)
        boots /= len(lst)
        res["matrix"][f"{arm}|{g}"] = {"delta": point, "ci95": [float(np.percentile(boots, 2.5)),
                                                                 float(np.percentile(boots, 97.5))],
                                        "n_tasks": len(lst)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in sorted(res["matrix"].items()):
        print(f"{k:32s} {100*v['delta']:+6.2f}  [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]  tasks={v['n_tasks']}")


if __name__ == "__main__":
    main()
