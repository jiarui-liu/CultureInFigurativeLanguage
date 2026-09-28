#!/usr/bin/env python3
"""Additional statistics for the 9B CPT study from the stored per-item records.

1. Alyah and DziriEval with their figurative-language items removed (the items
   that make up AR-Figurative), for every contrast.
2. Holm-Bonferroni correction of the Idiom-CPT vs Random-CPT McNemar p-values,
   over all 17 benchmarks and within each benchmark group.

Usage:
  python -m culture.bidirectional.stats_9b --eval_root $B/hf9b/eval \
      --old_report docs/paper_stats/ci_report.json --out docs/paper_stats/v2/stats_9b.json
"""
import argparse
import json

import numpy as np

from culture.evaluation.compute_cis import load_run, paired

GROUPS = {
    "idiom_meaning": [("ar", "kinayat_meaning"), ("zh", "chengyu_bench")],
    "figurative": [("ar", "ar_figurative"), ("hi", "mabl")],
    "cloze": [("ar", "kinayat_cloze"), ("zh", "chid")],
    "culture": [("ar", "alyah"), ("ar", "dzirieval"), ("ar", "arabculture"), ("ar", "arabic_cultural_qa"),
                ("ar", "global_piqa_ar"), ("hi", "global_piqa"), ("zh", "ccpm")],
    "regional": [("ar", "arabicmmlu"), ("hi", "milu"), ("zh", "cmmlu")],
    "control": [("ar", "global_piqa_ar_parallel")],
}


def holm(ps):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def contrast(ra, rb, qids, rng):
    q = [x for x in qids if x in ra and x in rb]
    a = [ra[x] for x in q]
    b = [rb[x] for x in q]
    d, lo, hi, a_win, b_win, p = paired(a, b, rng)
    return {"n": len(q), "acc_a": float(np.mean(a)), "acc_b": float(np.mean(b)), "delta": float(d),
            "ci95": [lo, hi], "a_win": a_win, "b_win": b_win, "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20260927)
    runs = {L: {r: load_run(f"{a.eval_root}/{L}/{r}") for r in ["base", "cpt", "unfiltered", "untagged"]}
            for L in ["ar", "hi", "zh"]}
    out = {"figurative_removed": {}, "holm": {}}

    # 1. Alyah / DziriEval without the figurative items
    fig = set(runs["ar"]["cpt"]["ar_figurative"])
    for t in ["alyah", "dzirieval"]:
        allq = list(runs["ar"]["cpt"][t])
        nonfig = [q for q in allq if q not in fig]
        res = {"n_all": len(allq), "n_nonfig": len(nonfig)}
        for other in ["base", "unfiltered", "untagged"]:
            res[f"cpt_vs_{other}"] = contrast(runs["ar"]["cpt"][t], runs["ar"][other][t], nonfig, rng)
        res["untagged_vs_unfiltered"] = contrast(runs["ar"]["untagged"][t], runs["ar"]["unfiltered"][t], nonfig, rng)
        res["acc_nonfig"] = {r: float(np.mean([runs["ar"][r][t][q] for q in nonfig if q in runs["ar"][r][t]]))
                             for r in ["base", "cpt", "unfiltered", "untagged"] if runs["ar"][r].get(t)}
        out["figurative_removed"][t] = res

    # 2. Holm over cpt vs unfiltered
    rows = []
    for g, items in GROUPS.items():
        for L, t in items:
            ra, rb = runs[L]["cpt"].get(t), runs[L]["unfiltered"].get(t)
            if not ra or not rb:
                continue
            c = contrast(ra, rb, list(ra), rng)
            rows.append({"group": g, "lang": L, "task": t, **c})
    ps = [r["p"] for r in rows]
    for r, adj in zip(rows, holm(ps)):
        r["p_holm_all"] = adj
    for g in GROUPS:
        idx = [i for i, r in enumerate(rows) if r["group"] == g]
        for i, adj in zip(idx, holm([rows[i]["p"] for i in idx])):
            rows[i]["p_holm_group"] = adj
    out["holm"] = rows
    json.dump(out, open(a.out, "w"), indent=1)
    for t, r in out["figurative_removed"].items():
        print(t, r["n_all"], "->", r["n_nonfig"], {k: (round(v["delta"], 4), [round(x, 4) for x in v["ci95"]], round(v["p"], 4))
                                                    for k, v in r.items() if isinstance(v, dict) and "delta" in v},
              {k: round(v, 4) for k, v in r["acc_nonfig"].items()})
    for r in rows:
        print(f"{r['group']:14s} {r['lang']} {r['task']:24s} d={r['delta']:+.4f} p={r['p']:.4g} "
              f"holm_all={r['p_holm_all']:.4g} holm_group={r['p_holm_group']:.4g}")


if __name__ == "__main__":
    main()
