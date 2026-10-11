#!/usr/bin/env python3
"""Does each kind of training text help the items that ask for the layer it contains?

§5.4 shows idioms are 62-84% symbolic/evaluative while culture benchmarks are 2-11% symbolic,
and uses that to explain why transfer is narrow.  The explanation makes a prediction *inside*
each benchmark: the handful of symbolic items a culture benchmark does contain should respond
to idiom-centered training, and its factual and material-practice items to culture-rich
training.  If no such interaction exists, the taxonomy describes the sources without
explaining the behaviour, and the paper should say so.

For every arm and benchmark we compute the gain over the token-matched control separately on
the symbolic items and on the rest, and test the difference (a paired item-level bootstrap of
the interaction, plus a label-permutation test).  Families are Holm-corrected.

    PYTHONPATH=src python -m culture.analysis.v3.layer_interaction
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval")
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

ARMS = {"base": "base", "random": "unfiltered", "idiom_untagged": "untagged",
        "idiom_cpt": "cpt", "culture": "culture", "culture_notes": "culturenotes"}
REF = "random"
LANG_OF = {}


def load(lang, arm, task):
    """Returns {key: correct}. Keys disambiguate repeated qids (ArabCulture repeats a qid
    across country variants); the taxonomy labels are joined on the base qid."""
    p = f"{EVAL}/{lang}/{ARMS[arm]}/{task}.json"
    if not os.path.exists(p):
        return None
    recs = json.load(open(p, encoding="utf-8"))["records"]
    seen, out = {}, {}
    for r in recs:
        q = r["qid"]
        n = seen.get(q, 0)
        seen[q] = n + 1
        out[q if n == 0 else f"{q}#{n}"] = int(r.get("correct_norm", r.get("correct", 0)))
    return out


def boot_interaction(d, mask, reps=5000, seed=0):
    """Paired bootstrap of mean(d[mask]) - mean(d[~mask]) over items."""
    d = np.asarray(d, float)
    mask = np.asarray(mask, bool)
    if mask.sum() < 10 or (~mask).sum() < 10:
        return None
    rng = np.random.default_rng(seed)
    ia, ib = np.flatnonzero(mask), np.flatnonzero(~mask)
    a = d[ia][rng.integers(0, len(ia), size=(reps, len(ia)))].mean(1)
    b = d[ib][rng.integers(0, len(ib), size=(reps, len(ib)))].mean(1)
    diff = a - b
    return {
        "delta_in": float(d[mask].mean()), "delta_out": float(d[~mask].mean()),
        "interaction": float(d[mask].mean() - d[~mask].mean()),
        "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "n_in": int(mask.sum()), "n_out": int((~mask).sum()),
    }


def perm_p(d, mask, reps=5000, seed=0):
    d = np.asarray(d, float)
    mask = np.asarray(mask, bool)
    obs = d[mask].mean() - d[~mask].mean()
    rng = np.random.default_rng(seed)
    k = int(mask.sum())
    hits = 0
    for _ in range(reps):
        p = rng.permutation(len(d))
        m = np.zeros(len(d), bool)
        m[p[:k]] = True
        if abs(d[m].mean() - d[~m].mean()) >= abs(obs):
            hits += 1
    return (hits + 1) / (reps + 1)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n, out, prev = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = max(prev, min(1.0, (n - rank) * pvals[i]))
        out[i] = adj
        prev = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="item_taxonomy.json")
    ap.add_argument("--out", default="layer_interaction.json")
    ap.add_argument("--only_tasks", default="",
                    help="restrict to these tasks (comma-separated); the pooled test is then "
                         "a within-benchmark-family test rather than idiom-vs-culture")
    ap.add_argument("--eval_root", default=None)
    ap.add_argument("--arms", default=None,
                    help="comma-separated name=dir pairs for a non-default grid (e.g. 2B)")
    ap.add_argument("--group", default="symbolic_evaluative",
                    help="category (or '+'-joined set) that defines the 'in' group")
    args = ap.parse_args()

    global EVAL, ARMS
    if args.eval_root:
        EVAL = args.eval_root
    if args.arms:
        ARMS = dict(p.split("=", 1) for p in args.arms.split(","))

    lab = json.load(open(os.path.join(OUT_DIR, args.labels), encoding="utf-8"))
    labels = lab["labels"]
    groups = set(args.group.split("+"))

    # task -> language, from the dumped items
    items_dir = os.environ.get(
        "CULTURE_ITEMS_DIR",
        "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/items")
    for t in labels:
        p = f"{items_dir}/{t}.jsonl"
        if os.path.exists(p):
            LANG_OF[t] = json.loads(open(p, encoding="utf-8").readline())["lang"]

    rows, keys = [], []
    pooled = defaultdict(lambda: ([], []))        # (lang, arm) -> (deltas, mask)
    only = {t for t in args.only_tasks.split(",") if t}
    for task, m in labels.items():
        lang = LANG_OF.get(task)
        if lang is None or (only and task not in only):
            continue
        ref = load(lang, REF, task)
        if not ref:
            continue
        for arm in ARMS:
            if arm in (REF, "base"):
                continue
            rec = load(lang, arm, task)
            if not rec:
                continue
            # labels are keyed on the base qid; expand to every occurrence of it
            qids = [q for q in rec if q in ref and q.split("#")[0] in m]
            if len(qids) < 40:
                continue
            d = np.array([rec[q] - ref[q] for q in qids], float)
            mask = np.array([m[q.split("#")[0]] in groups for q in qids], bool)
            pooled[(lang, arm)][0].extend(d.tolist())
            pooled[(lang, arm)][1].extend(mask.tolist())
            r = boot_interaction(d, mask)
            if r is None:
                continue
            r["p"] = perm_p(d, mask)
            r.update({"task": task, "lang": lang, "arm": arm})
            rows.append(r)
            keys.append(f"{lang}/{task}/{arm}")

    for r, p in zip(rows, holm([x["p"] for x in rows])):
        r["p_holm"] = p

    pooled_out = {}
    for (lang, arm), (d, mask) in pooled.items():
        r = boot_interaction(d, mask)
        if r is None:
            continue
        r["p"] = perm_p(d, mask)
        pooled_out[f"{lang}/{arm}"] = r

    out = {"group": sorted(groups), "annotator": lab.get("model"),
           "per_task": dict(zip(keys, rows)), "pooled": pooled_out,
           "category_counts": {t: dict(c) for t, c in lab.get("distribution", {}).items()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}")

    print(f"\n=== interaction: gain on {sorted(groups)} items minus gain on the rest ===")
    for k, r in sorted(out["per_task"].items()):
        flag = "*" if r["p_holm"] < 0.05 else (" " if r["p"] >= 0.05 else "~")
        print(f" {flag} {k:48s} in {r['delta_in']:+.3f} (n={r['n_in']:5d})  "
              f"out {r['delta_out']:+.3f} (n={r['n_out']:5d})  "
              f"int {r['interaction']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
              f"p={r['p']:.3f} holm={r['p_holm']:.3f}")
    print("\n=== pooled over benchmarks ===")
    for k, r in sorted(pooled_out.items()):
        print(f"   {k:20s} in {r['delta_in']:+.3f} (n={r['n_in']:5d})  "
              f"out {r['delta_out']:+.3f} (n={r['n_out']:5d})  "
              f"int {r['interaction']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p']:.3f}")


if __name__ == "__main__":
    main()
