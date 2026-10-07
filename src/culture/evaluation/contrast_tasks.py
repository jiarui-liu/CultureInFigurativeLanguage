#!/usr/bin/env python3
"""Paired contrasts of any per-item eval task against the token-matched control.

``compute_cis.py`` covers the task set of the original paper; this does the same
arithmetic (paired item bootstrap + exact McNemar, reused from that module) for an
arbitrary list of tasks, so newly added benchmarks are reported on exactly the same
footing as the old ones instead of as raw accuracies.

Usage:
  python -m culture.evaluation.contrast_tasks --lang zh \
      --tasks global_piqa_zh global_piqa_zh_cultural global_piqa_zh_parallel4
  python -m culture.evaluation.contrast_tasks --lang hi --control unfiltered-matched ...
"""
import argparse
import json
import os

import numpy as np

from culture.evaluation.compute_cis import boot_ci, paired

EVAL = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval"
BABEL = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/from_babel/eval/9b_babel"
ORDER = ["base", "unfiltered", "unfiltered-matched", "cpt", "untagged", "culture", "culturenotes"]


ROOTS = [EVAL, BABEL]  # searched in order; --eval_root replaces this (e.g. babel 2B)


def load(lang, arm, task):
    for root in (os.path.join(r, lang) for r in ROOTS):
        p = os.path.join(root, arm, task + ".json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            out, seen = {}, {}
            for r in d["records"]:
                q = r["qid"]
                k = seen.get(q, 0); seen[q] = k + 1       # qids are not always unique
                out[q if k == 0 else f"{q}#{k}"] = int(r.get("correct_norm", r.get("correct", 0)))
            return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--control", default="unfiltered")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval_root", nargs="+", default=None,
                    help="eval roots holding <lang>/<arm>/<task>.json (default: the 9B roots)")
    a = ap.parse_args()
    if a.eval_root:
        ROOTS[:] = a.eval_root

    root = os.path.join(ROOTS[0], a.lang)
    arms = a.arms or [x for x in ORDER if os.path.isdir(os.path.join(root, x))]
    rng = np.random.default_rng(a.seed)
    report = {"lang": a.lang, "control": a.control, "tasks": {}}
    if a.eval_root:
        report["eval_root"] = a.eval_root

    for task in a.tasks:
        ctrl = load(a.lang, a.control, task)
        if ctrl is None:
            print(f"\n[skip] {task}: control arm {a.control} has no records"); continue
        qids = sorted(ctrl)
        entry = {"n": len(qids), "arms": {}}
        print(f"\n=== {a.lang} / {task}  (n={len(qids)}, control={a.control}) ===")
        for arm in arms:
            items = load(a.lang, arm, task)
            if items is None:
                continue
            missing = [q for q in qids if q not in items]
            if missing:
                print(f"  [skip] {arm}: {len(missing)} qids absent"); continue
            v = [items[q] for q in qids]
            lo, hi = boot_ci(v, rng)
            e = {"acc": round(100 * float(np.mean(v)), 2), "ci": [round(100 * lo, 2), round(100 * hi, 2)]}
            line = f"  {arm:20} {e['acc']:6.2f}  [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]"
            if arm != a.control:
                d, dlo, dhi, win, loss, p = paired(v, [ctrl[q] for q in qids], rng)
                e.update(delta=round(100 * d, 2), delta_ci=[round(100 * dlo, 2), round(100 * dhi, 2)],
                         mcnemar_p=round(p, 5))
                star = "*" if p < 0.05 else " "
                line += (f"   Δ {100 * d:+6.2f}{star} [{100 * dlo:+.2f}, {100 * dhi:+.2f}]  p={p:.4f}")
            entry["arms"][arm] = e
            print(line)
        report["tasks"][task] = entry

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
