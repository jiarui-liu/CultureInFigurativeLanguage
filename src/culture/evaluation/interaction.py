#!/usr/bin/env python3
"""Interaction (difference-in-differences) of an arm effect across two item groups.

Reporting only the culture-bearing subset of a benchmark answers the wrong question.
A null on that subset cannot distinguish "the training did not install culture" from
"the training did nothing" or "this benchmark cannot detect anything". The interaction
keeps the non-culture items as a yardstick and asks whether the arm's effect *differs*
between the groups:

    interaction = (arm - control | culture items) - (arm - control | other items)

It uses every item, so no power is lost to subsetting, and it cannot be manufactured
by curating the subset: adding or dropping items pulls the two halves together.

Groups come from fields the dataset's own authors wrote (CMMLU `subject`, ParamBench
`subject`), never from our own annotation, so the split is independent of the method
under test.

Two grouping modes:
  --mode tasks   the groups are two separate eval tasks (ParamBench culture vs other).
                 Disjoint items -> independent bootstraps, difference of the two deltas.
  --mode subject one task, split by the per-record `subject` field (CMMLU). The items
                 are disjoint but come from one run; still independent bootstraps.

Usage:
  python -m culture.evaluation.interaction --lang hi --mode tasks \
      --group_a parambench_hi_culture --group_b parambench_hi_other
  python -m culture.evaluation.interaction --lang zh --mode subject --task cmmlu \
      --subjects_a chinese_food_culture ethnology
"""
import argparse
import json
import os

import numpy as np

EVAL = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval"
BABEL = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/from_babel/eval/9b_babel"
B = 10000


def records(lang, arm, task):
    for root in (os.path.join(EVAL, lang), os.path.join(BABEL, lang)):
        p = os.path.join(root, arm, task + ".json")
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))["records"]
    return None


def vec(recs, keep=None, field=None, values_a=None, side=None):
    """Per-item correctness in a stable qid order.

    ``keep``  restrict to these subjects (mode=subject)
    ``field`` split on an arbitrary per-record field (mode=field), e.g. Global-PIQA's
              own ``cultural`` flag, which run_eval flattens from the item meta.
    """
    out, seen = {}, {}
    for r in recs:
        q = r["qid"]
        k = seen.get(q, 0); seen[q] = k + 1
        key = q if k == 0 else f"{q}#{k}"
        if field is not None:
            in_a = str(r.get(field)) in values_a
            if (side == "a") != in_a:
                continue
        elif keep is not None and (r.get("subject") or q.rsplit("/", 1)[0]) not in keep:
            continue
        out[key] = int(r.get("correct_norm", r.get("correct", 0)))
    return out


def delta_boot(a_map, c_map, rng):
    """Paired bootstrap of mean(arm) - mean(control) over the shared items."""
    q = sorted(set(a_map) & set(c_map))
    a = np.array([a_map[x] for x in q], float)
    c = np.array([c_map[x] for x in q], float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    return a.mean() - c.mean(), a[idx].mean(1) - c[idx].mean(1), len(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--mode", choices=["tasks", "subject", "field"], required=True)
    ap.add_argument("--task"), ap.add_argument("--group_a"), ap.add_argument("--group_b")
    ap.add_argument("--subjects_a", nargs="*", default=[])
    ap.add_argument("--field", help="per-record field to split on (mode=field)")
    ap.add_argument("--values_a", nargs="*", default=["1"],
                    help="field values that define group A")
    ap.add_argument("--control", default="unfiltered")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--label_a", default="culture"), ap.add_argument("--label_b", default="other")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    def group_maps(arm):
        if a.mode == "tasks":
            ra, rb = records(a.lang, arm, a.group_a), records(a.lang, arm, a.group_b)
            if ra is None or rb is None:
                return None
            return vec(ra), vec(rb)
        recs = records(a.lang, arm, a.task)
        if recs is None:
            return None
        if a.mode == "field":
            va = set(a.values_a)
            return (vec(recs, field=a.field, values_a=va, side="a"),
                    vec(recs, field=a.field, values_a=va, side="b"))
        sa = set(a.subjects_a)
        allsub = {(r.get("subject") or r["qid"].rsplit("/", 1)[0]) for r in recs}
        return vec(recs, sa), vec(recs, allsub - sa)

    ctrl = group_maps(a.control)
    if ctrl is None:
        raise SystemExit(f"control {a.control} missing")
    print(f"=== {a.lang}: interaction on [{a.label_a}] vs [{a.label_b}], control={a.control} ===")
    print(f"    group sizes: {a.label_a} n={len(ctrl[0])}, {a.label_b} n={len(ctrl[1])}\n")

    report = {"lang": a.lang, "control": a.control, "mode": a.mode,
              "n_a": len(ctrl[0]), "n_b": len(ctrl[1]), "arms": {}}
    for arm in a.arms:
        g = group_maps(arm)
        if g is None:
            print(f"  [skip] {arm}"); continue
        da, ba, na = delta_boot(g[0], ctrl[0], rng)
        db, bb, nb = delta_boot(g[1], ctrl[1], rng)
        # The two groups share no items, so their bootstraps are independent and the
        # interaction distribution is just their difference.
        inter = ba - bb
        lo, hi = np.percentile(inter, [2.5, 97.5])
        p = 2 * min((inter <= 0).mean(), (inter >= 0).mean())
        star = "*" if p < 0.05 else " "
        print(f"  {arm:14} Δ{a.label_a}={100*da:+6.2f}  Δ{a.label_b}={100*db:+6.2f}  "
              f"interaction={100*(da-db):+6.2f}{star} [{100*lo:+.2f}, {100*hi:+.2f}] p={p:.4f}")
        report["arms"][arm] = {
            f"delta_{a.label_a}": round(100 * da, 2), f"delta_{a.label_b}": round(100 * db, 2),
            "interaction": round(100 * (da - db), 2),
            "interaction_ci": [round(100 * lo, 2), round(100 * hi, 2)],
            "p": round(float(p), 5)}

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
