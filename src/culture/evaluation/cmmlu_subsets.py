#!/usr/bin/env python3
"""Split the existing CMMLU eval records into culture-bearing subsets -- no new eval runs.

The paper's Chinese "culture" column is CCPM alone (classical-poetry matching, and the row
our own taxonomy analysis daggers at 0.16 annotator agreement). CMMLU is already evaluated
on every arm and every one of its 11,582 records carries a ``subject`` field, so two
China-specific subsets can be recovered by re-aggregating records that are already on disk:

  cmmlu_culture         chinese_food_culture + ethnology -- the two subjects that are
                        cultural practice rather than academic knowledge. ``sociology``
                        is deliberately excluded: it is not China-specific.
  cmmlu_china_specific  the 16 China-specific configs already defined in
                        ``culture.evaluation.tasks_zh.CMMLU_DEFAULT_SUBJECTS``.

Both are reported against the token-matched Random control (``unfiltered``), the same
comparison the paper uses everywhere else, with paired bootstrap CIs and exact McNemar
from ``compute_cis`` so the numbers are computed the same way as the main table.

Usage:
  python -m culture.evaluation.cmmlu_subsets                      # 9B
  python -m culture.evaluation.cmmlu_subsets --scale 2b           # 2B study
  python -m culture.evaluation.cmmlu_subsets --out docs/paper_stats/v2/cmmlu_subsets.json
"""
import argparse
import json
import os

import numpy as np

from culture.evaluation.compute_cis import boot_ci, paired
from culture.evaluation.tasks_zh import CMMLU_DEFAULT_SUBJECTS

EVAL_9B = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh"
EVAL_2B = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/from_babel/eval/2b/zh"

# chinese_food_culture and ethnology are the two CMMLU subjects that test cultural
# practice and ethnic tradition; the other 14 China-specific ones test history,
# literature, law, policy and medicine -- knowledge about China, not its culture.
CULTURE_SUBJECTS = ["chinese_food_culture", "ethnology"]

SUBSETS = {
    "cmmlu_culture": CULTURE_SUBJECTS,
    "cmmlu_china_specific": list(CMMLU_DEFAULT_SUBJECTS),
    "cmmlu_full": None,  # reference: the number already in the paper
}


def load_items(path):
    """qid -> correct_norm, plus qid -> subject. Records keep per-item subjects."""
    d = json.load(open(path, encoding="utf-8"))
    out, subj = {}, {}
    for r in d["records"]:
        q = r["qid"]
        out[q] = int(r.get("correct_norm", r.get("correct", 0)))
        # qids are "<subject>/<index>", but prefer the explicit field when present
        subj[q] = r.get("subject") or q.rsplit("/", 1)[0]
    return out, subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["9b", "2b"], default="9b")
    ap.add_argument("--control", default=None,
                    help="Arm used as the token-matched control (default: unfiltered / i_random).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = EVAL_9B if a.scale == "9b" else EVAL_2B
    control = a.control or ("unfiltered" if a.scale == "9b" else "i_random")
    arms = sorted(d for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, "cmmlu.json")))
    if control not in arms:
        raise SystemExit(f"control arm {control!r} has no cmmlu.json under {root}; have {arms}")

    rng = np.random.default_rng(a.seed)
    data = {arm: load_items(os.path.join(root, arm, "cmmlu.json")) for arm in arms}
    _, subj = data[control]

    report = {"scale": a.scale, "root": root, "control": control, "subsets": {}}
    for name, subjects in SUBSETS.items():
        qids = sorted(q for q, s in subj.items() if subjects is None or s in subjects)
        if not qids:
            print(f"[warn] {name}: no items matched"); continue
        present = sorted({subj[q] for q in qids})
        entry = {"n": len(qids), "subjects": present, "arms": {}}
        ctrl = [data[control][0][q] for q in qids]
        for arm in arms:
            items, _ = data[arm]
            miss = [q for q in qids if q not in items]
            if miss:
                print(f"[warn] {name}/{arm}: {len(miss)} qids absent, skipping arm"); continue
            v = [items[q] for q in qids]
            lo, hi = boot_ci(v, rng)
            e = {"acc": round(100 * float(np.mean(v)), 2),
                 "ci": [round(100 * lo, 2), round(100 * hi, 2)]}
            if arm != control:
                d, dlo, dhi, win, loss, p = paired(v, ctrl, rng)
                e.update(delta_vs_control=round(100 * d, 2),
                         delta_ci=[round(100 * dlo, 2), round(100 * dhi, 2)],
                         mcnemar_p=round(p, 5), n_win=win, n_loss=loss)
            entry["arms"][arm] = e
        report["subsets"][name] = entry

    for name, e in report["subsets"].items():
        print(f"\n=== {name}  (n={e['n']}, {len(e['subjects'])} subjects) ===")
        print(f"    {', '.join(e['subjects'])}")
        for arm, v in sorted(e["arms"].items()):
            d = ""
            if "delta_vs_control" in v:
                star = "*" if v["mcnemar_p"] < 0.05 else " "
                d = (f"   vs {control}: {v['delta_vs_control']:+6.2f}{star} "
                     f"[{v['delta_ci'][0]:+.2f}, {v['delta_ci'][1]:+.2f}]  p={v['mcnemar_p']:.4f}")
            print(f"  {arm:16} {v['acc']:6.2f}  [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]{d}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
