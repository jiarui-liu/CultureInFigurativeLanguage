#!/usr/bin/env python3
"""Training dynamics: when does each effect appear during continued pretraining?

The paper reports end-of-training numbers only, so it cannot say whether item-specific idiom
knowledge saturates early, whether the cost on unseen idioms grows with training, or whether a
cultural effect appears transiently and is then overwritten.  Arabic is the language to ask in:
1,608 steps, and the only language with a reverse-direction signal.  `eval_ar_dynamics.slurm`
evaluates Idiom-CPT, Random-CPT and Culture-CPT at steps 200/400/800/1200; this script joins
those records to the final checkpoints and reports accuracy and margin against the
step-matched Random-CPT checkpoint.

    PYTHONPATH=src python -m culture.analysis.v3.dynamics
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval")
DYN = f"{ROOT}/ar_dynamics"
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

STEPS = [200, 400, 800, 1200]
FINAL = {"cpt": "cpt", "unfiltered": "unfiltered", "culture": "culture"}
TASKS = ["kinayat_meaning", "ar_figurative", "arabculture", "alyah",
         "idiomatlas_mc_ar_seen", "idiomatlas_mc_ar_unseen", "symbolism_v2_ar_letter"]


def margin(r):
    lp = r.get("logprobs_norm") or r.get("logprobs") or []
    g = r.get("gold")
    if not lp or not isinstance(g, int) or g >= len(lp):
        return None
    o = [v for i, v in enumerate(lp) if i != g]
    return float(lp[g] - max(o)) if o else None


def read(path):
    if not os.path.exists(path):
        return None
    recs = json.load(open(path, encoding="utf-8"))["records"]
    return {r["qid"]: (int(r.get("correct_norm", r.get("correct", 0))), margin(r))
            for r in recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dynamics_ar.json")
    args = ap.parse_args()
    out = {}

    for task in TASKS:
        for step in STEPS + ["final"]:
            if step == "final":
                ref = read(f"{ROOT}/ar/unfiltered/{task}.json")
            else:
                ref = read(f"{DYN}/unfiltered_{step}/{task}.json")
            if ref is None:
                continue
            for arm in ("cpt", "culture"):
                p = (f"{ROOT}/ar/{FINAL[arm]}/{task}.json" if step == "final"
                     else f"{DYN}/{arm}_{step}/{task}.json")
                rec = read(p)
                if rec is None:
                    continue
                q = [k for k in rec if k in ref]
                da = np.mean([rec[k][0] - ref[k][0] for k in q])
                dm = [rec[k][1] - ref[k][1] for k in q
                      if rec[k][1] is not None and ref[k][1] is not None]
                out[f"{task}/{arm}/{step}"] = {
                    "n": len(q), "acc": float(np.mean([rec[k][0] for k in q])),
                    "acc_ref": float(np.mean([ref[k][0] for k in q])),
                    "delta_acc": float(da),
                    "delta_margin": float(np.mean(dm)) if dm else None,
                }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}")

    for task in TASKS:
        line = [f"{task:26s}"]
        for arm in ("cpt", "culture"):
            cells = []
            for step in STEPS + ["final"]:
                v = out.get(f"{task}/{arm}/{step}")
                cells.append(f"{100 * v['delta_acc']:+6.1f}" if v else "    --")
            line.append(f"{arm[:4]}:" + " ".join(cells))
        print("  ".join(line))


if __name__ == "__main__":
    main()
