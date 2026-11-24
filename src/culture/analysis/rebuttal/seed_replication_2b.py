#!/usr/bin/env python3
"""Seed replication at Qwen3.5-2B (Arabic): the three CPT conditions retrained with new data-order seeds.

Seed 42 = the original runs (eval2b/ar/i_{random,idiom_untagged,idiom_tagged}); new seeds live in
eval2b_seeds/ar/{arm}_s{seed}/. For each task and seed: Idiom-Text - Random-CPT, Idiom+Meaning - Random-CPT and
Idiom+Meaning - Idiom-Text (accuracy points, paired bootstrap 95% CI, exact McNemar p); then the across-seed
mean, sd and the number of seeds with the same sign.

Usage: python -m culture.analysis.rebuttal.seed_replication_2b --out docs/paper_stats/rebuttal/seed_replication_2b.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from culture.analysis.rebuttal.idiomtext_vs_random import mcnemar

B = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
TASKS = ["idiomatlas_mc_ar_seen", "idiomatlas_mc_ar_unseen", "kinayat_meaning", "ar_figurative",
         "symbolism_v2_ar_letter", "kinayat_cloze", "alyah"]
ARMS = {"random": "Random-CPT", "idiom_untagged": "Idiom-Text", "idiom_tagged": "Idiom+Meaning"}
CONTRASTS = [("idiom_untagged", "random"), ("idiom_tagged", "random"), ("idiom_tagged", "idiom_untagged")]


def run_dir(arm, seed):
    return B / "eval2b" / "ar" / f"i_{arm}" if seed == 42 else B / "eval2b_seeds" / "ar" / f"{arm}_s{seed}"


def load(p):
    if not p.exists():
        return None
    j = json.load(open(p)); mode = j.get("metrics", {}).get("score_mode", "continuation")
    key = "correct_norm" if mode == "continuation" else "correct"
    out, seen = {}, {}
    for r in j["records"]:
        q = r["qid"]; k = seen.get(q, 0); seen[q] = k + 1
        out[q if k == 0 else f"{q}#{k}"] = int(r[key])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2]); a = ap.parse_args()
    rng = np.random.default_rng(7)
    res = {}
    for t in TASKS:
        res[t] = {"per_seed": {}, "summary": {}}
        for s in a.seeds:
            recs = {arm: load(run_dir(arm, s) / f"{t}.json") for arm in ARMS}
            if any(v is None for v in recs.values()):
                continue
            q = sorted(set.intersection(*[set(v) for v in recs.values()]))
            A = {k: np.array([v[i] for i in q]) for k, v in recs.items()}
            idx = rng.integers(0, len(q), size=(10000, len(q)))
            row = {"n": len(q), "acc": {ARMS[k]: round(float(100 * v.mean()), 1) for k, v in A.items()}}
            for x, y in CONTRASTS:
                d = A[x] - A[y]; bs = d[idx].mean(1); p, w, l = mcnemar(A[x], A[y])
                row[f"{ARMS[x]} - {ARMS[y]}"] = {"delta": round(100 * d.mean(), 2),
                    "ci95": [round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)],
                    "mcnemar_p": round(float(p), 4)}
            res[t]["per_seed"][str(s)] = row
        for x, y in CONTRASTS:
            k = f"{ARMS[x]} - {ARMS[y]}"
            ds = [v[k]["delta"] for v in res[t]["per_seed"].values()]
            if ds:
                res[t]["summary"][k] = {"seeds": len(ds), "mean": round(float(np.mean(ds)), 2),
                                        "sd": round(float(np.std(ds, ddof=1)), 2) if len(ds) > 1 else None,
                                        "min": float(min(ds)), "max": float(max(ds)), "n_positive": int(sum(d > 0 for d in ds)),
                                        "n_negative": int(sum(d < 0 for d in ds))}
        seeds_done = list(res[t]["per_seed"])
        print(f"{t:26s} seeds={seeds_done}")
        for k, v in res[t]["summary"].items():
            print(f"   {k:30s} mean {v['mean']:+6.2f} sd {v['sd']} range [{v['min']:+.1f},{v['max']:+.1f}] +{v['n_positive']}/-{v['n_negative']}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
