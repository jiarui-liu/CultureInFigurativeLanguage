#!/usr/bin/env python3
"""Rebuttal R-4Spy: the same Idiom-Text / Idiom+Meaning / Random-CPT contrasts at Qwen3.5-2B, for two
starting checkpoints (prefix '' = Qwen3.5-2B, 'i_' = the second 2B init used in the bidirectional study),
so the 9B pattern can be checked across scale and initialization (independent training runs)."""
import json, sys
import numpy as np
from culture.analysis.rebuttal import idiomtext_vs_random as M

ROOT = M.ROOT / "eval2b"
TASKS = ["idiomatlas_mc_{L}_seen", "idiomatlas_mc_{L}_unseen", "kinayat_meaning", "ar_figurative", "mabl_hi",
         "symbolism_v2_{L}_letter", "chid", "chengyu_bench_app"]
out = []
rng = np.random.default_rng(1)
for pre in ["", "i_"]:
    for L in ["ar", "hi", "zh"]:
        runs = {"random": f"{pre}random", "untagged": f"{pre}idiom_untagged", "cpt": f"{pre}idiom_tagged"}
        for t in TASKS:
            t = t.format(L=L)
            acc = {}
            for k, r in runs.items():
                p = ROOT / L / r / f"{t}.json"
                if not p.exists():
                    continue
                j = json.load(open(p)); mode = j.get("metrics", {}).get("score_mode", "continuation")
                key = "correct_norm" if mode == "continuation" else "correct"
                acc[k] = {x["qid"]: int(x[key]) for x in j["records"]}
            if len(acc) < 3:
                continue
            q = sorted(set.intersection(*[set(v) for v in acc.values()]))
            A = {k: np.array([v[i] for i in q]) for k, v in acc.items()}
            idx = rng.integers(0, len(q), size=(10000, len(q)))
            row = {"init": pre or "main", "lang": L, "task": t, "n": len(q), "acc": {k: round(100 * v.mean(), 1) for k, v in A.items()}}
            for a, b in [("untagged", "random"), ("cpt", "random"), ("cpt", "untagged")]:
                d = A[a] - A[b]; bs = d[idx].mean(1); p, w, l = M.mcnemar(A[a], A[b])
                row[f"{a}-{b}"] = [round(100 * d.mean(), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1), round(p, 4)]
            out.append(row)
            print(row["init"], L, f"{t:28s}", row["n"], "IT-R", row["untagged-random"], "IM-R", row["cpt-random"], "IM-IT", row["cpt-untagged"])
json.dump(out, open(sys.argv[1], "w"), indent=1)
