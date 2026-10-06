"""Dose-response: how much does a model learn per occurrence of an idiom in its training data?

The IdiomAtlas-MC seen/unseen split tells us *that* idiom-centered training teaches the idioms it
was shown. It does not tell us how much exposure is needed, or whether the returns diminish --
the question that actually matters when building a training corpus.

The Chinese corpus statistics record how many documents each idiom occurs in (1 to ~39k, over
four orders of magnitude), and every IdiomAtlas-MC "seen" item names its idiom, so the two join
exactly: 100% of seen items have an exposure count and 0% of unseen items do, which independently
confirms the split. We bin items by exposure and trace accuracy for each training arm.

    PYTHONPATH=src:src/culture/analysis/v2 python exposure_dose_response.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from common import dump

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval",
)
DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data"
)
# Documents of each idiom corpus record the KB idioms they matched; these files are the
# per-idiom document counts. Arabic's was derived from train_ar's `matched_idioms`.
COUNT_FILES = {
    "zh": [f"{DATA}/fineweb-edu-zh-chengyu-cpt/stats/kept_idiom_counts_zh.json",
           f"{DATA}/mc4-zh-idiom-cpt/stats/kept_idiom_counts_zh.json"],
    "hi": [f"{DATA}/hi-proverbs-cpt/stats/kept_idiom_counts_hi.json"],
    "ar": [f"{DATA}/ar-amthal-cpt/stats/kept_idiom_counts_ar.json"],
}
LANGS = ("ar", "hi", "zh")
ARMS = {"base": "base", "random": "unfiltered", "idiom_untagged": "untagged",
        "idiom_cpt": "cpt", "culture": "culture", "culture_notes": "culturenotes"}
# Octave bins over four orders of magnitude.
EDGES = [1, 4, 16, 64, 256, 1024, 4096, 10**9]
BIN_LABEL = ["1-3", "4-15", "16-63", "64-255", "256-1023", "1024-4095", "4096+"]


def _rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_p(a, b, reps=10000, seed=0):
    r0 = spearman(a, b)
    if r0 is None:
        return None, None
    rng = np.random.default_rng(seed)
    b = np.asarray(b, float)
    c = sum(1 for _ in range(reps) if abs(spearman(a, rng.permutation(b))) >= abs(r0))
    return round(r0, 4), round((c + 1) / (reps + 1), 5)


def boot_ci(x, reps=5000, seed=0):
    x = np.asarray(x, float)
    if x.size < 3:
        return None
    rng = np.random.default_rng(seed)
    s = [float(np.mean(rng.choice(x, x.size, replace=True))) for _ in range(reps)]
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def load_counts(lang):
    cnt = {}
    for p in COUNT_FILES.get(lang, []):
        if os.path.exists(p):
            for k, v in json.load(open(p, encoding="utf-8")).items():
                cnt[k] = cnt.get(k, 0) + int(v)
    return cnt


def load_arm(lang, arm_dir, split):
    p = f"{EVAL}/{lang}/{arm_dir}/idiomatlas_mc_{lang}_{split}.json"
    if not os.path.exists(p):
        return None
    out = {}
    for r in json.load(open(p, encoding="utf-8"))["records"]:
        lp = r.get("logprobs_norm") or r.get("logprobs")
        g = r.get("gold")
        margin = None
        if lp and g is not None and 0 <= g < len(lp):
            others = [v for i, v in enumerate(lp) if i != g]
            if others:
                margin = float(lp[g] - max(others))
        out[r["qid"]] = {
            "idiom": r.get("idiom"),
            "correct": int(r.get("correct_norm", r.get("correct", 0))),
            "margin": margin,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exposure_dose_response.json")
    args = ap.parse_args()

    res = {"method": __doc__, "languages": {}}
    for lang in LANGS:
        cnt = load_counts(lang)
        if not cnt:
            continue
        arms = {k: load_arm(lang, v, "seen") for k, v in ARMS.items()}
        arms = {k: v for k, v in arms.items() if v}
        unseen = {k: load_arm(lang, v, "unseen") for k, v in ARMS.items()}
        unseen = {k: v for k, v in unseen.items() if v}
        if "base" not in arms:
            continue
        ref = "random" if "random" in arms else "base"

        qids = [q for q in arms[ref] if arms[ref][q]["idiom"] in cnt]
        if len(qids) < 30:
            continue
        exp = np.array([cnt[arms[ref][q]["idiom"]] for q in qids], float)
        L = {
            "n_seen_items": len(arms[ref]),
            "n_joined": len(qids),
            "join_rate_seen": round(len(qids) / max(1, len(arms[ref])), 4),
            "control_arm": ref,
            "exposure": {"min": int(exp.min()), "median": float(np.median(exp)),
                         "max": int(exp.max())},
            "arms_present": sorted(arms),
        }
        if unseen.get(ref):
            u = [v["idiom"] for v in unseen[ref].values()]
            L["unseen_join_rate"] = round(
                sum(1 for i in u if i in cnt) / max(1, len(u)), 4)

        per_arm = {}
        for a_, d in arms.items():
            sub = [q for q in qids if q in d]
            if not sub:
                continue
            acc = np.array([d[q]["correct"] for q in sub], float)
            le = np.log10([cnt[arms[ref][q]["idiom"]] for q in sub])
            r_a, p_a = perm_p(le, acc)
            per_arm[a_] = {"n": len(sub), "accuracy": round(float(acc.mean()), 4),
                           "spearman_logexposure_accuracy": r_a,
                           "p_perm_accuracy": p_a}
        L["per_arm"] = per_arm

        def bin_of(v):
            for i in range(len(EDGES) - 1):
                if EDGES[i] <= v < EDGES[i + 1]:
                    return i
            return len(EDGES) - 2

        bins = defaultdict(lambda: defaultdict(list))
        for q in qids:
            b = bin_of(cnt[arms[ref][q]["idiom"]])
            for a_, d in arms.items():
                if q in d:
                    bins[b][a_].append(d[q]["correct"])
        L["by_exposure_bin"] = {
            BIN_LABEL[b]: {"n": len(v.get(ref, [])),
                           **{a_: {"mean": round(float(np.mean(x)), 4),
                                   "ci95": boot_ci(x)}
                              for a_, x in sorted(v.items())}}
            for b, v in sorted(bins.items())
        }

        for a_ in ("idiom_cpt", "idiom_untagged", "culture", "culture_notes"):
            if a_ not in arms or ref == "base":
                continue
            sub = [q for q in qids if q in arms[a_]]
            gain = [arms[a_][q]["correct"] - arms[ref][q]["correct"] for q in sub]
            lx = [np.log10(cnt[arms[ref][q]["idiom"]]) for q in sub]
            r, p = perm_p(lx, gain)
            L[f"gain_vs_logexposure_{a_}"] = {
                "n": len(sub), "spearman": r, "p_perm": p,
                "mean_gain": round(float(np.mean(gain)), 4), "ci95": boot_ci(gain)}
        res["languages"][lang] = L

    print("wrote", dump(res, args.out))
    for lang, L in res["languages"].items():
        print(f"\n[{lang}] join seen={L['join_rate_seen']:.0%} "
              f"unseen={L.get('unseen_join_rate')} control={L['control_arm']} "
              f"exposure 1..{L['exposure']['max']}")
        for a_, v in L["per_arm"].items():
            print(f"   {a_:16s} acc={v['accuracy']:.3f} "
                  f"rho(logexp,acc)={v['spearman_logexposure_accuracy']} "
                  f"p={v['p_perm_accuracy']}")


if __name__ == "__main__":
    main()
