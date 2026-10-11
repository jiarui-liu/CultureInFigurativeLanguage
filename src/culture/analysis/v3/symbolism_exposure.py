#!/usr/bin/env python3
"""Does exposure to an entity's idioms teach what that entity symbolizes?

§5.5 finds that idiom-bearing documents lower the Hindi lure rate -- models stop reading an
entity the English way -- and §5.6 reports a null for the \\S4 embedding divergence as a
predictor of probe behaviour.  Neither asks the dose question that the forward direction asks
everywhere else: for the entity a probe item is about, does the model do better when the
training corpus contained more idioms using that entity?

Each symbolism-probe item names its entity, and the knowledge base gives, for every entity, the
idioms that contain it; the corpus statistics give each idiom's document count.  Summing the
counts over an entity's idioms gives its corpus exposure, which we relate to per-item accuracy
and to the lure rate, for each arm against the token-matched control.

    PYTHONPATH=src:src/culture/analysis/v2 python symbolism_exposure.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import common  # noqa: E402

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval")
DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")
MC = f"{DATA}/eval/mc"
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")

COUNT_FILES = {
    "zh": [f"{DATA}/fineweb-edu-zh-chengyu-cpt/stats/kept_idiom_counts_zh.json",
           f"{DATA}/mc4-zh-idiom-cpt/stats/kept_idiom_counts_zh.json"],
    "hi": [f"{DATA}/hi-proverbs-cpt/stats/kept_idiom_counts_hi.json"],
    "ar": [f"{DATA}/ar-amthal-cpt/stats/kept_idiom_counts_ar.json"],
}
ARMS = {"base": "base", "random": "unfiltered", "idiom_untagged": "untagged",
        "idiom_cpt": "cpt", "culture": "culture", "culture_notes": "culturenotes"}


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
    rho = spearman(a, b)
    if rho is None:
        return None, None
    rng = np.random.default_rng(seed)
    b = np.asarray(b, float)
    hits = sum(abs(spearman(a, rng.permutation(b))) >= abs(rho) for _ in range(reps))
    return rho, (hits + 1) / (reps + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="symbolism_exposure.json")
    args = ap.parse_args()
    out = {}

    for lang in ("ar", "hi", "zh"):
        counts = Counter()
        for p in COUNT_FILES[lang]:
            if os.path.exists(p):
                for k, v in json.load(open(p, encoding="utf-8")).items():
                    counts[k] += int(v)
        kb = common.load_kb(lang)
        ent_exposure = defaultdict(int)
        ent_idioms = Counter()
        for r in kb:
            c = counts.get(r["idiom"], 0)
            for e in r["entities"]:
                ent_exposure[e] += c
                ent_idioms[e] += 1

        path = f"{MC}/symbolism_v2_{lang}_letter.jsonl"
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        feats = {}
        for o in rows:
            m = o.get("meta") or {}
            e = common.norm_entity(m.get("entity", ""), lang)
            feats[o["qid"]] = {
                "entity": e,
                "exposure": ent_exposure.get(e, 0),
                "n_idioms": ent_idioms.get(e, 0),
                "lure": m.get("lure"),
            }
        covered = sum(1 for v in feats.values() if v["n_idioms"] > 0)
        res = {"n_items": len(feats), "entities_in_kb": covered, "arms": {}}

        ref = None
        for arm in ("random", "base", "idiom_cpt", "idiom_untagged", "culture",
                    "culture_notes"):
            p = f"{EVAL}/{lang}/{ARMS[arm]}/symbolism_v2_{lang}_letter.json"
            if not os.path.exists(p):
                continue
            recs = {r["qid"]: r for r in json.load(open(p, encoding="utf-8"))["records"]}
            if arm == "random":
                ref = recs
                continue
            if ref is None:
                continue
            q = [x for x in feats if x in recs and x in ref and feats[x]["n_idioms"] > 0]
            if len(q) < 30:
                continue
            gain = np.array([int(recs[x].get("correct", 0)) - int(ref[x].get("correct", 0))
                             for x in q], float)
            expo = np.array([np.log1p(feats[x]["exposure"]) for x in q])
            nid = np.array([np.log1p(feats[x]["n_idioms"]) for x in q])
            r1, p1 = perm_p(expo, gain)
            r2, p2 = perm_p(nid, gain)
            lure = [x for x in q if feats[x]["lure"] is not None]
            dl = None
            if lure:
                dl = float(np.mean([
                    int(recs[x].get("pred") == feats[x]["lure"])
                    - int(ref[x].get("pred") == feats[x]["lure"]) for x in lure]))
            res["arms"][arm] = {
                "n": len(q), "mean_gain": float(gain.mean()),
                "exposure_rho": r1, "exposure_p": p1,
                "n_idioms_rho": r2, "n_idioms_p": p2,
                "delta_lure_rate": dl,
            }
            print(f"[{lang}/{arm}] n={len(q)} gain={gain.mean():+.3f} "
                  f"exposure rho={r1} p={p1} dlure={dl}")
        out[lang] = res

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
