"""Where does a model fall back on the English reading of an entity?

The symbolism probe asks what an entity stands for in a given culture and records, per item,
whether the model chose the option carrying the English association (the "lure"). Section 4
assigns every entity a semantic type. Crossing the two asks a question neither answers alone:
is the English default spread evenly over the inventory, or concentrated in particular kinds of
entity?

We pool the three languages and read the *base* checkpoint, so the result describes the
pretrained model rather than anything our training did. Significance is a permutation test on
type labels; intervals are item bootstraps.

    PYTHONPATH=src:src/culture/analysis/v2 python lure_by_type.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from common import OUT, dump

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval",
)
LANGS = ("zh", "hi", "ar")
TYPE_LABEL = {
    # One head noun each; the caption gives the scope. "A & B" labels read as
    # codebook internals rather than as categories.
    "body_mind": "Body",            # body parts and the mind
    "artefact_household": "Objects",  # artefacts, tools, household things
    "nature_cosmos": "Nature",      # landscape, weather, sky
    "animal": "Animals",
    "kinship_social": "People",     # kin and social roles
    "food_drink": "Food",           # food and drink
    "religion_supernatural": "Religion",
    "occupation_economy": "Economy",  # money and work
    "abstract_other": "Abstract",
}


def load_types():
    types = {}
    for lang in ("en", "zh", "hi", "ar"):
        p = os.path.join(OUT, f"entity_typology_labels_{lang}.json")
        if not os.path.exists(p):
            continue
        lab = json.load(open(p, encoding="utf-8"))
        if lang == "en":
            for e, o in lab.items():
                types.setdefault(e, o["type"])
        else:
            tr = os.path.join(OUT, f"entity_translations_{lang}_en.json")
            if not os.path.exists(tr):
                continue
            for e, t in json.load(open(tr, encoding="utf-8")).items():
                if t and t.get("en") and t["en"] != "none" and e in lab:
                    types.setdefault(t["en"], lab[e]["type"])
    return types


def boot_ci(x, reps=10000, seed=0):
    x = np.asarray(x, float)
    if x.size < 3:
        return None
    rng = np.random.default_rng(seed)
    s = [float(np.mean(rng.choice(x, x.size, replace=True))) for _ in range(reps)]
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="base")
    ap.add_argument("--out", default="lure_by_type.json")
    args = ap.parse_args()

    types = load_types()
    rows = []  # (lang, entity, type, lured, correct)
    for lang in LANGS:
        p = f"{EVAL}/{lang}/{args.arm}/symbolism_{lang}.json"
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8"))["records"]:
            e, lu = r.get("entity_en"), r.get("lure")
            t = types.get(e)
            if not e or lu is None or not t:
                continue
            pred = r.get("pred_norm", r.get("pred"))
            rows.append(
                (lang, e, t, int(pred == lu),
                 int(r.get("correct_norm", r.get("correct", 0))))
            )

    by = defaultdict(list)
    for _, _, t, lured, _ in rows:
        by[t].append(lured)
    kept = {t: v for t, v in by.items() if len(v) >= 8}

    overall = float(np.mean([r[3] for r in rows]))
    summary = {
        TYPE_LABEL.get(t, t): {
            "n_items": len(v),
            "lure_rate": round(float(np.mean(v)), 4),
            "ci95": boot_ci(v),
        }
        for t, v in sorted(kept.items(), key=lambda x: -float(np.mean(x[1])))
    }

    # Permutation test: is the spread across types larger than chance?
    labels = [r[2] for r in rows if r[2] in kept]
    vals = np.array([r[3] for r in rows if r[2] in kept], float)
    groups = sorted(set(labels))
    idx = {g: np.array([i for i, l in enumerate(labels) if l == g]) for g in groups}
    obs = float(np.std([vals[idx[g]].mean() for g in groups]))
    rng = np.random.default_rng(0)
    cnt = 0
    for _ in range(10000):
        perm = rng.permutation(vals)
        if float(np.std([perm[idx[g]].mean() for g in groups])) >= obs:
            cnt += 1
    p_perm = round((cnt + 1) / 10001, 5)

    out = {
        "method": __doc__,
        "arm": args.arm,
        "languages": list(LANGS),
        "n_items": len(rows),
        "overall_lure_rate": round(overall, 4),
        "spread_across_types_p_perm": p_perm,
        "by_type": summary,
    }
    print("wrote", dump(out, args.out))
    print(f"arm={args.arm} n={len(rows)} overall lure={overall:.3f} "
          f"type-spread p={p_perm}")
    for t, v in summary.items():
        print(f"  {t:20s} n={v['n_items']:3d} lure={v['lure_rate']:.3f} {v['ci95']}")


if __name__ == "__main__":
    main()
