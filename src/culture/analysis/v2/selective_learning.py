"""Is the seen/unseen gap evidence of memorisation, or of item difficulty?

The paper reads the IdiomAtlas-MC seen/unseen split as evidence that idiom-centered training
teaches the idioms it was shown and not others. That reading is only safe if the two splits are
otherwise comparable. They are not guaranteed to be: "seen" idioms are by construction those that
occur in web text, so they are more frequent and more conventional than "unseen" ones, and a
model may do better on them without having been trained on anything.

The base checkpoint settles it. Any seen/unseen gap it shows is pure item difficulty, since it
never saw the training corpus. We therefore report, for every arm, the raw gap and the
difference-in-differences against base -- the part of the gap that training actually created --
each with an item-level bootstrap interval.

    PYTHONPATH=src:src/culture/analysis/v2 python selective_learning.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import dump

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval",
)
ARMS = {"base": "base", "random": "unfiltered", "idiom_cpt": "cpt",
        "idiom_untagged": "untagged", "culture": "culture",
        "culture_notes": "culturenotes"}
LANGS = ("zh", "hi", "ar")

# Point estimates for the forward-direction Idiom-CPT arm, whose per-item records live on the
# other cluster; from docs/paper_stats/v2/idiomatlas_9b.json. Used for the DiD point estimate
# only -- no interval, since we cannot resample items we do not have.
PUBLISHED = {
    "ar": {"base": (42.0, 37.8), "cpt": (68.3, 32.5), "untagged": (46.7, 40.0),
           "unfiltered": (43.2, 36.8)},
    "hi": {"base": (36.2, 38.8), "cpt": (75.8, 53.8), "untagged": (56.7, 53.3),
           "unfiltered": (50.5, 52.3)},
    "zh": {"base": (79.0, 52.6), "cpt": (90.5, 61.5), "unfiltered": (79.8, 57.7)},
}


def load_split(lang, arm_dir, split):
    p = f"{EVAL}/{lang}/{arm_dir}/idiomatlas_mc_{lang}_{split}.json"
    if not os.path.exists(p):
        return None
    return np.array(
        [
            int(r.get("correct_norm", r.get("correct", 0)))
            for r in json.load(open(p, encoding="utf-8"))["records"]
        ],
        float,
    )


def boot_gap(seen, unseen, reps=10000, seed=0):
    """Bootstrap the seen-minus-unseen gap, resampling items within each split."""
    rng = np.random.default_rng(seed)
    s = np.array(
        [
            rng.choice(seen, seen.size, replace=True).mean()
            - rng.choice(unseen, unseen.size, replace=True).mean()
            for _ in range(reps)
        ]
    )
    return float(np.mean(seen) - np.mean(unseen)), [
        float(np.percentile(s, 2.5)),
        float(np.percentile(s, 97.5)),
    ]


def boot_did(a_s, a_u, b_s, b_u, reps=10000, seed=0):
    """Bootstrap (arm gap - base gap). Items are paired across arms, so resample indices once."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        i = rng.integers(0, a_s.size, a_s.size)
        j = rng.integers(0, a_u.size, a_u.size)
        out.append(
            (a_s[i].mean() - a_u[j].mean()) - (b_s[i].mean() - b_u[j].mean())
        )
    out = np.array(out)
    did = (a_s.mean() - a_u.mean()) - (b_s.mean() - b_u.mean())
    return float(did), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="selective_learning.json")
    args = ap.parse_args()

    res = {"method": __doc__, "languages": {}}
    for lang in LANGS:
        base_s, base_u = load_split(lang, "base", "seen"), load_split(lang, "base", "unseen")
        if base_s is None or base_u is None:
            continue
        L = {"n_seen": int(base_s.size), "n_unseen": int(base_u.size), "arms": {}}
        bg, bci = boot_gap(base_s, base_u)
        L["base_gap"] = {"gap": round(bg * 100, 2),
                         "ci95": [round(c * 100, 2) for c in bci]}

        for arm, d in ARMS.items():
            s, u = load_split(lang, d, "seen"), load_split(lang, d, "unseen")
            if s is None or u is None:
                continue
            g, gci = boot_gap(s, u)
            e = {
                "seen": round(float(s.mean()) * 100, 2),
                "unseen": round(float(u.mean()) * 100, 2),
                "gap": round(g * 100, 2),
                "gap_ci95": [round(c * 100, 2) for c in gci],
            }
            if arm != "base" and s.size == base_s.size and u.size == base_u.size:
                did, dci = boot_did(s, u, base_s, base_u)
                e["gap_minus_base_gap"] = round(did * 100, 2)
                e["gap_minus_base_gap_ci95"] = [round(c * 100, 2) for c in dci]
            L["arms"][arm] = e

        # Point-estimate DiD for every published arm, including Idiom-CPT.
        pub = PUBLISHED.get(lang, {})
        if "base" in pub:
            bs, bu = pub["base"]
            L["published_did"] = {
                a: round((v[0] - v[1]) - (bs - bu), 2)
                for a, v in pub.items() if a != "base"
            }
            L["published_gaps"] = {a: round(v[0] - v[1], 2) for a, v in pub.items()}
        res["languages"][lang] = L

    print("wrote", dump(res, args.out))
    for lang, L in res["languages"].items():
        print(f"\n[{lang}] n_seen={L['n_seen']} n_unseen={L['n_unseen']}  "
              f"base gap={L['base_gap']['gap']:+.1f} {L['base_gap']['ci95']}")
        for a, e in L["arms"].items():
            did = e.get("gap_minus_base_gap")
            tail = (f"  DiD={did:+.1f} {e.get('gap_minus_base_gap_ci95')}"
                    if did is not None else "")
            print(f"   {a:16s} seen={e['seen']:5.1f} unseen={e['unseen']:5.1f} "
                  f"gap={e['gap']:+6.1f}{tail}")
        if "published_did" in L:
            print(f"   published gaps:  {L['published_gaps']}")
            print(f"   published DiD vs base: {L['published_did']}")


if __name__ == "__main__":
    main()
