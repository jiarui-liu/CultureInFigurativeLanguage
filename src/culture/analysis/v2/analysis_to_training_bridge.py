"""Does the Section-4 analysis predict the Section-5 training outcome?

Sections 4 and 5 of the paper currently stand side by side without touching: Section 4 measures
how far apart two languages place the same entity, Section 5 measures what continued pretraining
changes, and nothing links them. This script links them, entity by entity.

The symbolism probe stores, for every item, the entity it is about and whether the model chose
the English-associated distractor (the "lure"). Section 4 gives, for many of the same entities, a
prompt-independent cross-language divergence and an LLM count of shared vs culture-specific
aspects. Joining the two answers three questions:

  1. Are entities that diverge more across languages the ones on which a model defaults to the
     English association?
  2. Does idiom-centered training help more on exactly those entities?
  3. Which semantic types (animals, body parts, ...) drive both effects?

Everything here reads already-computed artefacts; no LLM or GPU is needed.

    PYTHONPATH=src:src/culture/analysis/v2 python analysis_to_training_bridge.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from common import OUT, dump

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval",
)
# Arm -> directory name, using the paper's vocabulary.
ARMS = {
    "base": "base",
    "random": "unfiltered",
    "idiom_cpt": "cpt",
    "idiom_untagged": "untagged",
    "culture": "culture",
}
LANGS = ("zh", "hi", "ar")


def boot_ci(x, fn=np.mean, reps=5000, seed=0):
    x = np.asarray(x, float)
    if x.size < 3:
        return None
    rng = np.random.default_rng(seed)
    s = np.array([fn(rng.choice(x, x.size, replace=True)) for _ in range(reps)])
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def _rank(x):
    """Average ranks, so that ties do not get an arbitrary order."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 4:
        return None
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_p(a, b, reps=10000, seed=0):
    """Two-sided permutation test on the Spearman correlation."""
    r0 = spearman(a, b)
    if r0 is None:
        return None, None
    rng = np.random.default_rng(seed)
    b = np.asarray(b, float)
    cnt = 0
    for _ in range(reps):
        if abs(spearman(a, rng.permutation(b))) >= abs(r0):
            cnt += 1
    return round(r0, 4), round((cnt + 1) / (reps + 1), 5)


def load_divergence():
    p = os.path.join(OUT, "entity_divergence_en_zh_per_entity.csv")
    div, shared, uniq = {}, {}, {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        e = r["entity_en"]
        if r["gloss_centroid_div"]:
            div[e] = float(r["gloss_centroid_div"])
        if r["llm_shared"]:
            s = int(r["llm_shared"])
            u = int(r["llm_en_unique"]) + int(r["llm_zh_unique"])
            shared[e] = s
            uniq[e] = u
    return div, shared, uniq


def load_types():
    """english-anchored entity -> semantic type, via each language's label file."""
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


def load_symbolism(lang, arm_dir):
    """entity -> per-item outcomes. One probe item per entity, so each value is a scalar.

    lure_margin = normalized logprob(gold) - logprob(English-associated option). Positive means
    the model prefers the culture-specific reading; this is a far better-powered outcome than the
    binary pick, because the probe has only one item per entity.
    """
    p = f"{EVAL}/{lang}/{arm_dir}/symbolism_{lang}.json"
    if not os.path.exists(p):
        return None
    recs = json.load(open(p, encoding="utf-8"))["records"]
    out = {}
    for r in recs:
        e = r.get("entity_en")
        lp = r.get("logprobs_norm") or r.get("logprobs")
        g, lu = r.get("gold"), r.get("lure")
        if not e or lp is None or g is None or lu is None:
            continue
        if not (0 <= g < len(lp) and 0 <= lu < len(lp)):
            continue
        pred = r.get("pred_norm", r.get("pred"))
        out[e] = {
            "n": 1,
            "correct": int(r.get("correct_norm", r.get("correct", 0))),
            "lure": int(pred == lu),
            "lure_margin": float(lp[g] - lp[lu]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis_to_training_bridge.json")
    args = ap.parse_args()

    div, shared, uniq = load_divergence()
    types = load_types()
    res = {"method": __doc__, "n_entities_with_divergence": len(div),
           "n_entities_typed": len(types), "languages": {}}

    for lang in LANGS:
        arms = {k: load_symbolism(lang, v) for k, v in ARMS.items()}
        arms = {k: v for k, v in arms.items() if v}
        if "random" not in arms:
            continue
        L = {"arms_present": sorted(arms)}

        # --- Q1: divergence vs resistance to the English default ------------
        ents = [e for e in arms["random"] if e in div]
        if len(ents) >= 15:
            d = [div[e] for e in ents]
            marg = [arms["random"][e]["lure_margin"] for e in ents]
            lure = [float(arms["random"][e]["lure"]) for e in ents]
            acc = [float(arms["random"][e]["correct"]) for e in ents]
            r_m, p_m = perm_p(d, marg)
            r_l, p_l = perm_p(d, lure)
            r_a, p_a = perm_p(d, acc)
            L["divergence_vs_lure_margin"] = {"n": len(ents), "spearman": r_m,
                                              "p_perm": p_m,
                                              "note": "negative rho = more divergent entities "
                                                      "are harder (smaller gold-lure margin)"}
            L["divergence_vs_lure_pick"] = {"n": len(ents), "spearman": r_l, "p_perm": p_l}
            L["divergence_vs_accuracy"] = {"n": len(ents), "spearman": r_a, "p_perm": p_a}

            # Tertile contrast on the continuous margin.
            q1, q2 = np.percentile(d, [33.3, 66.7])
            tert = defaultdict(list)
            for e, dv in zip(ents, d):
                b = "low" if dv <= q1 else ("high" if dv > q2 else "mid")
                tert[b].append(arms["random"][e]["lure_margin"])
            L["lure_margin_by_divergence_tertile"] = {
                k: {"n": len(v), "margin": round(float(np.mean(v)), 4), "ci95": boot_ci(v)}
                for k, v in tert.items()
            }

        # --- Q2: does training help most where divergence is highest? -------
        for arm in ("idiom_cpt", "idiom_untagged", "culture"):
            if arm not in arms:
                continue
            ents2 = [e for e in arms[arm] if e in div and e in arms["random"]]
            if len(ents2) < 15:
                continue
            d = [div[e] for e in ents2]
            dm = [
                arms[arm][e]["lure_margin"] - arms["random"][e]["lure_margin"]
                for e in ents2
            ]
            r, p = perm_p(d, dm)
            L[f"divergence_vs_margin_change_{arm}"] = {
                "n": len(ents2), "spearman": r, "p_perm": p,
                "mean_margin_change": round(float(np.mean(dm)), 4),
                "ci95": boot_ci(dm),
            }

        # --- Q3: by semantic type -------------------------------------------
        by_t = defaultdict(lambda: {"marg": [], "lure": [], "acc": [], "div": []})
        for e, v in arms["random"].items():
            t = types.get(e)
            if not t:
                continue
            by_t[t]["marg"].append(v["lure_margin"])
            by_t[t]["lure"].append(float(v["lure"]))
            by_t[t]["acc"].append(float(v["correct"]))
            if e in div:
                by_t[t]["div"].append(div[e])
        L["by_semantic_type"] = {
            t: {
                "n_entities": len(v["marg"]),
                "lure_margin": round(float(np.mean(v["marg"])), 4),
                "lure_margin_ci95": boot_ci(v["marg"]),
                "lure_rate": round(float(np.mean(v["lure"])), 4),
                "accuracy": round(float(np.mean(v["acc"])), 4),
                "divergence": round(float(np.mean(v["div"])), 4) if v["div"] else None,
                "n_with_divergence": len(v["div"]),
            }
            for t, v in sorted(by_t.items())
            if len(v["marg"]) >= 3
        }
        res["languages"][lang] = L

    # --- shared vs unique aspects as an alternative predictor (zh only) -----
    sym = load_symbolism("zh", "unfiltered")
    if sym:
        ents = [e for e in sym if e in uniq]
        if len(ents) >= 15:
            u = [uniq[e] for e in ents]
            marg = [sym[e]["lure_margin"] for e in ents]
            r, p = perm_p(u, marg)
            res["zh_unique_aspects_vs_lure_margin"] = {
                "n": len(ents), "spearman": r, "p_perm": p
            }

    print("wrote", dump(res, args.out))
    for lang, L in res["languages"].items():
        print(f"\n[{lang}] arms={L['arms_present']}")
        for k in ("divergence_vs_lure_margin", "divergence_vs_lure_pick",
                  "divergence_vs_accuracy"):
            if k in L:
                print(f"  {k}: rho={L[k]['spearman']} p={L[k]['p_perm']} n={L[k]['n']}")
        if "lure_margin_by_divergence_tertile" in L:
            for b in ("low", "mid", "high"):
                v = L["lure_margin_by_divergence_tertile"].get(b)
                if v:
                    print(f"    margin[{b:4s}] = {v['margin']:+.3f} {v['ci95']} (n={v['n']})")
        for k in list(L):
            if k.startswith("divergence_vs_margin_change_"):
                print(f"  {k}: rho={L[k]['spearman']} p={L[k]['p_perm']} "
                      f"mean={L[k]['mean_margin_change']:+.4f} {L[k]['ci95']}")
    if "zh_unique_aspects_vs_lure_margin" in res:
        v = res["zh_unique_aspects_vs_lure_margin"]
        print(f"\n[zh] unique-aspect count vs margin: rho={v['spearman']} p={v['p_perm']}")


if __name__ == "__main__":
    main()
