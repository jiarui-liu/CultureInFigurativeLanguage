#!/usr/bin/env python3
"""Statistics of the full-knowledge-base audit (labels from full_kb_audit_judge.py).

  * per language (and per Arabic register / source): rates of each meaning label, figurative rate and
    entity precision for each judge, for entries where both judges agree, with Wilson 95% intervals
  * agreement between the two judges (Cohen's kappa; 4-way and the paper's 3-way meaning label)
  * agreement of each judge with the earlier 200-entry Claude audit on the same entries
  * a per-entry quality flag (both judges: meaning wrong, or both: not figurative) written to
    <audit>/quality_flags.jsonl for release
  * the Section 4 language-by-entity-type Cramer's V recomputed without flagged entries
  * when native-speaker workbooks exist in docs/arr_oct_rebuttal/annotation/returned/: each judge's
    agreement with people on the shared entries, its confusion matrix, and a human-calibrated
    full-KB estimate of the share of acceptable meanings (Rogan-Gladen correction, bootstrap CI)
Usage: python -m culture.analysis.rebuttal.full_kb_audit --out docs/paper_stats/rebuttal/full_kb_audit.json
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
A = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir/audit/full")
MANUAL = REPO / "docs/paper_stats/v5/manual"
ANN = REPO / "docs/arr_oct_rebuttal/annotation"
JUDGES = ["qwen27b", "gemma"]
LANGS = ["en", "zh", "hi", "ar"]
M4 = ["correct", "correct_with_noise", "partly_correct", "wrong"]
TO3 = {"correct": "clean", "correct_with_noise": "noisy", "partly_correct": "noisy", "wrong": "wrong"}


def wilson(k, n):
    if not n:
        return None
    p, z = k / n, 1.96
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1)]


def kappa(a, b):
    pr = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pr) < 5:
        return None
    n = len(pr); po = sum(x == y for x, y in pr) / n
    ca, cb = Counter(x for x, _ in pr), Counter(y for _, y in pr)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / n / n
    return {"kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0, "agree": round(po, 3), "n": n}


def norm_meaning(v):
    v = str(v or "").strip().lower().replace(" ", "_").replace("-", "_")
    return v if v in M4 else None


def norm_yes(v):
    v = str(v or "").strip().lower()
    return "yes" if v.startswith("y") else ("no" if v.startswith("n") else None)


def register(e):
    v = " ".join(e.get("varieties", [])).lower()
    return "classical" if "classical" in v else ("msa" if "modern standard" in v or v.strip() == "msa" else "colloquial")


def summarize(rows, J):
    """rows: entries; J: judge -> id -> judgment. Returns per-judge and both-agree rates."""
    out = {}
    for j in JUDGES:
        lab = [J[j].get(e["id"]) for e in rows]
        lab = [x for x in lab if x]
        m = Counter(norm_meaning(x.get("meaning")) for x in lab)
        n = sum(v for k, v in m.items() if k)
        fig = Counter(norm_yes(x.get("is_figurative")) for x in lab)
        nl = nw = 0
        for e in rows:
            x = J[j].get(e["id"])
            if not x:
                continue
            listed = set(e["ent"]); wrong = {str(w) for w in (x.get("ent_wrong") or []) if isinstance(w, (str, int))}
            nl += len(listed); nw += len(listed & wrong)
        out[j] = {"n": n, **{k: wilson(m[k], n) for k in M4},
                  "acceptable(correct+noise)": wilson(m["correct"] + m["correct_with_noise"], n),
                  "is_figurative": wilson(fig["yes"], fig["yes"] + fig["no"]),
                  "entity_precision": wilson(nl - nw, nl)}
    both = [(norm_meaning(J["qwen27b"].get(e["id"], {}).get("meaning")), norm_meaning(J["gemma"].get(e["id"], {}).get("meaning")))
            for e in rows]
    both = [(a, b) for a, b in both if a and b]
    out["judge_kappa_meaning4"] = kappa([a for a, _ in both], [b for _, b in both])
    out["judge_kappa_meaning3"] = kappa([TO3[a] for a, _ in both], [TO3[b] for _, b in both])
    fb = [(norm_yes(J["qwen27b"].get(e["id"], {}).get("is_figurative")), norm_yes(J["gemma"].get(e["id"], {}).get("is_figurative"))) for e in rows]
    out["judge_kappa_figurative"] = kappa([a for a, _ in fb], [b for _, b in fb])
    agree = [a for a, b in both if TO3[a] == TO3[b]]
    out["both_agree"] = {"n": len(agree), "share_of_entries": wilson(len(agree), len(both)),
                         **{k: wilson(sum(TO3[a] == k for a in agree), len(agree)) for k in ["clean", "noisy", "wrong"]}}
    return out


def rogan_gladen(p_obs, se, sp):
    d = se + sp - 1
    return None if d <= 0 else min(1.0, max(0.0, (p_obs + sp - 1) / d))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    entries = [json.loads(l) for l in open(A / "entries.jsonl", encoding="utf-8")]
    J = {j: {} for j in JUDGES}
    for j in JUDGES:
        for fn in glob.glob(str(A / f"judge_{j}_*.jsonl")):
            for l in open(fn, encoding="utf-8"):
                x = json.loads(l); J[j][x["id"]] = x["judgment"]
    res = {"coverage": {j: {L: sum(1 for e in entries if e["lang"] == L and e["id"] in J[j]) for L in LANGS} for j in JUDGES},
           "kb_size": dict(Counter(e["lang"] for e in entries)), "per_lang": {}, "ar_by_register": {}, "ar_by_source": {}}
    for L in LANGS:
        res["per_lang"][L] = summarize([e for e in entries if e["lang"] == L], J)
    ar = [e for e in entries if e["lang"] == "ar"]
    for g in ["classical", "colloquial", "msa"]:
        rows = [e for e in ar if register(e) == g]
        if rows:
            res["ar_by_register"][g] = {"n_entries": len(rows), **summarize(rows, J)}
    src = defaultdict(list)
    for e in ar:
        for s in e.get("sources") or ["unknown"]:
            src[s].append(e)
    for s, rows in src.items():
        res["ar_by_source"][s] = {"n_entries": len(rows), **summarize(rows, J)}

    # quality flags
    flags = {}
    with open(A / "quality_flags.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            q, g = J["qwen27b"].get(e["id"]), J["gemma"].get(e["id"])
            if not (q and g):
                continue
            fl = []
            if norm_meaning(q.get("meaning")) == "wrong" and norm_meaning(g.get("meaning")) == "wrong":
                fl.append("meaning_wrong")
            if norm_yes(q.get("is_figurative")) == "no" and norm_yes(g.get("is_figurative")) == "no":
                fl.append("not_figurative")
            flags[e["id"]] = fl
            f.write(json.dumps({"id": e["id"], "lang": e["lang"], "idiom": e["idiom"], "flags": fl,
                                "qwen27b": q, "gemma": g}, ensure_ascii=False) + "\n")
    res["quality_flags"] = {L: {k: wilson(sum(k in flags[e["id"]] for e in entries if e["lang"] == L and e["id"] in flags),
                                          sum(1 for e in entries if e["lang"] == L and e["id"] in flags))
                                for k in ["meaning_wrong", "not_figurative"]} for L in LANGS}

    # agreement with the earlier 200-entry Claude audit (joined by idiom string)
    c3 = {"yes": "clean", "partly": "noisy", "no": "wrong"}
    byidiom = {(e["lang"], e["idiom"]): e["id"] for e in entries}
    res["vs_claude_audit_200"] = {}
    for L in LANGS:
        sheet = {json.loads(l)["id"]: json.loads(l) for l in open(MANUAL / f"kb_sheet_{L}.jsonl", encoding="utf-8")}
        man = [json.loads(l) for l in open(MANUAL / f"kb_manual_{L}.jsonl", encoding="utf-8")]
        r = {}
        for j in JUDGES:
            pairs = [(c3.get(m.get("fig_ok")), TO3.get(norm_meaning(J[j].get(byidiom.get((L, sheet[m["id"]]["idiom"].strip())), {}).get("meaning"))))
                     for m in man if m["id"] in sheet]
            r[j] = kappa([x for x, _ in pairs], [y for _, y in pairs])
            fp = [(m.get("is_figurative"), norm_yes(J[j].get(byidiom.get((L, sheet[m["id"]]["idiom"].strip())), {}).get("is_figurative")))
                  for m in man if m["id"] in sheet]
            r[j + "_figurative"] = kappa([x for x, _ in fp], [y for _, y in fp])
        res["vs_claude_audit_200"][L] = r

    # Cramer's V without flagged entries
    try:
        sys.path.insert(0, str(REPO / "src/culture/analysis/v2"))
        import entity_typology as et
        from common import load_kb, norm_entity
        drop = {(e["lang"], e["idiom"]) for e in entries if flags.get(e["id"])}
        labels = {}
        cnt = {"all": {}, "unflagged": {}}
        for L in et.LANGS:
            lab = json.load(open(REPO / f"docs/paper_stats/analysis_v2/entity_typology_labels_{L}.json"))
            labels[L] = {k: v["type"] for k, v in lab.items() if v.get("type")}
            ca, cu = Counter(), Counter()
            for r in load_kb(L):
                s = {n for n in (norm_entity(x, L) for x in r["raw_entities"]) if n}
                ca.update(s)
                if (L, str(r["idiom"] or "").strip()) not in drop:
                    cu.update(s)
            cnt["all"][L], cnt["unflagged"][L] = ca, cu
        res["entity_type_cramers_v"] = {k: {kk: v for kk, v in et.stats_block(et.contingency(c, labels)).items()
                                            if kk in ("cramers_v", "n_mentions", "pairwise_cramers_v", "p")}
                                        for k, c in cnt.items()}
    except Exception as ex:  # keep the rest of the report
        res["entity_type_cramers_v"] = {"error": repr(ex)[:300]}

    # human calibration, when workbooks are back
    try:
        from culture.analysis.rebuttal.score_annotations import read
        hum = {}
        for L in ["zh", "hi", "ar"]:
            fns = sorted(glob.glob(str(ANN / "returned" / f"IdiomAtlas_validation_{L}_*.xlsx")))
            if not fns:
                continue
            key = json.load(open(ANN / f"KEY_DO_NOT_SHARE_{L}.json"))
            rows = read(fns[0])["A_KB"]
            hl = [(k["idiom"], norm_meaning(r.get("A1_meaning_correct")), norm_yes(r.get("A2_is_figurative")))
                  for r, k in zip(rows, key["A_KB"])]
            hum[L] = {}
            for j in JUDGES:
                jl = [(TO3.get(m), TO3.get(norm_meaning(J[j].get(byidiom.get((L, idm.strip())), {}).get("meaning"))))
                      for idm, m, _ in hl if m]
                jl = [(h, x) for h, x in jl if x]
                if len(jl) < 10:
                    continue
                conf = {h: dict(Counter(x for hh, x in jl if hh == h)) for h in ["clean", "noisy", "wrong"]}
                # binary acceptable (clean or noisy) for Rogan-Gladen
                hb = np.array([h != "wrong" for h, _ in jl]); xb = np.array([x != "wrong" for _, x in jl])
                se = xb[hb].mean() if hb.any() else float("nan"); sp = (~xb[~hb]).mean() if (~hb).any() else 1.0
                allj = [norm_meaning(J[j][e["id"]].get("meaning")) for e in entries if e["lang"] == L and e["id"] in J[j]]
                p_obs = np.mean([TO3.get(x) != "wrong" for x in allj if x])
                rng = np.random.default_rng(0); bs = []
                for _ in range(2000):
                    i = rng.integers(0, len(jl), len(jl)); h2, x2 = hb[i], xb[i]
                    s2 = x2[h2].mean() if h2.any() else np.nan; p2 = (~x2[~h2]).mean() if (~h2).any() else 1.0
                    v = rogan_gladen(p_obs, s2, p2)
                    if v is not None and not np.isnan(v):
                        bs.append(v)
                est = rogan_gladen(p_obs, se, sp)
                hum[L][j] = {"kappa_meaning3": kappa([h for h, _ in jl], [x for _, x in jl]),
                             "confusion_human_rows": conf, "sensitivity_acceptable": round(float(se), 3),
                             "specificity_wrong": round(float(sp), 3),
                             "full_kb_judge_acceptable": round(100 * float(p_obs), 1),
                             "full_kb_calibrated_acceptable": None if est is None else round(100 * est, 1),
                             "calibrated_ci95": [round(100 * float(np.percentile(bs, 2.5)), 1), round(100 * float(np.percentile(bs, 97.5)), 1)] if bs else None,
                             "human_sample_acceptable": wilson(int(hb.sum()), len(hb))}
        res["human_calibration"] = hum or "no returned workbooks yet"
    except Exception as ex:
        res["human_calibration"] = {"error": repr(ex)[:300]}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ["coverage", "quality_flags"]}, indent=1))
    for L in LANGS:
        p = res["per_lang"][L]
        print(L, {j: (p[j]["n"], p[j]["acceptable(correct+noise)"], p[j]["wrong"], p[j]["is_figurative"]) for j in JUDGES},
              p["judge_kappa_meaning3"])


if __name__ == "__main__":
    main()
