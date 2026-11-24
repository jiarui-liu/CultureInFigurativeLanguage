#!/usr/bin/env python3
"""Score returned native-speaker workbooks (from build_annotation_sheets.py).

Put returned files in <ann_dir>/returned/ named IdiomAtlas_validation_<lang>_<annotator>.xlsx
(annotator = any short tag; the one with more filled rows is treated as primary).

Reports per language:
  A_KB   meaning correct / noisy / partly / wrong, figurative rate, entity precision, familiarity,
         Arabic region spread, all with Wilson 95% CIs; Cohen's kappa between annotators and
         between the person and each earlier automatic audit (Claude audit, Gemma, Qwen judges).
  B_MC   key shown: share of keys judged correct (seen / unseen), share with another acceptable
         option, item validity, and model contrasts restricted to human-validated items.
  C_Sym  gloss errors, human agreement with the key, gold recognized, English lure recognized in
         the target culture, item validity, and probe contrasts on human-validated items.
Usage: python -m culture.analysis.rebuttal.score_annotations --ann_dir docs/arr_oct_rebuttal/annotation \
         --out docs/arr_oct_rebuttal/annotation/human_validation_stats.json
"""
import argparse
import glob
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl

REPO = Path(__file__).resolve().parents[4]
MANUAL = REPO / "docs/paper_stats/v5/manual"
AUDIT = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir/audit")
EVAL = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir/eval9b")
RUN = {"unfiltered": "Random-CPT", "untagged": "Idiom-Text", "cpt": "Idiom+Meaning", "base": "Base"}


def wilson(k, n):
    if n == 0:
        return [None, None, None]
    p = k / n; z = 1.96
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * p, 1), round(100 * (c - h), 1), round(100 * (c + h), 1)]


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x not in (None, "") and y not in (None, "")]
    if len(pairs) < 5:
        return None
    labs = sorted({x for p in pairs for x in p}); n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[l] * cb[l] for l in labs) / n / n
    return {"kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0, "agree": round(po, 3), "n": n}


def letters(s):
    s = str(s or "").upper()
    return set() if "NONE" in s else set(re.findall(r"[ABCD]", s))


def read(fn):
    wb = openpyxl.load_workbook(fn, data_only=True)
    out = {}
    for s in ["A_KB", "B_MC", "C_Symbolism"]:
        ws = wb[s]
        head = [c.value for c in ws[1]]
        out[s] = [dict(zip(head, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
    return out


def filled(rows, col):
    return sum(1 for r in rows if r.get(col) not in (None, ""))


def meaning3(v):  # paper's three-way label
    return {"correct": "clean", "correct_with_noise": "noisy", "partly_correct": "noisy", "wrong": "wrong"}.get(v)


def boot_delta(a, b, rng, B=10000):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    if len(d) == 0:
        return None
    idx = rng.integers(0, len(d), size=(B, len(d))); bs = d[idx].mean(1)
    return [round(100 * d.mean(), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


def model_contrasts(lang, files, keep, rng):
    """keep: set of qids. files: task file stems. Returns Idiom-Text/Idiom+Meaning minus Random-CPT."""
    out = {}
    for stem in files:
        recs = {}
        for r in RUN:
            p = EVAL / lang / r / f"{stem}.json"
            if p.exists():
                j = json.load(open(p)); mode = j.get("metrics", {}).get("score_mode", "continuation")
                k = "correct_norm" if mode == "continuation" else "correct"
                recs[r] = {x["qid"]: int(x[k]) for x in j["records"]}
        if "unfiltered" not in recs:
            continue
        q = sorted(keep & set.intersection(*[set(v) for v in recs.values()]))
        out[stem] = {"n": len(q)}
        for t in ["untagged", "cpt"]:
            if t in recs:
                out[stem][f"{RUN[t]} - Random-CPT"] = boot_delta([recs[t][x] for x in q], [recs["unfiltered"][x] for x in q], rng)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ann_dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rng = np.random.default_rng(7)
    res = {}
    for lang in ["zh", "hi", "ar"]:
        fns = sorted(glob.glob(os.path.join(a.ann_dir, "returned", f"IdiomAtlas_validation_{lang}_*.xlsx")))
        if not fns:
            continue
        key = json.load(open(os.path.join(a.ann_dir, f"KEY_DO_NOT_SHARE_{lang}.json")))
        ann = {Path(f).stem.split(f"_{lang}_")[-1]: read(f) for f in fns}
        order = sorted(ann, key=lambda k: -filled(ann[k]["A_KB"], "A1_meaning_correct"))
        P = ann[order[0]]; R = {"annotators": order}

        # ---------- A_KB
        A = [r for r in P["A_KB"] if r.get("A1_meaning_correct")]
        n = len(A); m = Counter(r["A1_meaning_correct"] for r in A)
        ent_listed = ent_wrong = 0
        for r in A:
            listed = [e.strip() for e in str(r.get("entities") or "").split("|") if e.strip()]
            wrong = [e for e in re.split(r"[|,،;；，]", str(r.get("A3_wrong_entities") or "")) if e.strip()]
            ent_listed += len(listed); ent_wrong += min(len(wrong), len(listed))
        RA = {"n": n,
              "meaning_correct_incl_noise": wilson(m["correct"] + m["correct_with_noise"], n),
              "meaning_clean": wilson(m["correct"], n), "meaning_noise": wilson(m["correct_with_noise"], n),
              "meaning_partly": wilson(m["partly_correct"], n), "meaning_wrong": wilson(m["wrong"], n),
              "is_figurative": wilson(sum(r.get("A2_is_figurative") == "yes" for r in A), filled(A, "A2_is_figurative")),
              "entity_precision": wilson(ent_listed - ent_wrong, ent_listed)}
        if lang == "ar":
            reg = Counter(r.get("A7_region") for r in A if r.get("A7_region"))
            RA["region"] = {k: wilson(v, sum(reg.values())) for k, v in reg.items()}
            byreg = {}
            for r, k in zip(P["A_KB"], key["A_KB"]):
                if r.get("A1_meaning_correct"):
                    g = "classical" if "classical" in k["register"].lower() else "colloquial"
                    byreg.setdefault(g, []).append(r["A1_meaning_correct"] in ("correct", "correct_with_noise"))
            RA["meaning_correct_by_register"] = {g: wilson(sum(v), len(v)) for g, v in byreg.items()}
        if lang == "hi":
            sp = Counter(r.get("A7_spelling_ok") for r in A if r.get("A7_spelling_ok"))
            RA["spelling"] = {k: wilson(v, sum(sp.values())) for k, v in sp.items()}
        # agreement: annotators, and people vs automatic audits on the same entries
        if len(order) > 1:
            S = ann[order[1]]["A_KB"]
            RA["iaa_meaning3"] = kappa([meaning3(r.get("A1_meaning_correct")) for r in P["A_KB"]], [meaning3(r.get("A1_meaning_correct")) for r in S])
            RA["iaa_figurative"] = kappa([r.get("A2_is_figurative") for r in P["A_KB"]], [r.get("A2_is_figurative") for r in S])
        ids = [k["audit_id"] for k in key["A_KB"]]
        human3 = [meaning3(r.get("A1_meaning_correct")) for r in P["A_KB"]]
        claude = {json.loads(l)["id"]: json.loads(l) for l in open(MANUAL / f"kb_manual_{lang}.jsonl")}
        c3 = {"yes": "clean", "partly": "noisy", "no": "wrong"}
        RA["human_vs_claude_audit"] = kappa(human3, [c3.get(claude.get(i, {}).get("fig_ok")) for i in ids])
        RA["human_vs_claude_figurative"] = kappa([r.get("A2_is_figurative") for r in P["A_KB"]], [claude.get(i, {}).get("is_figurative") for i in ids])
        for judge in ["gemma", "qwen27b"]:
            jj = [json.loads(l) for l in open(AUDIT / f"judge_{judge}.jsonl")]
            jl = {x["idiom"]: x["judgment"].get("fig_ok") for x in jj if x["lang"] == lang}
            RA[f"human_vs_{judge}"] = kappa(human3, [c3.get(jl.get(k["idiom"])) for k in key["A_KB"]])
        R["A_KB"] = RA

        # ---------- B_MC
        Bk = {k["row"]: k for k in key["B_MC"]}
        # key shown to the annotator: B1 = is the keyed option correct, B2 = other acceptable options
        B = [(r, Bk[r["row"]]) for r in P["B_MC"] if r.get("B1_key_correct")]
        judged = [(r, k) for r, k in B if r["B1_key_correct"] != "dont_know"]
        valid = set()
        for r, k in judged:
            others = letters(r.get("B2_other_acceptable")) - {k["gold"]}
            if r["B1_key_correct"] == "yes" and not others:
                valid.add(k["qid"])
        lab = Counter(r["B1_key_correct"] for r, _ in B)
        RB = {"n": len(B), "n_judged": len(judged), "labels": dict(lab),
              "key_correct": {s: wilson(sum(r["B1_key_correct"] == "yes" for r, k in judged if k["split"] == s),
                                        sum(k["split"] == s for _, k in judged)) for s in ["seen", "unseen"]},
              "key_correct_or_partly": wilson(sum(r["B1_key_correct"] in ("yes", "partly") for r, _ in judged), len(judged)),
              "another_option_acceptable": wilson(sum(bool(letters(r.get("B2_other_acceptable")) - {k["gold"]}) for r, k in judged), len(judged)),
              "valid_items": wilson(len(valid), len(judged))}
        RB["model_contrasts_valid_items"] = model_contrasts(lang, [f"idiomatlas_mc_{lang}_seen", f"idiomatlas_mc_{lang}_unseen"], valid, rng)
        RB["model_contrasts_all_sampled"] = model_contrasts(lang, [f"idiomatlas_mc_{lang}_seen", f"idiomatlas_mc_{lang}_unseen"], {k["qid"] for _, k in B}, rng)
        R["B_MC"] = RB

        # ---------- C_Symbolism
        Ck = {k["row"]: k for k in key["C_Symbolism"]}
        # key shown: C2 = is the keyed option recognized, C3 = other recognized options (lure not marked)
        C = [(r, Ck[r["row"]]) for r in P["C_Symbolism"] if r.get("C2_key_recognized")]
        judged = [(r, k) for r, k in C if r["C2_key_recognized"] != "dont_know"]
        validC = set()
        for r, k in judged:
            other = letters(r.get("C3_other_recognized")) - {k["gold"]}
            if r.get("C1_gloss_ok") != "no" and r["C2_key_recognized"] in ("yes", "partly") and k["english_lure"] not in other:
                validC.add(k["qid"])
        RC = {"n": len(C), "n_judged": len(judged), "labels": dict(Counter(r["C2_key_recognized"] for r, _ in C)),
              "gloss_wrong": wilson(sum(r.get("C1_gloss_ok") == "no" for r, _ in C), filled([r for r, _ in C], "C1_gloss_ok")),
              "key_recognized": wilson(sum(r["C2_key_recognized"] == "yes" for r, _ in judged), len(judged)),
              "key_recognized_or_partly": wilson(sum(r["C2_key_recognized"] in ("yes", "partly") for r, _ in judged), len(judged)),
              "lure_recognized_in_target": wilson(sum(k["english_lure"] in letters(r.get("C3_other_recognized")) for r, k in judged), len(judged)),
              "a_distractor_recognized": wilson(sum(bool(letters(r.get("C3_other_recognized")) - {k["gold"], k["english_lure"]}) for r, k in judged), len(judged)),
              "valid_items": wilson(len(validC), len(judged))}
        cl = {json.loads(l)["qid"]: json.loads(l) for l in open(MANUAL / f"probe_manual_{lang}.jsonl")}
        RC["human_valid_vs_claude_valid"] = kappa(
            [k["qid"] in validC for _, k in C],
            [(cl.get(k["qid"], {}).get("target_option") == k["gold"] and "entity_misread" not in cl.get(k["qid"], {}).get("issues", [])) for _, k in C])
        RC["model_contrasts_valid_items"] = model_contrasts(lang, [f"symbolism_v2_{lang}_letter"], validC, rng)
        RC["model_contrasts_all_sampled"] = model_contrasts(lang, [f"symbolism_v2_{lang}_letter"], {k["qid"] for _, k in C}, rng)
        R["C_Symbolism"] = RC
        res[lang] = R
        print(json.dumps({lang: R}, ensure_ascii=False, indent=1))
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
