#!/usr/bin/env python3
"""Symbolism probe with idiom evidence in the prompt (diagnostic variants).

The paper's probe (variant A) asks, zero-shot and without context, what an entity symbolizes in a
language's idioms, and scores the answer letter. Here the same question, options and letter scoring
are kept, and a block of target-language idioms is prepended:

  A  no context (the paper's probe; re-scored as a reproduction check)
  B  the evidence idioms the item was built from (up to 8), idioms only
  C  the same evidence idioms with their KB figurative meanings
  D  up to 8 KB idioms that mention the entity, sampled at random (not selected to support the
     gold option), with their meanings: does the model infer the symbolism from unselected evidence?

  build : write <mc>/symbolism_ctx{B,C,D}_{L}_letter.jsonl
  stats : accuracy, English-lure rate, arm contrasts per variant, and per-arm variant gains
"""
import argparse
import ast
import json
import random
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
B_ = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
MC = B_ / "eval_data/mc"
MAN = REPO / "docs/paper_stats/v5/manual"
KB = {
    "zh": REPO / "culture/data/idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl",
    "hi": REPO / "culture/data/idioms/hi/idioms_merged_llm_formatted_figurative_only.jsonl",
    "ar": REPO / "culture/data/idioms/ar/idioms_merged_llm_formatted.jsonl",
}
HEAD = {
    "zh": ("以下是一些含有“{e}”的汉语成语和俗语：", "以下是一些含有“{e}”的汉语成语和俗语及其释义："),
    "hi": ("\"{e}\" वाली कुछ हिंदी कहावतें और लोकोक्तियाँ:", "\"{e}\" वाली कुछ हिंदी कहावतें और लोकोक्तियाँ, उनके अर्थ सहित:"),
    "ar": ("فيما يلي بعض الأمثال والتعابير العربية التي تذكر «{e}»:",
           "فيما يلي بعض الأمثال والتعابير العربية التي تذكر «{e}» مع معانيها:"),
}
SEP = {"zh": "：", "hi": ": ", "ar": ": "}
RUNS = {"base": "Base", "unfiltered": "Random-CPT", "untagged": "Idiom-Text", "cpt": "Idiom+Meaning"}
MAXN, MAXC = 8, 150


def norm(lang, s):
    s = s.strip()
    if lang == "ar":
        from culture.data_processing.ar_idioms.normalize import normalize_ar
        s = normalize_ar(s)
    return re.sub(r"\s+", " ", s)


def unlist(s):
    s = str(s).strip()
    if s.startswith("["):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return "; ".join(map(str, v))
        except Exception:
            pass
    return s


def first_meaning(o):
    fm = o["output"].get("figurative_meanings")
    if isinstance(fm, list) and fm:
        return unlist(fm[0])
    return ""


AR_PRE = ["وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف", "ك", ""]
AR_SUF = ["", "ه", "ها", "ي", "ك", "هم", "نا", "ين", "ان", "ات"]


def mentions(lang, idiom, ent, ent_list):
    if isinstance(ent_list, list) and ent in ent_list:
        return True
    if lang == "zh":
        return ent in idiom
    toks = re.findall(r"\w+", norm(lang, idiom))
    e = norm(lang, ent)
    if lang == "hi":
        return e in toks
    forms = {p + e + s for p in AR_PRE for s in AR_SUF}
    return any(t in forms for t in toks)


def clip(s):
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= MAXC else s[:MAXC] + "…"


def split_ev(e):
    m = re.match(r"^(.*?)\s*[:：]\s*(.*)$", e, flags=re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else (e.strip(), "")


def build(args):
    for lang in ["zh", "hi", "ar"]:
        kb = [json.loads(l) for l in open(KB[lang], encoding="utf-8")]
        mean = {}
        for o in kb:
            mean.setdefault(norm(lang, o["idiom"]), first_meaning(o))
        blind = {json.loads(l)["qid"]: json.loads(l) for l in open(MAN / f"probe_blind_{lang}.jsonl", encoding="utf-8")}
        items = [json.loads(l) for l in open(MC / f"symbolism_v2_{lang}_letter.jsonl", encoding="utf-8")]
        outs = {v: [] for v in "BCD"}
        cands = {}
        miss = 0
        for it in items:
            b = blind[it["qid"]]
            ent = b["entity"]
            ev = [split_ev(e) for e in b["target_language_idioms"][:MAXN]]
            pairs = []
            for idi, m in ev:
                km = mean.get(norm(lang, idi), "")
                if not km:
                    miss += 1
                pairs.append((idi, km or unlist(m)))
            # D: random KB idioms whose text or entity list mentions the entity
            if ent not in cands:
                cands[ent] = [o for o in kb if mentions(lang, o["idiom"], ent, o["output"].get("entities"))]
            cand = cands[ent]
            rnd = random.Random(f"{it['qid']}-D")
            samp = rnd.sample(cand, min(MAXN, len(cand)))
            dpairs = [(o["idiom"], first_meaning(o)) for o in samp]
            hb, hc = HEAD[lang][0].format(e=ent), HEAD[lang][1].format(e=ent)
            blocks = {
                "B": hb + "\n" + "\n".join(f"- {i}" for i, _ in pairs),
                "C": hc + "\n" + "\n".join(f"- {i}{SEP[lang]}{clip(m)}" for i, m in pairs),
                "D": hc + "\n" + "\n".join(f"- {i}{SEP[lang]}{clip(m)}" for i, m in dpairs),
            }
            for v, blk in blocks.items():
                o = dict(it)
                o["context"] = blk + "\n\n" + it["context"]
                o["meta"] = {**it["meta"], "variant": v, "n_ctx": len(pairs) if v != "D" else len(dpairs)}
                outs[v].append(o)
        for v, rows in outs.items():
            fn = MC / f"symbolism_ctx{v}_{lang}_letter.jsonl"
            with open(fn, "w", encoding="utf-8") as f:
                for o in rows:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            print("wrote", fn, len(rows))
        print(lang, "evidence idioms without KB meaning (fell back to evidence text):", miss,
              "| D pool size median:", int(np.median([len(c) for c in cands.values()])),
              "| entities with <4 D idioms:", sum(len(c) < 4 for c in cands.values()))


def recs(root, lang, arm, task):
    p = root / lang / arm / f"{task}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return {r["qid"]: r for r in d["records"]}


def boot(groups, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    pooled = np.concatenate(groups)
    st = [np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]).mean() for _ in range(B)]
    lo, hi = np.percentile(st, [2.5, 97.5])
    return [round(100 * float(pooled.mean()), 1), round(100 * float(lo), 1), round(100 * float(hi), 1), int(len(pooled))]


def stats(args):
    root = Path(args.eval_root)
    langs = ["zh", "hi", "ar"]
    task = {"A": "symbolism_v2_{L}_letter", "B": "symbolism_ctxB_{L}_letter",
            "C": "symbolism_ctxC_{L}_letter", "D": "symbolism_ctxD_{L}_letter"}
    lure = {L: {json.loads(l)["qid"]: json.loads(l)["meta"]["lure"]
                for l in open(MC / f"symbolism_v2_{L}_letter.jsonl", encoding="utf-8")} for L in langs}
    R = {}
    for v, t in task.items():
        for L in langs:
            for a in RUNS:
                r = recs(root, L, a, t.format(L=L))
                if r:
                    R[(v, L, a)] = r
    out = {"note": "accuracy / lure rate in %, 95% CI from item bootstrap stratified by language",
           "levels": {}, "contrasts": {}, "variant_gain": {}}

    def vec(v, L, a, kind):
        r = R[(v, L, a)]
        q = sorted(r)
        if kind == "acc":
            return q, np.array([r[k]["correct"] for k in q], float)
        return q, np.array([int(r[k]["pred"]) == lure[L][k] for k in q], float)

    for v in task:
        for a in RUNS:
            for kind in ["acc", "lure"]:
                gs, per = [], {}
                for L in langs:
                    if (v, L, a) in R:
                        _, x = vec(v, L, a, kind)
                        gs.append(x)
                        per[L] = round(100 * x.mean(), 1)
                if gs:
                    out["levels"].setdefault(v, {}).setdefault(RUNS[a], {})[kind] = {"pooled": boot(gs), **per}
        for x, y in [("untagged", "unfiltered"), ("cpt", "unfiltered"), ("cpt", "untagged"),
                     ("unfiltered", "base")]:
            for kind in ["acc", "lure"]:
                gs, per = [], {}
                for L in langs:
                    if (v, L, x) in R and (v, L, y) in R:
                        q, a1 = vec(v, L, x, kind)
                        _, a2 = vec(v, L, y, kind)
                        d = a1 - a2
                        gs.append(d)
                        per[L] = boot([d])
                if gs:
                    out["contrasts"].setdefault(v, {}).setdefault(f"{RUNS[x]} - {RUNS[y]}", {})[kind] = {
                        "pooled": boot(gs), **per}
    for a in RUNS:
        for v in "BCD":
            for kind in ["acc", "lure"]:
                gs, per = [], {}
                for L in langs:
                    if (v, L, a) in R and ("A", L, a) in R:
                        _, x1 = vec(v, L, a, kind)
                        _, x0 = vec("A", L, a, kind)
                        gs.append(x1 - x0)
                        per[L] = boot([x1 - x0])
                if gs:
                    out["variant_gain"].setdefault(RUNS[a], {}).setdefault(f"{v} - A", {})[kind] = {
                        "pooled": boot(gs), **per}
    # does training change how much the model gains from context? (difference in differences)
    out["gain_contrasts"] = {}
    for x, y in [("untagged", "unfiltered"), ("cpt", "unfiltered"), ("unfiltered", "base")]:
        for v in "BCD":
            gs = []
            for L in langs:
                keys = [(v, L, x), ("A", L, x), (v, L, y), ("A", L, y)]
                if all(k in R for k in keys):
                    d = (vec(v, L, x, "acc")[1] - vec("A", L, x, "acc")[1]) - (
                        vec(v, L, y, "acc")[1] - vec("A", L, y, "acc")[1])
                    gs.append(d)
            if gs:
                out["gain_contrasts"].setdefault(f"{RUNS[x]} - {RUNS[y]}", {})[f"{v} - A"] = boot(gs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["levels"], indent=1, ensure_ascii=False)[:4000])


def stats2b(args):
    """Variant A at 2B (existing records): accuracy and English-lure rate, both initializations."""
    root = B_ / "eval2b"
    langs = ["zh", "hi", "ar"]
    arms = {"random": "Random-CPT", "idiom_untagged": "Idiom-Text", "idiom_tagged": "Idiom+Meaning", "base": "Base"}
    lure = {L: {json.loads(l)["qid"]: json.loads(l)["meta"]["lure"]
                for l in open(MC / f"symbolism_v2_{L}_letter.jsonl", encoding="utf-8")} for L in langs}
    out = {}
    for init, pre in [("Qwen3.5-2B", "i_"), ("Qwen3.5-2B-Base", "")]:
        V = {}
        for L in langs:
            for a in arms:
                r = recs(root, L, pre + a, f"symbolism_v2_{L}_letter")
                if r:
                    q = sorted(r)
                    V[(L, a)] = (np.array([r[k]["correct"] for k in q], float),
                                 np.array([int(r[k]["pred"]) == lure[L][k] for k in q], float))
        o = {"levels": {}, "contrasts": {}}
        for a, nm in arms.items():
            for j, kind in enumerate(["acc", "lure"]):
                gs = [V[(L, a)][j] for L in langs if (L, a) in V]
                if gs:
                    o["levels"].setdefault(nm, {})[kind] = {
                        "pooled": boot(gs), **{L: round(100 * V[(L, a)][j].mean(), 1) for L in langs if (L, a) in V}}
        for x, y in [("idiom_untagged", "random"), ("idiom_tagged", "random"), ("idiom_tagged", "idiom_untagged")]:
            for j, kind in enumerate(["acc", "lure"]):
                gs, per = [], {}
                for L in langs:
                    if (L, x) in V and (L, y) in V:
                        d = V[(L, x)][j] - V[(L, y)][j]
                        gs.append(d)
                        per[L] = boot([d])
                if gs:
                    o["contrasts"].setdefault(f"{arms[x]} - {arms[y]}", {})[kind] = {"pooled": boot(gs), **per}
        out[init] = o
    fn = Path(args.out).with_name("probe_lure_2b.json")
    json.dump(out, open(fn, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats", "stats2b"])
    ap.add_argument("--eval_root", default=str(B_ / "eval9b_probectx"))
    ap.add_argument("--out", default=str(REPO / "docs/paper_stats/rebuttal/probe_context.json"))
    a = ap.parse_args()
    {"build": build, "stats": stats, "stats2b": stats2b}[a.cmd](a)
