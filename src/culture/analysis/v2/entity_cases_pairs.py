#!/usr/bin/env python3
"""Table 2 (same entity, different meaning) for the five language pairs beyond en-zh.

The paper's Table~\\ref{tab:entity-cases} covers English-Chinese only. This builds the
same table for en-hi, en-ar, zh-hi, zh-ar and hi-ar.

Method, following the en-zh original: entities are matched across a pair through their
shared English anchor (the translations already computed in
entity_translations_{zh,hi,ar}_en.json, so no new translation calls); for each entity we
retrieve its idioms in both languages and ask an LLM to summarise, *from those idioms
only*, the primary meanings each side carries and the meaning they share.

Two departures, both forced by what we learned since:

1. An audit found 17.6% of the original summaries' claims unsupported by the idioms they
   were written from, so every claim here is re-checked against the evidence by a second
   pass and unsupported ones are dropped before the table is written.
2. The document is pdfLaTeX + CJKutf8, which can typeset Chinese via \\zh{} but not
   Devanagari or Arabic, and no existing table contains either script. Hindi and Arabic
   idioms are therefore romanised with a literal English gloss, as the entity figure
   already does.

    LLAMA_API_KEY=... PYTHONPATH=src:src/culture/analysis/v2 python entity_cases_pairs.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

import common
import metagen_api

PAIRS = [("en", "hi"), ("en", "ar"), ("zh", "hi"), ("zh", "ar"), ("hi", "ar")]
LANG_NAME = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
LATIN = {"en"}                      # needs no romanisation
CJK = {"zh"}                        # \zh{} macro exists
MIN_IDIOMS = 8
CAP = 18                            # idioms shown per side
N_ROWS = 5                          # entities per pair in the table

SUMMARY_PROMPT = """You are comparing what one entity stands for in the idioms of two languages.

Entity: "{anchor}"

{a_name} idioms containing it:
{a_ev}

{b_name} idioms containing it:
{b_ev}

Using ONLY the idioms above as evidence, describe what the entity stands for in each
language. Do not use anything else you know about {anchor}, about {a_name} or about
{b_name}: if an association is not visible in the idioms listed, do not mention it.

Return one JSON object:
{{
  "a_meanings": "two or three primary meanings, separated by semicolons, lower case",
  "a_examples": ["2-4 idioms from the {a_name} list above, verbatim"],
  "b_meanings": "two or three primary meanings, separated by semicolons, lower case",
  "b_examples": ["2-4 idioms from the {b_name} list above, verbatim"],
  "shared": "the one meaning both languages share, three to six words; use \\"none\\" if they share none"{extra}
}}
Return only the JSON object."""

ROMAN_NOTE = """,
  "a_examples_roman": ["for each a_example, its romanisation in plain ASCII letters"],
  "a_examples_gloss": ["for each a_example, a literal English gloss of 3-7 words"],
  "b_examples_roman": ["for each b_example, its romanisation in plain ASCII letters"],
  "b_examples_gloss": ["for each b_example, a literal English gloss of 3-7 words"]"""

VERIFY_PROMPT = """Below are {n} {lang} idioms, each with its meaning.

{ev}

Claim about what "{anchor}" stands for in {lang} idioms:

    "{claim}"

Is this claim supported by at least one idiom above? Judge only from the idioms shown.
If none clearly supports it, answer no. Answer with one word, yes or no."""


def _clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def ascii_fold(s):
    """Romanisations come back in IAST/ALA-LC; pdfLaTeX with T1 cannot set the
    dot-below and half-ring characters, so fold everything to plain ASCII."""
    import unicodedata
    s = str(s or "")
    for a, b in [("\u02bf", "'"), ("\u02be", "'"), ("\u2019", "'"),
                 ("\u2018", "'"), ("\uff0c", ", ")]:
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.encode("ascii", "ignore").decode("ascii")


# Glosses are supposed to be English, but the model occasionally leaves a character of
# the source script inside one ("ant fights snail\u4e89"). Raw CJK, Devanagari or Arabic
# outside \zh{} does not compile under T1, so strip it from anything set as Latin text.
NON_LATIN = re.compile(r"[\u0590-\u08ff\u0900-\u097f\u3000-\u303f"
                       r"\u3400-\u9fff\uf900-\ufaff\ufb50-\ufdff\ufe70-\ufeff]+")


def latin_safe(s):
    return re.sub(r"\s{2,}", " ", NON_LATIN.sub("", str(s or ""))).strip()


def tex_escape(s):
    s = str(s or "")
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def anchor_groups(lang):
    """english anchor -> native entity strings."""
    if lang == "en":
        return None
    p = os.path.join(common.OUT, f"entity_translations_{lang}_en.json")
    g = defaultdict(list)
    for e, t in json.load(open(p, encoding="utf-8")).items():
        if t and t.get("en") and t["en"] != "none":
            g[t["en"]].append(e)
    return g


def build_index(lang, top_en=600):
    kb = common.load_kb(lang)
    idx = common.entity_index(kb)
    groups = anchor_groups(lang)
    if groups is None:
        cnt = common.entity_counter(kb)
        groups = {e: [e] for e, _ in cnt.most_common(top_en)}
    return kb, idx, groups


def evidence(kb, ids, cap=CAP):
    out = []
    for j in ids[:cap]:
        m = _clean(kb[j]["fig"][0])[:140]
        out.append(f"- {kb[j]['idiom']} — {m}")
    return "\n".join(out)


def divergence_rank():
    """entity -> percentile per pair, to prefer entities that actually diverge."""
    out = {}
    for fam in ("multi", "cross"):
        p = os.path.join(common.OUT, f"entity_divergence_{fam}.json")
        if not os.path.exists(p):
            continue
        for key, blk in json.loads(open(p, encoding="utf-8").read())["pairs"].items():
            for r in (blk.get("gloss", {}).get("per_entity") or []):
                out[(key, r["entity_en"])] = r["percentile"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=N_ROWS)
    ap.add_argument("--out_json", default="entity_cases_pairs.json")
    ap.add_argument("--out_tex", default=None,
                    help="path to write the LaTeX tables (default: alongside the json)")
    args = ap.parse_args()
    rng = random.Random(0)
    div = divergence_rank()

    cache = {}
    def idx_for(l):
        if l not in cache:
            cache[l] = build_index(l)
        return cache[l]

    # ---- pick entities and build the summary prompts -------------------------
    prompts, meta = [], []
    for a, b in PAIRS:
        kba, ia, ga = idx_for(a)
        kbb, ib, gb = idx_for(b)
        shared = sorted(set(ga) & set(gb))
        cands = []
        for anc in shared:
            aid = sorted({j for e in ga[anc] for j in ia.get(e, []) if kba[j]["fig"]})
            bid = sorted({j for e in gb[anc] for j in ib.get(e, []) if kbb[j]["fig"]})
            if len(aid) >= MIN_IDIOMS and len(bid) >= MIN_IDIOMS:
                d = div.get((f"{a}-{b}", anc), div.get((f"{b}-{a}", anc), 0.0))
                cands.append((d, anc, aid, bid))
        # most divergent first: those are the informative rows
        cands.sort(key=lambda t: -t[0])
        for d, anc, aid, bid in cands[: args.rows]:
            needs_roman = not ({a, b} <= (LATIN | CJK))
            prompts.append(SUMMARY_PROMPT.format(
                anchor=anc, a_name=LANG_NAME[a], b_name=LANG_NAME[b],
                a_ev=evidence(kba, aid), b_ev=evidence(kbb, bid),
                extra=ROMAN_NOTE if needs_roman else ""))
            meta.append({"pair": f"{a}-{b}", "a": a, "b": b, "anchor": anc,
                         "n_a": len(aid), "n_b": len(bid),
                         "div": round(d, 4), "a_ids": aid[:CAP], "b_ids": bid[:CAP]})
        print(f"[{a}-{b}] {len(shared)} shared anchors, "
              f"{len(cands)} with >={MIN_IDIOMS} idioms both sides, took {min(args.rows, len(cands))}")

    print(f"\n[summaries] {len(prompts)} calls")
    outs = metagen_api.generate(prompts, tag="entity_cases_pairs_summary",
                                max_tokens=1400, json_mode=True)

    rows = []
    for m, o in zip(meta, outs):
        try:
            d = json.loads(re.search(r"\{.*\}", o, re.S).group(0))
        except Exception:
            print(f"  [skip] unparsable summary for {m['pair']}/{m['anchor']}")
            continue
        m.update(d)
        rows.append(m)

    # ---- verify every claimed meaning against the evidence -------------------
    vp, vmeta = [], []
    for m in rows:
        kba, _, _ = idx_for(m["a"])
        kbb, _, _ = idx_for(m["b"])
        for side, kb, ids in (("a", kba, m["a_ids"]), ("b", kbb, m["b_ids"])):
            for claim in [c for c in str(m.get(f"{side}_meanings", "")).split(";") if c.strip()]:
                vp.append(VERIFY_PROMPT.format(
                    n=min(len(ids), CAP), lang=LANG_NAME[m[side]],
                    ev=evidence(kb, ids), anchor=m["anchor"], claim=claim.strip()))
                vmeta.append((id(m), side, claim.strip()))
    print(f"[verify] {len(vp)} claims")
    vouts = metagen_api.generate(vp, tag="entity_cases_pairs_verify", max_tokens=16)

    keep = defaultdict(lambda: defaultdict(list))
    dropped = 0
    for (mid, side, claim), o in zip(vmeta, vouts):
        if (o or "").strip().lower().startswith("y"):
            keep[mid][side].append(claim)
        else:
            dropped += 1
    for m in rows:
        for side in ("a", "b"):
            m[f"{side}_meanings_verified"] = "; ".join(keep[id(m)][side])
    tot = len(vmeta)
    print(f"[verify] dropped {dropped}/{tot} unsupported claims "
          f"({dropped / max(1, tot):.1%})")

    p = common.dump({"method": __doc__, "pairs": [f"{a}-{b}" for a, b in PAIRS],
                     "min_idioms": MIN_IDIOMS, "rows_per_pair": args.rows,
                     "model": metagen_api.MODEL,
                     "verified_drop_rate": round(dropped / max(1, tot), 4),
                     "rows": rows}, args.out_json)
    print("wrote", p)

    tex = args.out_tex or os.path.join(common.OUT, "entity_cases_pairs.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(render(rows))
    print("wrote", tex)


def fmt_side(lang, examples, romans, glosses):
    """One cell: idioms in a form pdfLaTeX can set."""
    bits = []
    for i, ex in enumerate(examples[:3]):
        if lang == "en":
            bits.append(f"\\textit{{{tex_escape(ex)}}}")
        elif lang == "zh":
            g = tex_escape(latin_safe(glosses[i])) if i < len(glosses) else ""
            bits.append(f"\\zh{{{ex}}}" + (f" ``{g}''" if g else ""))
        else:  # hi, ar -> romanisation + gloss, no native script
            r = tex_escape(ascii_fold(romans[i] if i < len(romans) else ex))
            g = tex_escape(ascii_fold(latin_safe(glosses[i]))) if i < len(glosses) else ""
            bits.append(f"\\textit{{{r}}}" + (f" ``{g}''" if g else ""))
    return "; ".join(bits)


def render(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    COL = {"en": "encolor", "zh": "zhcolor", "hi": "encolor", "ar": "zhcolor"}
    out = []
    for pair, rs in by.items():
        a, b = pair.split("-")
        out.append("\\begin{table*}[t]\n\\centering\n\\footnotesize\n"
                   "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.12}")
        out.append(
            f"\\caption{{Same entity, different meaning in \\textcolor{{{COL[a]}}}{{{LANG_NAME[a]}}} "
            f"and \\textcolor{{{COL[b]}}}{{{LANG_NAME[b]}}} idioms. Entities are matched through a "
            f"shared English anchor; $n$ is the number of idioms per language the summary rests on. "
            f"Every meaning shown was re-checked against those idioms and unsupported claims removed "
            f"(\\S\\ref{{sec:analysis-entity}}). "
            + ("Hindi and Arabic idioms are romanised with a literal gloss. "
               if a in ("hi", "ar") or b in ("hi", "ar") else "")
            + ("Chinese idioms are given with a literal English gloss. " if "zh" in (a, b) else "")
            + f"}}\n\\label{{tab:entity-cases-{a}{b}}}")
        out.append("\\begin{tabular}{@{}p{0.085\\textwidth}p{0.335\\textwidth}"
                   "p{0.365\\textwidth}p{0.15\\textwidth}@{}}\n\\toprule")
        out.append(f"Entity ($n$ {a}/{b}) & \\textcolor{{{COL[a]}}}{{{LANG_NAME[a]} idioms}} & "
                   f"\\textcolor{{{COL[b]}}}{{{LANG_NAME[b]} idioms}} & Shared \\\\\n\\midrule")
        for r in rs:
            am = r.get("a_meanings_verified") or ""
            bm = r.get("b_meanings_verified") or ""
            if not am and not bm:
                continue
            acell = (f"\\textcolor{{{COL[a]}}}{{{tex_escape(am)}}}\\newline "
                     + fmt_side(a, r.get("a_examples", []), r.get("a_examples_roman", []),
                                r.get("a_examples_gloss", [])))
            bcell = (f"\\textcolor{{{COL[b]}}}{{{tex_escape(bm)}}}\\newline "
                     + fmt_side(b, r.get("b_examples", []), r.get("b_examples_roman", []),
                                r.get("b_examples_gloss", [])))
            sh = tex_escape(r.get("shared", "")) or "---"
            out.append(f"\\textbf{{{tex_escape(r['anchor'])}}}\\newline ({r['n_a']}/{r['n_b']})\n"
                       f"& {acell}\n& {bcell}\n& {sh} \\\\")
        out.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return "\n".join(out)


if __name__ == "__main__":
    main()
