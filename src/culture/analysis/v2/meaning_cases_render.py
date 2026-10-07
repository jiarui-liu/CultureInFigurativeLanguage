#!/usr/bin/env python3
"""Render Table 3 (same meaning, different entity) for the five pairs beyond en-zh.

meaning_cases_pairs.py does the pairing on GPU and writes the candidate clusters; this
turns them into the table. For each cluster an LLM is shown the idioms of both sides and
asked, from those idioms only, for a short label for the meaning they share, which
idioms actually carry it, and for each of those a literal gloss of the *image* with the
entity words marked -- that image is what the table exists to contrast.

As in Table 2, every row is re-checked by a second pass before it is written: an audit
found 17.6% of our original entity summaries made claims the idioms did not support, so
a claim here is kept only if a verifier agrees the chosen idioms express the meaning.

pdfLaTeX + CJKutf8 can set Chinese via \\zh{} but not Devanagari or Arabic, so Hindi and
Arabic idioms are romanised with a literal gloss.

    LLAMA_API_KEY=... PYTHONPATH=src:src/culture/analysis/v2 python meaning_cases_render.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import common
import metagen_api
from entity_cases_pairs import LANG_NAME, ascii_fold, tex_escape, _clean

N_ROWS = 6          # clusters per pair in the table
MAX_SIDE = 3        # idioms shown per language per row
POOL = 6            # idioms per side offered to the LLM

LABEL_PROMPT = """Two languages have idioms that appear to express a similar meaning.

{a_name} idioms (with the meaning our knowledge base records):
{a_ev}

{b_name} idioms (with the meaning our knowledge base records):
{b_ev}

Using ONLY the idioms and meanings above, answer. Do not use anything else you know
about these languages. If the two lists do not in fact express one shared meaning, say so.

Return one JSON object:
{{
  "shared": "the meaning both sides express, three to six words, lower case; or \\"none\\"",
  "a_pick": ["1-3 idioms from the {a_name} list, verbatim, that express it"],
  "b_pick": ["1-3 idioms from the {b_name} list, verbatim, that express it"],
  "a_image": ["for each a_pick, a literal 3-8 word gloss of the picture the words paint, with the concrete things it names wrapped in <e></e>; write \\"no imagery\\" if the idiom is not figurative"],
  "b_image": ["same, for each b_pick"],
  "a_roman": ["for each a_pick, its romanisation in plain ASCII letters; empty strings if {a_name} is English"],
  "b_roman": ["same, for each b_pick"]
}}
Return only the JSON object."""

VERIFY_PROMPT = """{lang} idiom: {idiom}
Recorded meaning: {mean}

Claim: this idiom expresses the meaning "{shared}".

Judge only from the idiom and the recorded meaning above. If it does not clearly express
that meaning, answer no. Answer with one word, yes or no."""


def ev(items, n=POOL):
    return "\n".join(f"- {it['idiom']}  ({_clean(it['mean'])[:120]})" for it in items[:n])


def unsmart(s):
    """Curly quotes and dashes out of the LLM; keep the source plain ASCII."""
    for a, b in [("\u2019", "'"), ("\u2018", "'"), ("\u201c", "``"), ("\u201d", "''"),
                 ("\u2013", "--"), ("\u2014", "---")]:
        s = s.replace(a, b)
    return s


def underline_ents(text, ents):
    """Underline the knowledge base's entity strings where they appear in the idiom.

    For English there is nothing to romanise and no gloss to add -- the idiom *is* the
    imagery -- so the entities are marked in place rather than in a paraphrase.
    """
    out = tex_escape(unsmart(text))
    for e in sorted({str(x) for x in ents if x}, key=len, reverse=True):
        e = tex_escape(unsmart(str(e)))
        if not e:
            continue
        out = re.sub(rf"(?<!\{{)\b({re.escape(e)})\b", r"\\underline{\1}", out,
                     count=1, flags=re.I)
    return out


def mark(s, lang):
    """<e>x</e> -> \\underline{x}, after escaping and (for hi/ar) folding to ASCII."""
    s = unsmart(_clean(s))
    parts, out = re.split(r"(<e>.*?</e>)", s), []
    for p in parts:
        m = re.fullmatch(r"<e>(.*?)</e>", p, re.S)
        t = m.group(1) if m else p
        if lang in ("hi", "ar"):
            t = ascii_fold(t)
        t = tex_escape(t)
        out.append(f"\\underline{{{t}}}" if m else t)
    return "".join(out).strip()


def cell(lang, picks, images, romans, ents):
    bits = []
    for i, idm in enumerate(picks[:MAX_SIDE]):
        img = mark(images[i] if i < len(images) else "", lang)
        if img.lower().replace("\\underline{", "").rstrip("}") == "no imagery":
            img = "\\textit{(no imagery)}"
        if lang == "en":
            bits.append(underline_ents(idm, ents.get(idm, [])))
        elif lang == "zh":
            bits.append(f"\\zh{{{idm}}} {img}".strip())
        else:
            r = tex_escape(ascii_fold(romans[i] if i < len(romans) and romans[i] else idm))
            bits.append(f"\\textit{{{r}}} {img}".strip())
    return "; ".join(b for b in bits if b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=N_ROWS)
    ap.add_argument("--cands", default=os.path.join(
        common.OUT, "meaning_cases_pairs_candidates.json"))
    ap.add_argument("--out_json", default="meaning_cases_pairs.json")
    ap.add_argument("--out_tex", default=None)
    args = ap.parse_args()

    cand = json.load(open(args.cands, encoding="utf-8"))["pairs"]

    prompts, meta = [], []
    for pair, d in cand.items():
        a, b = pair.split("-")
        for c in d["candidates"]:
            prompts.append(LABEL_PROMPT.format(
                a_name=LANG_NAME[a], b_name=LANG_NAME[b],
                a_ev=ev(c["a"]), b_ev=ev(c["b"])))
            meta.append({"pair": pair, "a": a, "b": b, "n_a": c["n_a"], "n_b": c["n_b"],
                         "sim": c["sim"], "a_items": c["a"][:POOL], "b_items": c["b"][:POOL]})
    print(f"[label] {len(prompts)} calls", flush=True)
    outs = metagen_api.generate(prompts, tag="meaning_cases_pairs_label",
                                max_tokens=1400, json_mode=True)

    rows = []
    for m, o in zip(meta, outs):
        try:
            d = json.loads(re.search(r"\{.*\}", o, re.S).group(0))
        except Exception:
            continue
        sh = _clean(d.get("shared", "")).lower().strip(" .")
        if not sh or sh == "none" or not d.get("a_pick") or not d.get("b_pick"):
            continue
        m.update(d)
        m["shared"] = sh
        rows.append(m)
    print(f"[label] {len(rows)}/{len(meta)} clusters got a shared meaning")

    # ---- verify each chosen idiom really expresses the shared meaning --------
    vp, vmeta = [], []
    for m in rows:
        for side in ("a", "b"):
            look = {it["idiom"]: it["mean"] for it in m[f"{side}_items"]}
            for idm in m.get(f"{side}_pick", [])[:MAX_SIDE]:
                vp.append(VERIFY_PROMPT.format(
                    lang=LANG_NAME[m[side]], idiom=idm,
                    mean=_clean(look.get(idm, ""))[:200], shared=m["shared"]))
                vmeta.append((id(m), side, idm))
    print(f"[verify] {len(vp)} idiom-meaning claims", flush=True)
    vouts = metagen_api.generate(vp, tag="meaning_cases_pairs_verify", max_tokens=16)

    ok = defaultdict(set)
    bad = 0
    for (mid, side, idm), o in zip(vmeta, vouts):
        if (o or "").strip().lower().startswith("y"):
            ok[(mid, side)].add(idm)
        else:
            bad += 1
    print(f"[verify] dropped {bad}/{len(vmeta)} unsupported "
          f"({bad / max(1, len(vmeta)):.1%})")

    kept = []
    for m in rows:
        for side in ("a", "b"):
            keep = [(i, p) for i, p in enumerate(m.get(f"{side}_pick", [])[:MAX_SIDE])
                    if p in ok[(id(m), side)]]
            m[f"{side}_pick_v"] = [p for _, p in keep]
            for f in ("image", "roman"):
                src = m.get(f"{side}_{f}", [])
                m[f"{side}_{f}_v"] = [src[i] if i < len(src) else "" for i, _ in keep]
        # a row only makes the point if both sides survive
            m[f"{side}_ents"] = {it["idiom"]: it.get("ents", [])
                                 for it in m[f"{side}_items"]}
        if m["a_pick_v"] and m["b_pick_v"]:
            kept.append(m)
    print(f"[verify] {len(kept)}/{len(rows)} clusters survive with both sides")

    final, seen = defaultdict(list), defaultdict(set)
    for m in kept:
        if len(final[m["pair"]]) >= args.rows or m["shared"] in seen[m["pair"]]:
            continue
        seen[m["pair"]].add(m["shared"])
        final[m["pair"]].append(m)
    for p in cand:
        print(f"  {p}: {len(final[p])} rows")

    p = common.dump({"method": __doc__, "model": metagen_api.MODEL,
                     "verified_drop_rate": round(bad / max(1, len(vmeta)), 4),
                     "n_pairs": {k: v["n_pairs"] for k, v in cand.items()},
                     "rows": {k: v for k, v in final.items()}}, args.out_json)
    print("wrote", p)

    tex = args.out_tex or os.path.join(common.OUT, "meaning_cases_pairs.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(render(final))
    print("wrote", tex)


def render(final):
    COL = {"a": "encolor", "b": "zhcolor"}   # by column, so the two never collide
    out = []
    for pair, rs in final.items():
        if not rs:
            continue
        a, b = pair.split("-")
        out.append("\\begin{table*}[t]\n\\centering\n\\footnotesize\n"
                   "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.12}")
        note = ""
        if "zh" in (a, b):
            note += "Chinese idioms are given with a literal gloss. "
        if a in ("hi", "ar") or b in ("hi", "ar"):
            note += "Hindi and Arabic idioms are romanised with a literal gloss. "
        out.append(
            f"\\caption{{Same meaning, different entity in \\textcolor{{{COL[a]}}}{{{LANG_NAME[a]}}} "
            f"and \\textcolor{{{COL[b]}}}{{{LANG_NAME[b]}}} idioms; entities are \\underline{{underlined}}. "
            f"Cluster sizes (number of idioms, {a}/{b}) show asymmetric lexicalization. "
            f"{note}Every idiom shown was re-checked against the shared meaning and unsupported "
            f"ones removed (\\S\\ref{{sec:analysis-meaning}}).}}\n"
            f"\\label{{tab:meaning-cases-{a}{b}}}")
        out.append("\\begin{tabular}{@{}p{0.20\\textwidth}p{0.37\\textwidth}"
                   "p{0.37\\textwidth}@{}}\n\\toprule")
        out.append(f"Shared meaning\\newline ({a}/{b} idioms) & "
                   f"\\textcolor{{{COL[a]}}}{{{LANG_NAME[a]} imagery}} & "
                   f"\\textcolor{{{COL[b]}}}{{{LANG_NAME[b]} imagery}} \\\\\n\\midrule")
        for r in rs:
            out.append(
                f"{tex_escape(r['shared'])} ({r['n_a']}/{r['n_b']})\n"
                f"& {cell(a, r['a_pick_v'], r['a_image_v'], r['a_roman_v'], r['a_ents'])}\n"
                f"& {cell(b, r['b_pick_v'], r['b_image_v'], r['b_roman_v'], r['b_ents'])} \\\\")
        out.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return "\n".join(out)


if __name__ == "__main__":
    main()
