#!/usr/bin/env python3
"""Build native-speaker validation workbooks (one .xlsx per language: zh, hi, ar).

Three tasks per workbook, sized for ~2 h for one annotator. Every answer cell is pre-filled with the
default ("correct" / "yes" / empty); the annotator changes only what they disagree with:
  A_KB         100 IdiomAtlas entries, drawn from the 200-entry audit sample (so the earlier
               automatic audits can be scored against people). Meaning, figurativeness, entities,
               Hindi OCR spelling, and (Arabic) regional spread and literal reading.
  B_MC         60 IdiomAtlas-MC items (30 seen, 30 unseen, mixed), keyed option highlighted. The annotator
               judges whether the key is correct and lists any other acceptable option.
  C_Symbolism  60 symbolism-probe items, keyed (target-culture) option highlighted; the English lure
               is not marked. The annotator judges whether the key is a recognized association
               and lists any other option that is also recognized.
One native speaker per language (no overlap subset).

Answer keys and provenance (split, gold, lure, earlier audit ids) go to a separate key file that
annotators never see.

Usage: python -m culture.analysis.rebuttal.build_annotation_sheets --out_dir <dir>
"""
import argparse
import json
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

REPO = Path(__file__).resolve().parents[4]
MANUAL = REPO / "docs/paper_stats/v5/manual"
MC = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir/eval_data/mc")
KB_AR = REPO / "culture/data/idioms/ar/idioms_merged_llm_formatted.jsonl"
LANG = {"zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
N_KB, N_KB_OVL = 100, 50
N_MC, N_MC_OVL = 60, 30
N_SYM, N_SYM_OVL = 60, 30
SEED = 20261124
FILL_IN = PatternFill("solid", fgColor="FFF4CC")
HEAD = PatternFill("solid", fgColor="DDE7F3")
KEY_FILL = PatternFill("solid", fgColor="B7E1B0")
LETTERS = "ABCD"

INSTRUCTIONS = """Native-speaker validation of IdiomAtlas ({lang})

Thank you for helping. Every yellow answer cell is PRE-FILLED with the default answer ("correct",
"yes", or empty = nothing wrong). Please read every row and change a cell only when you disagree, using
its drop-down list. Text cells (wrong/missing entities, other acceptable options) stay empty unless you
find something. Please do not search the web or ask an AI system; we want your own judgment as a
native speaker. "dont_know" / "unsure" is always a fine answer.

Estimated time: A_KB ~60 min, B_MC ~25 min, C_Symbolism ~45 min (about 2 hours in total).
You can stop and continue later; save the file and send it back when done.

---------------------------------------------------------------------------------------------------
Sheet A_KB: dictionary entries (100 rows)
Each row shows an idiom or proverb, the figurative meaning(s) we extracted from a dictionary,
{lit_note}and the concrete entities (nouns such as animals, body parts, objects) we listed for it.
  A1 meaning_correct (default: correct): is the figurative meaning right?
       correct            = right and clean
       correct_with_noise = right, but the text also contains extra material (a quotation, a story,
                            a source citation, broken text)
       partly_correct     = captures only part of the meaning, or is vague/too general
       wrong              = not the meaning of this idiom
  A2 is_figurative (default: yes): is this a figurative expression (idiom, proverb, saying, metaphorical phrase)
       rather than a plain literal phrase? yes / no / unsure
  A3 wrong_entities: copy any listed entity that is NOT a concrete thing mentioned in the idiom
       (leave empty if all are fine).
  A4 missing_entities: concrete nouns in the idiom that are missing from the list (optional).
{extra_a}
---------------------------------------------------------------------------------------------------
Sheet B_MC: multiple-choice meaning questions (60 rows)
Each row shows an idiom and four candidate meanings A-D (this is our benchmark IdiomAtlas-MC).
The option highlighted in GREEN is our answer key (column "key" repeats its letter).
  B1 key_correct (default: yes): is the green option a correct meaning of the idiom?
       yes / partly (right but incomplete or vague) / no / dont_know (I don't know this idiom)
  B2 other_acceptable: list any OTHER option (not the green one) that is also an acceptable meaning
       of the idiom, e.g. "C" or "A,D"; leave empty if none.

---------------------------------------------------------------------------------------------------
Sheet C_Symbolism: what an entity symbolizes (60 rows)
Each row names an entity (e.g. owl, dog, water) in {lang} with an English gloss, and four
candidate associations A-D, all in {lang}. The question is:
  "In {lang} idioms and sayings, <entity> usually symbolizes ..."
The option highlighted in GREEN is our answer key (column "key" repeats its letter).
  C1 gloss_ok (default: yes): is the English gloss the right sense of the {lang} word as used here? yes / no
       (some words have several senses; mark no if the gloss picks the wrong one).
  C2 key_recognized (default: yes): is the green option a recognized association of this entity in {lang}
       idioms and sayings? yes / partly / no / dont_know
  C3 other_recognized: list any OTHER option (not the green one) that is ALSO a recognized
       association of this entity in {lang}-speaking culture, e.g. "A" or "A,C"; leave empty if none.
"""

EXTRA_A = {
    "hi": "  A7 spelling_ok (default: yes): is the idiom text spelled correctly (the dictionary was digitized with OCR)?\n"
          "       yes / minor_errors / unreadable\n",
    "ar": "  A7 region (default: dont_know): where is this expression used? pan_arab /\n"
          "       classical_literary / regional / dont_know. Change it only if you know.\n"
          "  A8 literal_ok (default: correct): is the word-for-word literal reading correct? correct / partly / wrong\n",
    "zh": "",
}
LIT_NOTE = {"ar": "a word-for-word literal reading, ", "zh": "the literal meaning or source when available, ",
            "hi": ""}


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ar_register():
    """idiom string -> (sources, varieties) from the Arabic KB metadata."""
    import ast
    out = {}
    for r in rows(KB_AR):
        try:
            m = ast.literal_eval(r["meta"]) if isinstance(r.get("meta"), str) else r.get("meta", {})
        except Exception:
            m = {}
        out[r["idiom"].strip()] = ("; ".join(m.get("sources", [])), "; ".join(m.get("varieties_raw", [])))
    return out


def add_sheet(wb, title, head, data, fill_cols, validations, widths):
    ws = wb.create_sheet(title)
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = HEAD
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for d in data:
        ws.append(d)
    n = len(data) + 1
    for col in fill_cols:
        idx = head.index(col) + 1
        for r in range(2, n + 1):
            ws.cell(row=r, column=idx).fill = FILL_IN
    for col, choices in validations.items():
        idx = head.index(col) + 1
        letter = ws.cell(row=1, column=idx).column_letter
        dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{letter}2:{letter}{n}")
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    return ws


def build(lang, out_dir):
    rng = random.Random(f"{SEED}-{lang}")
    key = {"lang": lang, "A_KB": [], "B_MC": [], "C_Symbolism": []}
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Instructions"
    for line in INSTRUCTIONS.format(lang=LANG[lang], lit_note=LIT_NOTE[lang], extra_a=EXTRA_A[lang]).split("\n"):
        ws0.append([line])
    ws0.column_dimensions["A"].width = 110

    # ---- A: KB entries, sampled from the 200-entry audit sample
    sheet = rows(MANUAL / f"kb_sheet_{lang}.jsonl")
    reg = ar_register() if lang == "ar" else {}
    if lang == "ar":  # keep both registers: stratify by classical vs colloquial label
        def is_classical(x):
            return "classical" in reg.get(x["idiom"].strip(), ("", ""))[1].lower()
        cl = [x for x in sheet if is_classical(x)]
        co = [x for x in sheet if not is_classical(x)]
        k = round(N_KB * len(cl) / len(sheet))
        pick = rng.sample(cl, min(k, len(cl))) + rng.sample(co, N_KB - min(k, len(cl)))
        rng.shuffle(pick)
    else:
        pick = rng.sample(sheet, N_KB)
    head = ["row", "idiom", "figurative_meaning"]
    if lang in ("ar", "zh"):
        head.append("literal_meaning_or_source")
    head += ["entities", "A1_meaning_correct", "A2_is_figurative", "A3_wrong_entities",
             "A4_missing_entities"]
    if lang == "hi":
        head.append("A7_spelling_ok")
    if lang == "ar":
        head += ["A7_region", "A8_literal_ok"]
    head.append("notes")
    default = {"A1_meaning_correct": "correct", "A2_is_figurative": "yes", "A7_spelling_ok": "yes",
               "A7_region": "dont_know", "A8_literal_ok": "correct"}
    data = []
    for i, x in enumerate(pick, 1):
        r = [i, x["idiom"], x.get("figurative_meanings", "")]
        if lang in ("ar", "zh"):
            r.append(x.get("literal_meanings", ""))
        r.append(x.get("entities", ""))
        r += [default.get(h, "") for h in head[len(r):]]
        data.append(r)
        key["A_KB"].append({"row": i, "audit_id": x["id"], "idiom": x["idiom"],
                            "register": reg.get(x["idiom"].strip(), ("", ""))[1] if lang == "ar" else "",
                            "source": reg.get(x["idiom"].strip(), ("", ""))[0] if lang == "ar" else ""})
    fill = [h for h in head if h[:2] in ("A1", "A2", "A3", "A4", "A7", "A8")] + ["notes"]
    val = {"A1_meaning_correct": ["correct", "correct_with_noise", "partly_correct", "wrong"],
           "A2_is_figurative": ["yes", "no", "unsure"]}
    if lang == "hi":
        val["A7_spelling_ok"] = ["yes", "minor_errors", "unreadable"]
    if lang == "ar":
        val["A7_region"] = ["pan_arab", "classical_literary", "regional", "dont_know"]
        val["A8_literal_ok"] = ["correct", "partly", "wrong"]
    widths = [5, 30, 50] + ([40] if lang in ("ar", "zh") else []) + [20] + [16] * (len(head) - (5 if lang in ("ar", "zh") else 4))
    add_sheet(wb, "A_KB", head, data, fill, val, widths)

    # ---- B: IdiomAtlas-MC items, 30 seen + 30 unseen (zh unseen has 78 items)
    picks = []
    for split in ["seen", "unseen"]:
        items = rows(MC / f"idiomatlas_mc_{lang}_{split}.jsonl")
        picks += [(split, x) for x in rng.sample(items, N_MC // 2)]
    # interleave so the overlap subset (first 30) holds 15 seen + 15 unseen
    seen_p = [p for p in picks if p[0] == "seen"]
    uns_p = [p for p in picks if p[0] == "unseen"]
    order = []
    for a, b in zip(seen_p, uns_p):
        order += [a, b] if rng.random() < 0.5 else [b, a]
    head = ["row", "idiom", "A", "B", "C", "D", "key", "B1_key_correct", "B2_other_acceptable"]
    data = []
    for i, (split, x) in enumerate(order, 1):
        opts = [o.strip() for o in x["options"]]
        data.append([i, x["meta"]["idiom"], *opts, LETTERS[x["gold"]], "yes", ""])
        key["B_MC"].append({"row": i, "qid": x["qid"], "split": split, "gold": LETTERS[x["gold"]],
                            "idiom": x["meta"]["idiom"]})
    ws = add_sheet(wb, "B_MC", head, data, ["B1_key_correct", "B2_other_acceptable"],
                   {"B1_key_correct": ["yes", "partly", "no", "dont_know"]},
                   [5, 28, 40, 40, 40, 40, 6, 14, 16])
    for i, (split, x) in enumerate(order, 2):  # highlight the keyed option in each row
        c = ws.cell(row=i, column=head.index(LETTERS[x["gold"]]) + 1)
        c.fill = KEY_FILL
        c.font = Font(bold=True)

    # ---- C: symbolism probe items
    blind = {x["qid"]: x for x in rows(MANUAL / f"probe_blind_{lang}.jsonl")}
    pkey = {x["qid"]: x for x in json.load(open(MANUAL / f"probe_key_{lang}.json"))}
    qids = sorted(blind, key=lambda q: int(q.split("/")[-1]))
    pick = rng.sample(qids, min(N_SYM, len(qids)))
    head = ["row", "entity", "english_gloss", "question", "A", "B", "C", "D", "key", "C1_gloss_ok",
            "C2_key_recognized", "C3_other_recognized"]
    data = []
    for i, q in enumerate(pick, 1):
        b = blind[q]
        data.append([i, b["entity"], b["entity_en"], b["question"],
                     *[b["options"][L] for L in LETTERS], pkey[q]["gold"], "yes", "yes", ""])
        key["C_Symbolism"].append({"row": i, "qid": q, "gold": pkey[q]["gold"], "english_lure": pkey[q]["english_lure"],
                                   "entity": b["entity"], "entity_en": b["entity_en"]})
    ws = add_sheet(wb, "C_Symbolism", head, data, ["C1_gloss_ok", "C2_key_recognized", "C3_other_recognized"],
                   {"C1_gloss_ok": ["yes", "no"], "C2_key_recognized": ["yes", "partly", "no", "dont_know"]},
                   [5, 14, 14, 36, 34, 34, 34, 34, 6, 10, 14, 16])
    for i, q in enumerate(pick, 2):  # highlight the keyed (target-culture) option; the English lure is not marked
        c = ws.cell(row=i, column=head.index(pkey[q]["gold"]) + 1)
        c.fill = KEY_FILL
        c.font = Font(bold=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"IdiomAtlas_validation_{lang}.xlsx"
    wb.save(fn)
    json.dump(key, open(out_dir / f"KEY_DO_NOT_SHARE_{lang}.json", "w"), ensure_ascii=False, indent=1)
    print("wrote", fn, {k: len(v) for k, v in key.items() if isinstance(v, list)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--langs", nargs="+", default=["zh", "hi", "ar"])
    ap.add_argument("--c_v2_only", action="store_true",
                    help="only write the C_Symbolism v2 workbook (needs the existing KEY file)")
    a = ap.parse_args()
    for L in a.langs:
        if not a.c_v2_only:
            build(L, Path(a.out_dir))
        build_c_v2(L, Path(a.out_dir))



# ---------------------------------------------------------------------------------------------
# C_Symbolism v2: same 60 items and row order as the main workbook, plus the target-language
# evidence idioms each item was built from. Annotators reported that the entity alone is often not
# enough to judge an LLM-summarized association, so v2 asks two separate questions:
# does the key fairly summarize these idioms, and is it a conventional association in the culture.
C2_INSTR = """Native-speaker validation of the {lang} symbolism probe (sheet C, version 2)

Each row names an entity, the {lang} idioms our probe was built from (with their dictionary meanings),
and four candidate associations A-D. The question asked of the model is:
  "In {lang} idioms and sayings, <entity> usually symbolizes ..."
The option highlighted in GREEN is our answer key (column "key" repeats its letter).
Every answer cell is PRE-FILLED with the default; change it with the drop-down only when you disagree.

  C1 gloss_ok (default: yes): is the English gloss the right sense of the {lang} word in these idioms?
  C2 fits_idioms (default: yes): reading the idioms shown, is the green option a fair description of
       what the entity stands for in them? yes / partly / no
  C3 conventional (default: yes): beyond these particular idioms, is this a conventional, widely
       recognized association of the entity in {lang}? yes / partly / no
       (choose no if the entity has no conventional symbolic meaning in {lang}, e.g. it is only used
       literally, or its "meaning" exists only inside these few idioms)
  C4 other_recognized (default: empty): any OTHER option that is ALSO a recognized association of this
       entity in {lang}-speaking culture, e.g. "A" or "A,C".
Estimated time: about 40 minutes for 60 rows.
"""


def build_c_v2(lang, out_dir, max_idioms=8, max_chars=140):
    key = json.load(open(out_dir / f"KEY_DO_NOT_SHARE_{lang}.json"))
    blind = {x["qid"]: x for x in rows(MANUAL / f"probe_blind_{lang}.jsonl")}
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Instructions"
    for line in C2_INSTR.format(lang=LANG[lang]).split("\n"):
        ws0.append([line])
    ws0.column_dimensions["A"].width = 110
    head = ["row", "entity", "english_gloss", f"{LANG[lang]}_idioms_with_this_entity", "A", "B", "C", "D", "key",
            "C1_gloss_ok", "C2_fits_idioms", "C3_conventional", "C4_other_recognized"]
    data = []
    for k in key["C_Symbolism"]:
        b = blind[k["qid"]]
        ev = [e if len(e) <= max_chars else e[:max_chars] + "…" for e in b["target_language_idioms"][:max_idioms]]
        more = len(b["target_language_idioms"]) - len(ev)
        evid = "\n".join(f"• {e}" for e in ev) + (f"\n(+{more} more)" if more > 0 else "")
        data.append([k["row"], b["entity"], b["entity_en"], evid, *[b["options"][L] for L in LETTERS], k["gold"],
                     "yes", "yes", "yes", ""])
    ws = add_sheet(wb, "C_Symbolism", head, data, ["C1_gloss_ok", "C2_fits_idioms", "C3_conventional", "C4_other_recognized"],
                   {"C1_gloss_ok": ["yes", "no"], "C2_fits_idioms": ["yes", "partly", "no"],
                    "C3_conventional": ["yes", "partly", "no"]},
                   [5, 12, 12, 60, 30, 30, 30, 30, 6, 10, 12, 13, 14])
    for i, k in enumerate(key["C_Symbolism"], 2):
        c = ws.cell(row=i, column=head.index(k["gold"]) + 1)
        c.fill = KEY_FILL
        c.font = Font(bold=True)
    fn = out_dir / f"IdiomAtlas_validation_{lang}_C_v2.xlsx"
    wb.save(fn)
    print("wrote", fn, len(data))


if __name__ == "__main__":
    main()
