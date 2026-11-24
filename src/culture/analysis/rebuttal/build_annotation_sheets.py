#!/usr/bin/env python3
"""Build native-speaker validation workbooks (one .xlsx per language: zh, hi, ar).

Three tasks per workbook, sized for ~3 h for one annotator:
  A_KB         100 IdiomAtlas entries, drawn from the 200-entry audit sample (so the earlier
               automatic audits can be scored against people). Meaning, figurativeness, entities,
               the annotator's own familiarity with the idiom, and (Arabic) regional spread.
  B_MC         60 IdiomAtlas-MC items (30 seen, 30 unseen, mixed). The annotator answers blind,
               then marks every option that is an acceptable meaning.
  C_Symbolism  60 symbolism-probe items. The annotator answers blind, then marks every option
               that is a recognized association of the entity in their culture.
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
LETTERS = "ABCD"

INSTRUCTIONS = """Native-speaker validation of IdiomAtlas ({lang})

Thank you for helping. Please work in this file and fill only the yellow columns. Most cells have a
drop-down list; free-text cells are optional unless stated. Please do not search the web or ask an AI
system; we want your own judgment as a native speaker. "Don't know" / "unsure" is always a fine answer.

Estimated time: A_KB ~80 min, B_MC ~30 min, C_Symbolism ~60 min (about 3 hours in total).
You can stop and continue later; save the file and send it back when done.

---------------------------------------------------------------------------------------------------
Sheet A_KB: dictionary entries (100 rows)
Each row shows an idiom or proverb, the figurative meaning(s) we extracted from a dictionary,
{lit_note}and the concrete entities (nouns such as animals, body parts, objects) we listed for it.
  A1 meaning_correct: is the figurative meaning right?
       correct            = right and clean
       correct_with_noise = right, but the text also contains extra material (a quotation, a story,
                            a source citation, broken text)
       partly_correct     = captures only part of the meaning, or is vague/too general
       wrong              = not the meaning of this idiom
  A2 is_figurative: is this a figurative expression (idiom, proverb, saying, metaphorical phrase)
       rather than a plain literal phrase? yes / no / unsure
  A3 wrong_entities: copy any listed entity that is NOT a concrete thing mentioned in the idiom
       (leave empty if all are fine).
  A4 missing_entities: concrete nouns in the idiom that are missing from the list (optional).
  A5 familiarity:
       know_same      = I know this idiom and understand it the way the dictionary meaning says
       know_different = I know it, but I (or people around me) use/understand it differently
       heard_unsure   = I may have heard it but am not sure what it means
       unknown        = I don't know this idiom
  A6 my_meaning: only if A5 = know_different, write in a few words how you understand it.
{extra_a}
---------------------------------------------------------------------------------------------------
Sheet B_MC: multiple-choice meaning questions (60 rows)
Each row shows an idiom and four candidate meanings A-D (this is our benchmark IdiomAtlas-MC).
  B1 best_option: which option gives the meaning of the idiom? A / B / C / D / none
       Answer from your own knowledge first. If you don't know the idiom, give your best guess and
       set B3 = 1.
  B2 acceptable_options: list EVERY option that is an acceptable meaning, e.g. "B" or "B,D";
       write "none" if no option is acceptable.
  B3 confidence: 1 = guess, 2 = fairly sure, 3 = certain.

---------------------------------------------------------------------------------------------------
Sheet C_Symbolism: what an entity symbolizes (60 rows)
Each row names an entity (e.g. owl, dog, water) in {lang} with an English gloss, and four
candidate associations A-D, all in {lang}. The question is:
  "In {lang} idioms and sayings, <entity> usually symbolizes ..."
  C1 gloss_ok: is the English gloss the right sense of the {lang} word as used here? yes / no
       (some words have several senses; mark no if the gloss picks the wrong one).
  C2 best_option: which option best states what the entity usually symbolizes in {lang}
       idioms and sayings? A / B / C / D / none
  C3 recognized_options: list EVERY option that is a recognized association of this entity in
       {lang}-speaking culture (e.g. "A" or "A,C"); write "none" if none is.
  C4 confidence: 1 = guess, 2 = fairly sure, 3 = certain.
  C5 notes: optional, e.g. "in my region X means Y instead".
"""

EXTRA_A = {
    "hi": "  A7 spelling_ok: is the idiom text spelled correctly (the dictionary was digitized with OCR)?\n"
          "       yes / minor_errors / unreadable\n",
    "ar": "  A7 region: where is this expression used? pan_arab / classical_literary / regional /\n"
          "       dont_know. If regional, name the region or dialect in A8 (e.g. Egyptian, Levantine,\n"
          "       Gulf, Maghrebi, Hassaniya).\n"
          "  A9 literal_ok: is the word-for-word literal reading correct? correct / partly / wrong\n",
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
             "A4_missing_entities", "A5_familiarity", "A6_my_meaning"]
    if lang == "hi":
        head.append("A7_spelling_ok")
    if lang == "ar":
        head += ["A7_region", "A8_region_name", "A9_literal_ok"]
    head.append("notes")
    data = []
    for i, x in enumerate(pick, 1):
        r = [i, x["idiom"], x.get("figurative_meanings", "")]
        if lang in ("ar", "zh"):
            r.append(x.get("literal_meanings", ""))
        r.append(x.get("entities", ""))
        r += [""] * (len(head) - len(r))
        data.append(r)
        key["A_KB"].append({"row": i, "audit_id": x["id"], "idiom": x["idiom"],
                            "register": reg.get(x["idiom"].strip(), ("", ""))[1] if lang == "ar" else "",
                            "source": reg.get(x["idiom"].strip(), ("", ""))[0] if lang == "ar" else ""})
    fill = [h for h in head if h[:2] in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9")] + ["notes"]
    val = {"A1_meaning_correct": ["correct", "correct_with_noise", "partly_correct", "wrong"],
           "A2_is_figurative": ["yes", "no", "unsure"],
           "A5_familiarity": ["know_same", "know_different", "heard_unsure", "unknown"]}
    if lang == "hi":
        val["A7_spelling_ok"] = ["yes", "minor_errors", "unreadable"]
    if lang == "ar":
        val["A7_region"] = ["pan_arab", "classical_literary", "regional", "dont_know"]
        val["A9_literal_ok"] = ["correct", "partly", "wrong"]
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
    head = ["row", "idiom", "A", "B", "C", "D", "B1_best_option", "B2_acceptable_options",
            "B3_confidence", "notes"]
    data = []
    for i, (split, x) in enumerate(order, 1):
        opts = [o.strip() for o in x["options"]]
        data.append([i, x["meta"]["idiom"], *opts, "", "", "", ""])
        key["B_MC"].append({"row": i, "qid": x["qid"], "split": split, "gold": LETTERS[x["gold"]],
                            "idiom": x["meta"]["idiom"]})
    add_sheet(wb, "B_MC", head, data, ["B1_best_option", "B2_acceptable_options", "B3_confidence", "notes"],
              {"B1_best_option": ["A", "B", "C", "D", "none"], "B3_confidence": ["1", "2", "3"]},
              [5, 28, 40, 40, 40, 40, 12, 16, 12, 30])

    # ---- C: symbolism probe items
    blind = {x["qid"]: x for x in rows(MANUAL / f"probe_blind_{lang}.jsonl")}
    pkey = {x["qid"]: x for x in json.load(open(MANUAL / f"probe_key_{lang}.json"))}
    qids = sorted(blind, key=lambda q: int(q.split("/")[-1]))
    pick = rng.sample(qids, min(N_SYM, len(qids)))
    head = ["row", "entity", "english_gloss", "question", "A", "B", "C", "D", "C1_gloss_ok",
            "C2_best_option", "C3_recognized_options", "C4_confidence", "C5_notes"]
    data = []
    for i, q in enumerate(pick, 1):
        b = blind[q]
        data.append([i, b["entity"], b["entity_en"], b["question"],
                     *[b["options"][L] for L in LETTERS], "", "", "", "", ""])
        key["C_Symbolism"].append({"row": i, "qid": q, "gold": pkey[q]["gold"], "english_lure": pkey[q]["english_lure"],
                                   "entity": b["entity"], "entity_en": b["entity_en"]})
    add_sheet(wb, "C_Symbolism", head, data, ["C1_gloss_ok", "C2_best_option", "C3_recognized_options",
                                              "C4_confidence", "C5_notes"],
              {"C1_gloss_ok": ["yes", "no"], "C2_best_option": ["A", "B", "C", "D", "none"],
               "C4_confidence": ["1", "2", "3"]},
              [5, 14, 14, 36, 34, 34, 34, 34, 10, 12, 16, 12, 30])

    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"IdiomAtlas_validation_{lang}.xlsx"
    wb.save(fn)
    json.dump(key, open(out_dir / f"KEY_DO_NOT_SHARE_{lang}.json", "w"), ensure_ascii=False, indent=1)
    print("wrote", fn, {k: len(v) for k, v in key.items() if isinstance(v, list)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--langs", nargs="+", default=["zh", "hi", "ar"])
    a = ap.parse_args()
    for L in a.langs:
        build(L, Path(a.out_dir))


if __name__ == "__main__":
    main()
