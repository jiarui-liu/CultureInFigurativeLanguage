#!/usr/bin/env python3
"""Build the symbolism probe: what an entity symbolizes in a culture's idioms.

For a target language L in {zh, hi, ar} and English as the contrast culture, each
item asks what an entity (dog, dragon, the colour red, ...) typically symbolizes in
L's idioms and sayings, with four options written in L:

  gold        an association attested in L's idioms but not in English idioms
  lure        an association attested in ENGLISH idioms for the same entity but
              not in L's idioms (the cross-cultural lure)
  distractor  two plausible associations attested in neither language

Accuracy measures culture-specific symbolic knowledge; the lure rate (share of
items answered with the English association) measures transfer of Anglophone
symbolism. The stages are cached, so the script can be re-run:

  pairs     entity pairs (zh: the 515 en-zh analyses of Sec. 4; hi/ar: top entities
            translated to English by an LLM and matched to the English inventory)
  analyze   shared/unique-aspect analysis per pair (same prompt as the en-zh
            analysis, generalized to the language)
  generate  item generation from the analysis and the evidence idioms (Qwen3.5-27B-FP8)
  verify    blind verification by a second LLM family (Gemma-4-26B-A4B-it) that sees
            the evidence idioms of both languages and the shuffled options; an item
            is kept only if the verifier independently picks the gold option as the
            one supported by L's idioms and the lure as the one supported by English
            idioms.

Usage:
  python -m culture.bidirectional.build_symbolism_probe --lang hi --out_dir $B/eval_data/mc
"""
import argparse
import collections
import json
import os
import random
import re
from pathlib import Path

from culture.bidirectional.llm_api import complete_many, parse_json

REPO = Path(__file__).resolve().parents[3]
D = REPO / "culture/data"
B = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
LANG_NAME = {"zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
KB = {
    "en": D / "idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl",
    "zh": D / "idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl",
    "hi": D / "idioms/hi/idioms_merged_llm_formatted_figurative_only.jsonl",
    "ar": D / "idioms/ar/idioms_merged_llm_formatted.jsonl",
}
STEM = {
    "zh": "在汉语的成语和俗语中，“{entity}”常用来象征：",
    "hi": "हिंदी की कहावतों और लोकोक्तियों में \"{entity}\" प्रायः इसका प्रतीक होता है:",
    "ar": "في الأمثال والتعابير العربية، يرمز «{entity}» عادةً إلى:",
}
GEN = dict(provider="vllm", model=os.environ.get("GEN_MODEL_PATH", ""))
VERIFY = dict(provider="vllm", model=os.environ.get("VERIFY_MODEL_PATH", ""))


def _lst(v):
    if not v or v == "NAN":
        return []
    return v if isinstance(v, list) else [v]


def ar_key(e):
    """Arabic entity key: undiacritized, hamza-folded, without a leading article."""
    from culture.data_processing.ar_idioms.normalize import normalize_ar
    k = normalize_ar(e).strip()
    if k.startswith("ال") and len(k) > 3 and not k.startswith("الله"):
        k = k[2:]
    return k


def load_kb(lang, ar_entities=None):
    rows = [json.loads(l) for l in open(KB[lang], encoding="utf-8") if l.strip()]
    out = []
    ents_ar = {}
    if lang == "ar" and ar_entities:
        for l in open(ar_entities, encoding="utf-8"):
            o = json.loads(l)
            ents_ar[o["idiom"]] = o.get("entities", [])
    for r in rows:
        o = r["output"]
        ents = [ar_key(e) for e in ents_ar.get(o["idiom"], [])] if lang == "ar" else _lst(o.get("entities"))
        fm = _lst(o.get("figurative_meanings"))
        if not fm:
            continue
        out.append({"idiom": o["idiom"].strip(), "entities": [str(e).strip().lower() if lang == "en" else str(e).strip() for e in ents],
                    "meaning": "; ".join(str(m) for m in fm)[:300]})
    return out


def index_by_entity(kb):
    idx = collections.defaultdict(list)
    for it in kb:
        for e in set(it["entities"]):
            idx[e].append(it)
    return idx


def fmt(idioms, k=40):
    return "\n".join(f"- {x['idiom']}: {x['meaning']}" for x in idioms[:k])


ANALYZE = """You are a cultural linguistics expert analyzing how the same entity/concept conveys different meanings across English and {L} idioms.

Entity in English: "{en}"
Entity in {L}: "{tgt}"

English idioms containing this entity:
{en_idioms}

{L} idioms containing this entity:
{tgt_idioms}

Analyze the cultural and semantic differences:
1. What are the PRIMARY figurative meanings/connotations associated with this entity in English idioms?
2. What are the PRIMARY figurative meanings/connotations associated with this entity in {L} idioms?
3. What cultural values, beliefs, or perspectives might explain these differences?
4. Are there any SHARED meanings across both languages?
5. What unique cultural aspects does each language capture that the other doesn't?

Provide a structured analysis in JSON format:
{{"english_primary_meanings": [...], "target_primary_meanings": [...], "shared_meanings": [...], "english_unique_aspects": [...], "target_unique_aspects": [...], "cultural_explanation": "...", "summary": "..."}}"""

GENERATE = """You are building a multiple-choice probe of culture-specific SYMBOLISM in idioms: what an entity stands for, or how it is evaluated, in a culture's idioms and sayings.

Entity: "{tgt}" ({L}) / "{en}" (English).
Analysis of what the entity symbolizes in each language's idioms:
- Associations unique to {L} idioms: {tgt_unique}
- Associations unique to English idioms: {en_unique}
- Shared associations: {shared}

Evidence — {L} idioms with this entity (idiom: meaning):
{tgt_idioms}

Evidence — English idioms with this entity (idiom: meaning):
{en_idioms}

First decide whether the entity is suitable. Return {{"skip": true}} if ANY of these holds:
- the entity is abstract or generic (a word like "thing", "person", "place", "time", "way", "matter"), or a function word;
- the idioms use the entity in different senses of a polysemous word rather than attributing a symbolic or evaluative meaning to one concrete thing;
- you cannot find at least TWO {L} idioms above that share one symbolic association that the English idioms do not express, AND at least TWO English idioms that share one association that the {L} idioms do not express.

Otherwise write four answer options IN {L}, each a short noun phrase of 2-8 words that completes "In {L} idioms and sayings, '{tgt}' typically symbolizes ...":
- "gold": the {L}-specific symbolic association, supported by at least two of the {L} idioms above;
- "lure": the English-specific symbolic association, supported by at least two of the English idioms above (it must be a conventional English association);
- "distractors": two associations that sound plausible for this entity but are supported by NEITHER list of idioms.
Do not use the entity word itself or any word of any idiom in the options. Do not name either language or culture. Keep the four options similar in length and style.
Return JSON only: {{"gold": "...", "lure": "...", "distractors": ["...", "..."], "gold_evidence": [<the {L} idioms, copied exactly>], "lure_evidence": [<the English idioms, copied exactly>]}}"""

REWRITE = """Four answer options for the question "In {L} idioms and sayings, '{tgt}' typically symbolizes ..." are given below, with their roles.

gold (the {L}-specific association): {gold}
lure (the English-specific association): {lure}
distractor 1: {d1}
distractor 2: {d2}

Rewrite the four options IN {L} so that a reader cannot guess the answer from the form of the options:
- keep the meaning of the gold and of the lure unchanged;
- replace each distractor by an association that is specific to '{tgt}' and sounds just as plausible and as concrete as the gold, but is expressed by NEITHER {L} nor English idioms; never use generic filler themes that could fit any entity (for example wealth accumulation, natural seasons, or harmony in general);
- make all four options similar in length (within about 20 percent of each other), grammatical form, and level of detail;
- do not use the word '{tgt}' or any idiom words; do not name a language or culture.
Return JSON only: {{"gold": "...", "lure": "...", "distractors": ["...", "..."]}}"""

VERIFY_PROMPT = """Below are idioms from two languages that contain the same entity, followed by four candidate associations (written in {L}).

{L} idioms containing "{tgt}":
{tgt_idioms}

English idioms containing "{en}":
{en_idioms}

Options:
A. {A}
B. {B}
C. {C}
D. {D}

Answer three questions using ONLY the idioms above:
1. Which ONE option is most clearly what "{tgt}" symbolizes in the {L} idioms (and is not expressed in the English idioms)?
2. Which ONE option is most clearly what "{en}" symbolizes in the English idioms (and is not expressed in the {L} idioms)?
3. For each option, is it supported by the {L} idioms? (true/false)
Return JSON only: {{"q1": "<letter>", "q2": "<letter>", "supported_in_target": {{"A": bool, "B": bool, "C": bool, "D": bool}}}}"""


def zh_pairs(kb_en, kb_zh):
    tab = json.load(open(REPO / "docs/data/tab1_entity_meanings.json"))
    recs = [x for v in tab.values() for x in (v if isinstance(v, list) else [v])]
    ien, izh = index_by_entity(kb_en), index_by_entity(kb_zh)
    pairs, seen = [], set()
    for r in recs:
        en, zh = r["entity_en"].lower(), r["entity_zh"]
        if (en, zh) in seen:
            continue
        seen.add((en, zh))
        zh_forms = [zh] + [t for t in r.get("matched_translations", []) if t]
        tgt_idioms = [x for f in zh_forms for x in izh.get(f, [])]
        tgt_idioms = list({x["idiom"]: x for x in tgt_idioms}.values())
        pairs.append({"en": en, "tgt": zh, "en_idioms": ien.get(en, [])[:40], "tgt_idioms": tgt_idioms[:40],
                      "analysis": {"english_unique_aspects": r.get("en_unique_aspects", []),
                                   "target_unique_aspects": r.get("zh_unique_aspects", []),
                                   "shared_meanings": r.get("shared_meanings", [])}})
    return pairs


def xl_pairs(lang, kb_en, kb_t, top_k=400):
    ien, it = index_by_entity(kb_en), index_by_entity(kb_t)
    ents = sorted(it, key=lambda e: -len(it[e]))[:top_k]
    prompts = []
    for i in range(0, len(ents), 50):
        chunk = ents[i:i + 50]
        prompts.append(f"Translate each {LANG_NAME[lang]} noun (an entity mentioned in idioms) into the single most "
                       f"common English noun in singular, lower case. Return a JSON object mapping each input "
                       f"string exactly to its English translation.\n" + json.dumps(chunk, ensure_ascii=False))
    tr = {}
    for r in complete_many(prompts, json_mode=True, **GEN, tag=f"probe_translate_{lang}"):
        o = parse_json(r) or {}
        tr.update({k: str(v).lower().strip() for k, v in o.items() if isinstance(v, str)})
    pairs = []
    for e in ents:
        en = tr.get(e)
        if not en or len(ien.get(en, [])) < 3 or len(it[e]) < 3:
            continue
        pairs.append({"en": en, "tgt": e, "en_idioms": ien[en][:40], "tgt_idioms": it[e][:40]})
    # one pair per English entity (keep the most frequent target entity)
    best = {}
    for p in pairs:
        if p["en"] not in best:
            best[p["en"]] = p
    return list(best.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["zh", "hi", "ar"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ar_entities", default=str(B / "ar_entities.jsonl"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--stage", default="generate", choices=["generate", "verify"])
    a = ap.parse_args()
    L = LANG_NAME[a.lang]
    rng = random.Random(a.seed)
    work = B / "probe" / a.lang
    work.mkdir(parents=True, exist_ok=True)

    if a.stage == "verify":
        return verify_stage(a, L, work)
    kb_en = load_kb("en")
    kb_t = load_kb(a.lang, a.ar_entities)
    pairs = zh_pairs(kb_en, kb_t) if a.lang == "zh" else xl_pairs(a.lang, kb_en, kb_t)
    pairs = [p for p in pairs if len(p["en_idioms"]) >= 3 and len(p["tgt_idioms"]) >= 3]
    print(a.lang, "pairs", len(pairs))

    # analysis (hi/ar only; zh reuses the Sec. 4 analyses)
    todo = [p for p in pairs if "analysis" not in p]
    if todo:
        res = complete_many([ANALYZE.format(L=L, en=p["en"], tgt=p["tgt"], en_idioms=fmt(p["en_idioms"]),
                                            tgt_idioms=fmt(p["tgt_idioms"])) for p in todo],
                            json_mode=True, **GEN, tag=f"probe_analyze_{a.lang}")
        for p, r in zip(todo, res):
            p["analysis"] = parse_json(r) or {}
    pairs = [p for p in pairs if p["analysis"].get("target_unique_aspects") and p["analysis"].get("english_unique_aspects")]
    json.dump(pairs, open(work / "analyses.json", "w"), ensure_ascii=False, indent=1)

    # generation
    res = complete_many([GENERATE.format(L=L, en=p["en"], tgt=p["tgt"],
                                         tgt_unique="; ".join(map(str, p["analysis"]["target_unique_aspects"])),
                                         en_unique="; ".join(map(str, p["analysis"]["english_unique_aspects"])),
                                         shared="; ".join(map(str, p["analysis"].get("shared_meanings", []))),
                                         tgt_idioms=fmt(p["tgt_idioms"]), en_idioms=fmt(p["en_idioms"]))
                         for p in pairs], json_mode=True, **GEN, tag=f"probe_generate_v2_{a.lang}")
    items = []
    for p, r in zip(pairs, res):
        o = parse_json(r)
        if not isinstance(o, dict) or o.get("skip") or not o.get("gold") or not o.get("lure") \
                or len(o.get("distractors", [])) != 2:
            continue
        opts = [o["gold"], o["lure"]] + list(o["distractors"])
        if len({x.strip() for x in opts}) < 4:
            continue
        # grounding: >=2 evidence idioms per side that really are in the provided lists
        tset = {x["idiom"] for x in p["tgt_idioms"]}
        eset = {x["idiom"] for x in p["en_idioms"]}
        # evidence items are often copied as "idiom: meaning"; map each back to a provided idiom
        def ground(ev, pool):
            hits = set()
            for x in ev:
                x = str(x).strip()
                for idm in pool:
                    if x == idm or x.startswith(idm) or (len(idm) >= 4 and idm in x.split(":")[0]):
                        hits.add(idm)
                        break
            return hits
        ge = ground(o.get("gold_evidence", []), tset)
        le = ground(o.get("lure_evidence", []), eset)
        if len(set(ge)) < 2 or len(set(le)) < 2:
            continue
        # the entity word must not appear in any option
        if any(p["tgt"] in x for x in opts):
            continue
        order = list(range(4))
        rng.shuffle(order)
        items.append({"p": p, "o": o, "opts": [opts[k] for k in order], "order": order})
    print(a.lang, "generated", len(items))
    # style-matching rewrite of the options (same roles and order)
    res = complete_many([REWRITE.format(L=L, tgt=it["p"]["tgt"], gold=it["o"]["gold"], lure=it["o"]["lure"],
                                        d1=it["o"]["distractors"][0], d2=it["o"]["distractors"][1])
                         for it in items], json_mode=True, **GEN, tag=f"probe_rewrite_{a.lang}")
    n_rw = 0
    for it, r in zip(items, res):
        o = parse_json(r)
        if isinstance(o, dict) and o.get("gold") and o.get("lure") and len(o.get("distractors", [])) == 2:
            new = [o["gold"], o["lure"]] + list(o["distractors"])
            if len({x.strip() for x in new}) == 4 and not any(it["p"]["tgt"] in x for x in new):
                it["o_orig"] = {k: it["o"][k] for k in ("gold", "lure", "distractors")}
                it["o"].update({"gold": new[0], "lure": new[1], "distractors": new[2:]})
                it["opts"] = [new[k] for k in it["order"]]
                n_rw += 1
    print(a.lang, "rewritten", n_rw)
    json.dump(items, open(work / "items.json", "w"), ensure_ascii=False)
    return


def verify_stage(a, L, work):
    items = json.load(open(work / "items.json"))

    # blind verification by a second model family
    res = complete_many([VERIFY_PROMPT.format(L=L, en=it["p"]["en"], tgt=it["p"]["tgt"],
                                              tgt_idioms=fmt(it["p"]["tgt_idioms"]),
                                              en_idioms=fmt(it["p"]["en_idioms"]),
                                              A=it["opts"][0], B=it["opts"][1], C=it["opts"][2], D=it["opts"][3])
                         for it in items], json_mode=True, tag=f"probe_verify_v3_{a.lang}", **VERIFY)
    kept, stats = [], collections.Counter()
    for it, r in zip(items, res):
        v = parse_json(r) or {}
        gold_letter = "ABCD"[it["order"].index(0)]
        lure_letter = "ABCD"[it["order"].index(1)]
        sup = v.get("supported_in_target", {}) or {}
        dis_letters = ["ABCD"[it["order"].index(k)] for k in (2, 3)]
        ok_gold = v.get("q1") == gold_letter
        ok_lure = v.get("q2") == lure_letter
        ok_dis = not any(sup.get(x) for x in dis_letters)
        stats["gold_ok"] += ok_gold
        stats["lure_ok"] += ok_lure
        stats["dist_ok"] += ok_dis
        it["verify"] = v
        if ok_gold and ok_lure and ok_dis:
            kept.append(it)
    stats["n_generated"] = len(items)
    stats["n_kept"] = len(kept)
    # form-only shortcut baselines on the kept items (chance = 0.25)
    toks = (lambda x: set(x.strip())) if a.lang == "zh" else (lambda x: set(re.findall(r"\w+", x)))
    wf = collections.Counter(w for it in kept for o in it["opts"] for w in toks(o))
    rare = sum(min(range(4), key=lambda k: sum(wf[w] for w in toks(it["opts"][k])) / max(1, len(toks(it["opts"][k]))))
               == it["order"].index(0) for it in kept)
    longest = sum(max(range(4), key=lambda k: len(it["opts"][k])) == it["order"].index(0) for it in kept)
    stats["baseline_rare_words"] = round(rare / max(1, len(kept)), 3)
    stats["baseline_longest"] = round(longest / max(1, len(kept)), 3)
    print(a.lang, dict(stats))

    os.makedirs(a.out_dir, exist_ok=True)
    with open(Path(a.out_dir) / f"symbolism_{a.lang}.jsonl", "w", encoding="utf-8") as f:
        for i, it in enumerate(kept):
            f.write(json.dumps({
                "qid": f"symbolism_{a.lang}/{i}", "context": STEM[a.lang].format(entity=it["p"]["tgt"]),
                "options": [" " + x for x in it["opts"]], "gold": it["order"].index(0),
                "meta": {"entity": it["p"]["tgt"], "entity_en": it["p"]["en"], "lure": it["order"].index(1),
                         "distractors": [it["order"].index(2), it["order"].index(3)],
                         "gold_evidence": it["o"].get("gold_evidence", []),
                         "lure_evidence": it["o"].get("lure_evidence", [])}}, ensure_ascii=False) + "\n")
    json.dump({"stats": dict(stats), "items": [{"entity": it["p"]["tgt"], "en": it["p"]["en"], "gen": it["o"],
                                                 "verify": it.get("verify")} for it in items]},
              open(work / "generation_log.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
