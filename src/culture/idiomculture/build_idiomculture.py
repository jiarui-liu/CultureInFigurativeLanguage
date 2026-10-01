#!/usr/bin/env python3
"""IdiomCulture: an in-domain culture benchmark derived from the cultural knowledge in idioms.

Idioms presuppose and encode culture-specific knowledge (what an entity symbolizes, which
behaviour is valued, how a social relation works, which practice or belief is assumed).
This pipeline extracts that knowledge from IdiomAtlas, consolidates it into culture
aspects, and turns each aspect into a multiple-choice question in the target language
that can be answered from cultural knowledge alone: no idiom, proverb, or saying appears
in the question or the options, and no question asks what an idiom means.

Stages (each reads the previous stage's output under --work and is resumable; all LLM
calls go through culture.bidirectional.llm_api, which caches every response on disk):

  exposure     local, no API. Per idiom: seen (in the idiom-annotated CPT corpus) /
               unseen (in no scanned web document) / other. Run on the training server.
  extract      API (generator). Per idiom: 0-3 culture aspects it presupposes or encodes.
  consolidate  local. Embed aspects, merge near-duplicates across idioms, keep evidence.
  questions    API (generator). One 4-option MC question per aspect, in the target language.
  verify       API (verifier, a different model family). Blind answer + quality judgment.
  export       local. Leakage / bias filters, balanced sampling, MC jsonl for run_eval.

Usage (other server, API keys in the environment):
  export LLM_CACHE_DIR=/path/to/llm_cache
  python -m culture.idiomculture.build_idiomculture --stage extract --lang zh \
      --work /path/to/idiomculture --gen_provider gemini --gen_model gemini-3.8-pro
  ... --stage consolidate / questions / verify (--ver_provider openrouter --ver_model ...) / export
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
D = REPO / "culture/data"
KB = {
    "zh": D / "idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl",
    "hi": D / "idioms/hi/idioms_merged_llm_formatted_figurative_only.jsonl",
    "ar": D / "idioms/ar/idioms_merged_llm_formatted.jsonl",
}
EXPOSURE = D / "idiomculture"
LANG = {"zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
COMMUNITY = {"zh": "Chinese culture", "hi": "Indian (Hindi-speaking) culture", "ar": "Arab culture"}
ANSWER_CUE = {"zh": "答案：", "hi": "उत्तर:", "ar": "الإجابة:"}
ASPECT_TYPES = ["value_norm", "belief_symbolism", "practice_custom", "social_relation",
                "material_culture", "history_religion", "environment_livelihood"]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)


def flat(x):
    if isinstance(x, (list, tuple)):
        return [s for y in x for s in flat(y)]
    return [str(x).strip()] if x not in (None, "") else []


def kb_rows(lang):
    out = []
    for r in read_jsonl(KB[lang]):
        o = r["output"]
        if o.get("figurative_meanings"):
            out.append({"idiom": o["idiom"].strip(), "entities": flat(o.get("entities", [])),
                        "literal": flat(o.get("literal_meanings", [])),
                        "figurative": flat(o["figurative_meanings"])})
    return out


# --------------------------------------------------------------------------- #
# exposure (training server only)
# --------------------------------------------------------------------------- #
def stage_exposure(a):
    from culture.bidirectional.build_idiomatlas_mc import seen_counts
    seen, ever = seen_counts(a.lang, a.ar_counts)
    lab = {r["idiom"]: ("seen" if r["idiom"] in seen else "unseen" if r["idiom"] not in ever else "other")
           for r in kb_rows(a.lang)}
    os.makedirs(EXPOSURE, exist_ok=True)
    json.dump(lab, open(EXPOSURE / f"exposure_{a.lang}.json", "w"), ensure_ascii=False)
    print(a.lang, Counter(lab.values()))


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
EXTRACT_PROMPT = """You are an expert on {community}. Below is a {lang} idiom with its literal and figurative meanings.

Idiom: {idiom}
Entities: {entities}
Literal meaning: {literal}
Figurative meaning: {figurative}

Identify the pieces of CULTURAL KNOWLEDGE that this idiom presupposes or encodes: knowledge about {community} that a person must have for the idiom to make sense, or that the idiom expresses about how people in this culture think and live. Examples of kinds: what an entity symbolizes or connotes; a value or social norm; a custom, ritual, or practice; a social role or relationship; an object, food, or material item and its cultural role; a historical, religious, or literary reference; a feature of the environment or livelihood.

Rules:
- Each aspect must be a self-contained factual statement about the culture, written WITHOUT quoting or paraphrasing the idiom, and it must NOT be a restatement of the idiom's meaning. Bad: "The idiom means one should be patient." Good: "In traditional Chinese culture, the crane symbolizes longevity."
- Only include knowledge that is specific to, or distinctively expressed in, {community}. For each aspect, judge whether it is culture_specific (differs from or is absent in Anglophone/Western culture) or universal (shared by most cultures).
- Do not invent facts. If the idiom presupposes no specific cultural knowledge (e.g., it relies only on universal experience), return an empty list.
- At most 3 aspects.

Return JSON only:
{{"aspects": [{{"statement_en": "...", "statement_native": "<same statement in {lang}>", "type": one of {types}, "specificity": "culture_specific" | "universal", "key_entity": "<entity in {lang} or empty>", "confidence": 1-5}}]}}"""


def stage_extract(a):
    from culture.bidirectional.llm_api import complete_many, parse_json
    rows = kb_rows(a.lang)
    expo = json.load(open(EXPOSURE / f"exposure_{a.lang}.json"))
    rng = random.Random(0)
    # stratify: all unseen idioms (they test generalization) + a random sample of the rest
    by = defaultdict(list)
    for r in rows:
        by[expo.get(r["idiom"], "other")].append(r)
    pick = []
    for split, n in (("unseen", a.n_unseen), ("seen", a.n_seen), ("other", a.n_other)):
        pool = by[split][:]
        rng.shuffle(pool)
        pick += [dict(r, exposure=split) for r in pool[:n]]
    prompts = [EXTRACT_PROMPT.format(community=COMMUNITY[a.lang], lang=LANG[a.lang], idiom=r["idiom"],
                                     entities=", ".join(r["entities"]) or "-",
                                     literal="; ".join(r["literal"]) or "-",
                                     figurative="; ".join(r["figurative"]), types=ASPECT_TYPES)
               for r in pick]
    outs = complete_many(prompts, workers=a.workers, provider=a.gen_provider, model=a.gen_model,
                         temperature=0.0, json_mode=True, tag=f"ic_extract_{a.lang}")
    res = []
    for r, o in zip(pick, outs):
        try:
            asp = parse_json(o).get("aspects", [])
        except Exception:
            continue
        for i, x in enumerate(asp[:3]):
            if x.get("type") in ASPECT_TYPES and x.get("statement_en") and x.get("statement_native") \
                    and int(x.get("confidence", 0)) >= a.min_conf:
                res.append({"aid": f"{a.lang}/{len(res)}", "idiom": r["idiom"], "exposure": r["exposure"], **x})
    write_jsonl(f"{a.work}/{a.lang}/aspects_raw.jsonl", res)
    print(a.lang, "idioms", len(pick), "aspects", len(res), Counter(x["type"] for x in res))


# --------------------------------------------------------------------------- #
# consolidate
# --------------------------------------------------------------------------- #
def stage_consolidate(a):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    rows = read_jsonl(f"{a.work}/{a.lang}/aspects_raw.jsonl")
    emb = SentenceTransformer(a.emb_model).encode([r["statement_en"] for r in rows],
                                                  normalize_embeddings=True, batch_size=64,
                                                  show_progress_bar=True)
    # greedy clustering: an aspect joins the first cluster whose centroid is within threshold
    cents, members = [], []
    for i, e in enumerate(emb):
        if cents:
            sims = np.array(cents) @ e
            j = int(sims.argmax())
            if sims[j] >= a.merge_sim:
                members[j].append(i)
                c = emb[members[j]].mean(0)
                cents[j] = c / np.linalg.norm(c)
                continue
        cents.append(e)
        members.append([i])
    out = []
    for k, m in enumerate(members):
        rs = [rows[i] for i in m]
        # representative: the member closest to the centroid
        rep = rs[int((emb[m] @ cents[k]).argmax())]
        expos = {r["exposure"] for r in rs}
        out.append({"cid": f"{a.lang}/c{k}", "statement_en": rep["statement_en"],
                    "statement_native": rep["statement_native"], "type": rep["type"],
                    "specificity": Counter(r["specificity"] for r in rs).most_common(1)[0][0],
                    "key_entity": rep.get("key_entity", ""),
                    "evidence_idioms": sorted({r["idiom"] for r in rs}),
                    "n_idioms": len({r["idiom"] for r in rs}),
                    "exposure": "seen" if "seen" in expos else ("unseen" if expos == {"unseen"} else "other")})
    write_jsonl(f"{a.work}/{a.lang}/aspects.jsonl", out)
    print(a.lang, "aspects", len(rows), "-> clusters", len(out),
          "multi-idiom", sum(o["n_idioms"] > 1 for o in out), Counter(o["exposure"] for o in out))


# --------------------------------------------------------------------------- #
# questions
# --------------------------------------------------------------------------- #
QUESTION_PROMPT = """You write test questions about {community} for native {lang} speakers.

Cultural fact (verified from the culture's idioms; do not mention idioms):
EN: {statement_en}
{lang}: {statement_native}

Write ONE multiple-choice question in {lang} that tests whether someone knows this cultural fact.

Rules:
- The question and all options must be in natural {lang}. Do NOT quote, paraphrase, or allude to any idiom, proverb, or saying, and do NOT ask what an expression means. Ask about the culture itself (what something symbolizes, what is valued or expected, what people customarily do, what an object or role signifies, etc.).
- The question must be answerable from cultural knowledge alone and must not reveal the answer.
- Exactly four options of similar length and style, exactly one correct:
  * "gold": the correct answer, faithful to the fact.
  * "lure": what a person reasoning from Anglophone/Western culture would plausibly answer, but which is wrong for {community}. If no such contrast exists, write another plausible wrong answer.
  * "d1", "d2": two other plausible but wrong answers for {community}.
- Avoid options that are trivially wrong, that overlap in meaning with the gold, or that contain giveaways (e.g. only the gold mentions the culture by name).

Return JSON only:
{{"question": "...", "gold": "...", "lure": "...", "d1": "...", "d2": "...", "lure_is_anglophone": true|false}}"""


def stage_questions(a):
    from culture.bidirectional.llm_api import complete_many, parse_json
    rows = read_jsonl(f"{a.work}/{a.lang}/aspects.jsonl")
    prompts = [QUESTION_PROMPT.format(community=COMMUNITY[a.lang], lang=LANG[a.lang],
                                      statement_en=r["statement_en"], statement_native=r["statement_native"])
               for r in rows]
    outs = complete_many(prompts, workers=a.workers, provider=a.gen_provider, model=a.gen_model,
                         temperature=0.0, json_mode=True, tag=f"ic_questions_{a.lang}")
    res = []
    for r, o in zip(rows, outs):
        try:
            q = parse_json(o)
        except Exception:
            continue
        if all(q.get(k) for k in ("question", "gold", "lure", "d1", "d2")):
            res.append({**r, **{k: q[k] for k in ("question", "gold", "lure", "d1", "d2")},
                        "lure_is_anglophone": bool(q.get("lure_is_anglophone"))})
    write_jsonl(f"{a.work}/{a.lang}/questions.jsonl", res)
    print(a.lang, "questions", len(res))


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
ANSWER_PROMPT = """Answer the following multiple-choice question about {community}. Reply with the letter only.

{question}
{options}"""

JUDGE_PROMPT = """You are reviewing a test question about {community} written in {lang}.

Question: {question}
Options:
{options}
Claimed correct option: {gold_letter}

Judge each criterion strictly and return JSON only:
{{"answerable_without_idiom": true|false,   // answerable from cultural knowledge; no idiom/proverb/saying is quoted or alluded to, and it does not ask what an expression means
 "gold_correct": true|false,                 // the claimed option is factually correct for {community}
 "single_correct": true|false,               // no other option is also acceptable
 "culture_specific": true|false,             // the answer differs from what one would assume from Anglophone/Western culture
 "no_giveaway": true|false,                  // the correct option is not identifiable from form alone (length, wording, naming the culture)
 "fluent": true|false,                       // natural {lang}
 "note": "<one short sentence>"}}"""


def letters(opts):
    return "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(opts))


def stage_verify(a):
    from culture.bidirectional.llm_api import complete_many, parse_json
    rows = read_jsonl(f"{a.work}/{a.lang}/questions.jsonl")
    rng = random.Random(1)
    for r in rows:
        opts = [("gold", r["gold"]), ("lure", r["lure"]), ("d1", r["d1"]), ("d2", r["d2"])]
        rng.shuffle(opts)
        r["order"] = [k for k, _ in opts]
        r["options"] = [v for _, v in opts]
        r["gold_idx"] = r["order"].index("gold")
    ans = complete_many([ANSWER_PROMPT.format(community=COMMUNITY[a.lang], question=r["question"],
                                              options=letters(r["options"])) for r in rows],
                        workers=a.workers, provider=a.ver_provider, model=a.ver_model, temperature=0.0,
                        tag=f"ic_answer_{a.lang}")
    jud = complete_many([JUDGE_PROMPT.format(community=COMMUNITY[a.lang], lang=LANG[a.lang],
                                             question=r["question"], options=letters(r["options"]),
                                             gold_letter="ABCD"[r["gold_idx"]]) for r in rows],
                        workers=a.workers, provider=a.ver_provider, model=a.ver_model, temperature=0.0,
                        json_mode=True, tag=f"ic_judge_{a.lang}")
    for r, x, j in zip(rows, ans, jud):
        m = re.search(r"[ABCD]", x or "")
        r["ver_answer"] = m.group(0) if m else None
        r["ver_correct"] = r["ver_answer"] == "ABCD"[r["gold_idx"]]
        r["ver_picked"] = r["order"]["ABCD".index(r["ver_answer"])] if m else None
        try:
            r["judge"] = parse_json(j)
        except Exception:
            r["judge"] = {}
    write_jsonl(f"{a.work}/{a.lang}/verified.jsonl", rows)
    keys = ["answerable_without_idiom", "gold_correct", "single_correct", "culture_specific", "no_giveaway", "fluent"]
    print(a.lang, "verifier accuracy", sum(r["ver_correct"] for r in rows) / max(1, len(rows)),
          {k: sum(bool(r["judge"].get(k)) for r in rows) for k in keys})


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def units(lang, s):
    s = re.sub(r"[ً-ٰٟ]", "", s)
    return set(s) - set(" ，。、；：！？,.;:!?\"'()（）《》「」") if lang == "zh" else set(re.findall(r"\w+", s))


def stage_export(a):
    from culture.bidirectional.build_pool import make_matcher
    rows = read_jsonl(f"{a.work}/{a.lang}/verified.jsonl")
    matcher = make_matcher(a.lang)
    keys = ["answerable_without_idiom", "gold_correct", "single_correct", "fluent", "no_giveaway"]
    drop = Counter()
    keep = []
    for r in rows:
        if not all(r["judge"].get(k) for k in keys):
            drop["judge"] += 1
            continue
        if a.require_verifier and not r["ver_correct"]:
            drop["verifier_wrong"] += 1
            continue
        if any(matcher.match(t) for t in [r["question"]] + r["options"]):
            drop["idiom_leak"] += 1
            continue
        # no option may share most of its words with an evidence idiom
        ev = set().union(*(units(a.lang, i) for i in r["evidence_idioms"]))
        if any(len(units(a.lang, o) & ev) / max(1, len(units(a.lang, o))) > a.max_overlap for o in r["options"]):
            drop["lexical_overlap"] += 1
            continue
        r["split_specificity"] = "culture_specific" if r["judge"].get("culture_specific") else "universal"
        keep.append(r)
    # form-only baselines on the kept set should be near chance (25%)
    longest = sum(max(range(4), key=lambda i: len(r["options"][i])) == r["gold_idx"] for r in keep)
    print(a.lang, "kept", len(keep), "dropped", dict(drop),
          "longest-option baseline", round(longest / max(1, len(keep)), 3),
          Counter((r["exposure"], r["split_specificity"]) for r in keep))
    cue = ANSWER_CUE[a.lang]
    out_letter, out_cont = [], []
    for k, r in enumerate(keep):
        meta = {"cid": r["cid"], "type": r["type"], "exposure": r["exposure"],
                "specificity": r["split_specificity"], "n_idioms": r["n_idioms"],
                "evidence_idioms": r["evidence_idioms"], "order": r["order"],
                "lure_is_anglophone": r["lure_is_anglophone"], "statement_en": r["statement_en"]}
        qid = f"idiomculture_{a.lang}/{k}"
        out_letter.append({"qid": qid, "context": f"{r['question']}\n{letters(r['options'])}\n{cue}",
                           "options": [" A", " B", " C", " D"], "gold": r["gold_idx"], "meta": meta})
        out_cont.append({"qid": qid, "context": f"{r['question']}\n{cue}",
                         "options": [" " + o for o in r["options"]], "gold": r["gold_idx"], "meta": meta})
    os.makedirs(a.out_dir, exist_ok=True)
    write_jsonl(f"{a.out_dir}/idiomculture_{a.lang}_letter.jsonl", out_letter)
    write_jsonl(f"{a.out_dir}/idiomculture_{a.lang}.jsonl", out_cont)
    write_jsonl(f"{a.work}/{a.lang}/human_sheet.jsonl",
                [{"qid": o["qid"], "question": r["question"], "options": letters(r["options"]),
                  "claimed": "ABCD"[r["gold_idx"]], "statement_en": r["statement_en"],
                  "native_ok": "", "culture_specific_ok": "", "comment": ""}
                 for o, r in zip(out_letter, keep)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["exposure", "extract", "consolidate", "questions", "verify", "export"])
    ap.add_argument("--lang", required=True, choices=["zh", "hi", "ar"])
    ap.add_argument("--work", default="idiomculture_work")
    ap.add_argument("--out_dir", default="idiomculture_mc")
    ap.add_argument("--ar_counts", nargs="*", default=[])
    ap.add_argument("--n_unseen", type=int, default=2000)
    ap.add_argument("--n_seen", type=int, default=2000)
    ap.add_argument("--n_other", type=int, default=1000)
    ap.add_argument("--min_conf", type=int, default=4)
    ap.add_argument("--emb_model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--merge_sim", type=float, default=0.85)
    ap.add_argument("--gen_provider", default="gemini")
    ap.add_argument("--gen_model", default="gemini-3.8-pro")
    ap.add_argument("--ver_provider", default="openrouter")
    ap.add_argument("--ver_model", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--require_verifier", type=int, default=1)
    ap.add_argument("--max_overlap", type=float, default=0.5)
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
