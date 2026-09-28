#!/usr/bin/env python3
"""Automatic audit of IdiomAtlas entries by two LLM judges from different families,
plus annotation sheets for the human validation.

For 200 random entries per language, each judge answers, from the entry alone:
  fig_ok      are the listed figurative meanings figurative/idiomatic senses of the
              expression (not literal paraphrases, etymology, or citations)?  yes/partly/no
  lit_ok      if literal meanings are listed, are they literal readings?     yes/partly/no/na
  ent_prec    share of listed entities that are concrete nouns occurring in the expression
  ent_missing concrete nouns in the expression that the entity list misses
Agreement between the judges (Cohen's kappa on fig_ok/lit_ok) and the per-judge rates
are written to <out_dir>/audit_summary.json; the per-item judgments and a CSV sheet
for native-speaker annotators are written next to it.

Usage (run twice, one judge per call, then summarize):
  python -m culture.bidirectional.kb_audit judge --judge_name gemma --model $GEMMA --out_dir $B/audit
  python -m culture.bidirectional.kb_audit judge --judge_name qwen27b --model $QWEN --out_dir $B/audit
  python -m culture.bidirectional.kb_audit summarize --out_dir $B/audit
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

from culture.bidirectional.llm_api import complete_many, parse_json

REPO = Path(__file__).resolve().parents[3]
D = REPO / "culture/data"
B = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
KB = {
    "en": D / "idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl",
    "zh": D / "idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl",
    "hi": D / "idioms/hi/idioms_merged_llm_formatted_figurative_only.jsonl",
    "ar": B / "hf9b/data/idioms/ar/idioms_merged_llm_formatted.jsonl",
}
LANG = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}

PROMPT = """You are auditing an entry of a {lang} idiom/proverb dictionary. Judge it using your knowledge of {lang}.

Expression: {idiom}
Figurative meanings listed: {fig}
Literal meanings listed: {lit}
Entities listed (concrete nouns said to occur in the expression): {ent}

Questions:
1. fig_ok: Are the listed figurative meanings genuine figurative/idiomatic senses of the expression (not literal paraphrases, etymology, or source citations)? Answer "yes", "partly", or "no".
2. lit_ok: If literal meanings are listed, are they literal readings of the words of the expression? Answer "yes", "partly", "no", or "na" if none are listed.
3. ent_correct: How many of the listed entities are concrete nouns that actually occur in the expression? Give an integer.
4. ent_missing: List concrete nouns that occur in the expression but are missing from the entity list (empty list if none).
Return JSON only: {{"fig_ok": "...", "lit_ok": "...", "ent_correct": <int>, "ent_missing": [...]}}"""


def _lst(v):
    if not v or v == "NAN":
        return []
    return v if isinstance(v, list) else [v]


def sample(lang, n, seed):
    rows = [json.loads(l) for l in open(KB[lang], encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        o = r["output"]
        out.append({"lang": lang, "idiom": o["idiom"], "fig": [str(x) for x in _lst(o.get("figurative_meanings"))][:4],
                    "lit": [str(x) for x in _lst(o.get("literal_meanings"))][:3],
                    "ent": [str(x) for x in _lst(o.get("entities"))]})
    return out


def kappa(a, b):
    labs = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "summarize"])
    ap.add_argument("--judge_name")
    ap.add_argument("--model")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = [x for L in ["en", "zh", "hi", "ar"] for x in sample(L, a.n, a.seed)]
    if a.cmd == "judge":
        prompts = [PROMPT.format(lang=LANG[x["lang"]], idiom=x["idiom"], fig=json.dumps(x["fig"], ensure_ascii=False),
                                 lit=json.dumps(x["lit"], ensure_ascii=False) if x["lit"] else "(none)",
                                 ent=json.dumps(x["ent"], ensure_ascii=False)) for x in items]
        res = complete_many(prompts, provider="vllm", model=a.model, max_tokens=300, tag=f"kb_audit_{a.judge_name}")
        with open(out / f"judge_{a.judge_name}.jsonl", "w", encoding="utf-8") as f:
            for x, r in zip(items, res):
                f.write(json.dumps({**x, "judgment": parse_json(r) or {}, "raw": r}, ensure_ascii=False) + "\n")
        # human annotation sheet (same questions, blank answers)
        with open(out / "human_sheet.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["lang", "idiom", "figurative_meanings", "literal_meanings", "entities",
                        "fig_ok(yes/partly/no)", "lit_ok(yes/partly/no/na)", "ent_correct(int)", "ent_missing",
                        "ocr_fidelity_hi(yes/no)", "ar_literal_reading_ok(yes/partly/no)", "notes"])
            for x in items:
                w.writerow([x["lang"], x["idiom"], " | ".join(x["fig"]), " | ".join(x["lit"]), " | ".join(x["ent"])]
                           + [""] * 7)
        return
    judges = sorted(p.stem.replace("judge_", "") for p in out.glob("judge_*.jsonl"))
    J = {j: [json.loads(l) for l in open(out / f"judge_{j}.jsonl", encoding="utf-8")] for j in judges}
    summ = {"judges": judges, "per_lang": {}}
    for L in ["en", "zh", "hi", "ar"]:
        s = {}
        for j in judges:
            rows = [r for r in J[j] if r["lang"] == L]
            g = [r["judgment"] for r in rows]
            fig = [x.get("fig_ok") for x in g]
            lit = [x.get("lit_ok") for x in g if x.get("lit_ok") not in (None, "na")]
            n_ent = sum(len(r["ent"]) for r in rows)
            n_cor = sum(min(int(x.get("ent_correct", 0) or 0), len(r["ent"])) for r, x in zip(rows, g))
            n_mis = sum(len(x.get("ent_missing", []) or []) for x in g)
            s[j] = {"n": len(rows), "fig_yes": fig.count("yes") / len(fig), "fig_yes_or_partly":
                    (fig.count("yes") + fig.count("partly")) / len(fig),
                    "lit_n": len(lit), "lit_yes": (lit.count("yes") / len(lit)) if lit else None,
                    "entity_precision": n_cor / max(1, n_ent),
                    "entity_recall": n_cor / max(1, n_cor + n_mis)}
        if len(judges) >= 2:
            a0 = [r["judgment"].get("fig_ok") for r in J[judges[0]] if r["lang"] == L]
            a1 = [r["judgment"].get("fig_ok") for r in J[judges[1]] if r["lang"] == L]
            s["kappa_fig"] = kappa([str(x) for x in a0], [str(x) for x in a1])
            s["agree_fig"] = sum(x == y for x, y in zip(a0, a1)) / len(a0)
        summ["per_lang"][L] = s
    json.dump(summ, open(out / "audit_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
