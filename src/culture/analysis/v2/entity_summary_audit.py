#!/usr/bin/env python3
"""Are the Section-4 entity summaries supported by the idioms they were written from?

The dog and dragon rows of Table~\\ref{tab:entity-cases} claimed things no idiom in the
knowledge base says -- English "loyalty" for the dog, a mythical destroyer for the dragon --
because the summarising model reached for what it already believed about the culture despite
being told to use only the evidence shown. Those two were caught by hand. This checks the
other several hundred the same way, automatically.

For every claim in a summary (each English and Chinese primary meaning, and each
language-unique aspect) we show a judge the claim and the idioms of that entity in that
language, and ask whether at least one idiom supports it. The judge is a different model
family from the one that wrote the summaries, and is told to answer "no" when unsure, so the
unsupported rate is a lower bound on agreement rather than on error.

    PYTHONPATH=src:src/culture/analysis/v2 python entity_summary_audit.py [--max_entities N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import common
import local_llm

SRC = os.path.join(
    os.environ.get("CULTURE_REPO", common.REPO),
    "culture/data/idioms/cross_lingual_analysis/cultural_analysis_results.jsonl",
)
FIELDS = [
    ("english_primary_meanings", "en"),
    ("english_unique_aspects", "en"),
    ("chinese_primary_meanings", "zh"),
    ("chinese_unique_aspects", "zh"),
]
LANG_NAME = {"en": "English", "zh": "Chinese"}

PROMPT = """Below are {n} {lang} idioms that contain the entity "{ent}", each with its meaning.

{evidence}

Someone has summarised what "{ent}" stands for in {lang} idioms, and one of their claims is:

    "{claim}"

Is that claim supported by at least one of the idioms above? Judge only from the idioms shown;
do not use anything else you know about {lang} or about {ent}. If no idiom clearly supports the
claim, answer no.

Answer with one word, "yes" or "no"."""


def _parse(ca):
    if isinstance(ca, dict):
        return ca
    m = re.search(r"\{.*\}", ca or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _evidence(idioms, cap=20):
    out = []
    for r in idioms[:cap]:
        # Some entries nest their meanings one list deeper.
        parts = common.flatten(r.get("figurative_meanings")) or \
            common.flatten(r.get("literal_meanings"))
        mean = "; ".join(str(x) for x in parts[:2])
        if r.get("idiom") and mean:
            mean = re.sub(r"\s+", " ", mean)[:150]
            out.append(f"- {r['idiom']} — {mean}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_entities", type=int, default=0)
    ap.add_argument("--model", choices=["primary", "third", "fourth"], default="fourth")
    ap.add_argument("--out", default="entity_summary_audit.json")
    args = ap.parse_args()
    model = {"primary": local_llm.PRIMARY, "third": local_llm.THIRD,
             "fourth": local_llm.FOURTH}[args.model]

    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    if args.max_entities:
        rows = rows[: args.max_entities]

    prompts, meta = [], []
    for r in rows:
        d = _parse(r.get("cultural_analysis"))
        if not d:
            continue
        ev = {"en": _evidence(r.get("idioms_en") or []),
              "zh": _evidence(r.get("idioms_zh") or [])}
        n = {"en": len(r.get("idioms_en") or []), "zh": len(r.get("idioms_zh") or [])}
        for field, lang in FIELDS:
            for claim in (d.get(field) or []):
                if not isinstance(claim, str) or len(claim) < 8 or not ev[lang]:
                    continue
                prompts.append(PROMPT.format(
                    n=min(n[lang], 20), lang=LANG_NAME[lang],
                    ent=r["entity_en"] if lang == "en" else r.get("entity_zh", ""),
                    evidence=ev[lang], claim=claim.strip()))
                meta.append({"entity_en": r["entity_en"], "field": field,
                             "lang": lang, "claim": claim.strip()})

    print(f"[audit] {len(prompts)} claims over {len(rows)} entities")
    outs = local_llm.generate(prompts, f"entity_summary_audit_{args.model}",
                              model=model, max_tokens=4)

    by_field, unsupported = Counter(), []
    tot_field = Counter()
    per_entity = defaultdict(lambda: [0, 0])
    for m, o in zip(meta, outs):
        yes = (o or "").strip().lower().startswith("y")
        tot_field[m["field"]] += 1
        per_entity[m["entity_en"]][1] += 1
        if yes:
            by_field[m["field"]] += 1
            per_entity[m["entity_en"]][0] += 1
        else:
            unsupported.append(m)

    worst = sorted(per_entity.items(), key=lambda kv: (kv[1][0] / max(1, kv[1][1])))[:25]
    out = {
        "method": __doc__,
        "source": os.path.basename(SRC),
        "judge": local_llm.NAMES.get(model, model),
        "n_entities": len(per_entity),
        "n_claims": len(meta),
        "supported_overall": round(sum(by_field.values()) / max(1, len(meta)), 4),
        "supported_by_field": {f: round(by_field[f] / t, 4)
                               for f, t in tot_field.items()},
        "n_claims_by_field": dict(tot_field),
        "worst_entities": [{"entity": e, "supported": s, "claims": t} for e, (s, t) in worst],
        "unsupported_claims": unsupported[:400],
    }
    print("wrote", common.dump(out, args.out))
    print(f"\nsupported overall: {out['supported_overall']:.1%} of {len(meta)} claims")
    for f, v in out["supported_by_field"].items():
        print(f"  {f:28s} {v:.1%}  (n={tot_field[f]})")
    print("\nentities with the least supported summaries:")
    for w in out["worst_entities"][:10]:
        print(f"  {w['entity']:18s} {w['supported']}/{w['claims']} supported")


if __name__ == "__main__":
    main()
