#!/usr/bin/env python3
"""T2, generalised: write what each entity symbolizes, from the knowledge base only.

`ar_symbolism_summaries.py` did this for Arabic with the symbolism-probe entities held out.
The pass-3 log names two things that generalisation cannot be separated from without more
arms, and this script produces the text for both:

  * ``--lang hi``  the same condition in Hindi, the language where idiom-bearing documents
    already move the symbolism-probe lure rate (\\S5.5), so a null there is a stronger null;
  * ``--include_probe``  the same condition in Arabic *without* holding out the 98 probe
    entities.  If the probe still does not move when the training text states exactly what
    the probe asks about, the layer resists continued pretraining; if it moves, the T2 null
    is a failure to generalise, not a failure to learn.

    PYTHONPATH=src:src/culture/analysis/v2 python t2_symbolism_summaries.py --lang hi
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import common  # noqa: E402
import local_llm  # noqa: E402

DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")
MC = f"{DATA}/eval/mc"

PROMPTS = {
    "ar": """Below are Arabic proverbs that all mention the same concrete noun, together with the \
figurative meaning each proverb has according to Arabic lexicographers.

Noun: {entity}

Proverbs:
{idioms}

In ONE or TWO short sentences, written in Modern Standard Arabic, state what this noun \
typically stands for or connotes in Arabic proverbs: the qualities, roles, or judgements it is \
used to express. Write only what the proverbs above support; if they do not agree on anything, \
say so briefly. Do not invent proverbs, do not list the proverbs, and do not explain any single \
proverb. Reply with the Arabic sentences only.""",
    "hi": """Below are Hindi proverbs and sayings that all mention the same concrete noun, \
together with the figurative meaning each one has according to Hindi dictionaries.

Noun: {entity}

Proverbs:
{idioms}

In ONE or TWO short sentences, written in Hindi (Devanagari), state what this noun typically \
stands for or connotes in Hindi proverbs: the qualities, roles, or judgements it is used to \
express. Write only what the proverbs above support; if they do not agree on anything, say so \
briefly. Do not invent proverbs, do not list the proverbs, and do not explain any single \
proverb. Reply with the Hindi sentences only.""",
}

# the probe file whose entities are held out unless --include_probe
PROBE = {"ar": "symbolism_v2_ar_letter.jsonl", "hi": "symbolism_v2_hi_letter.jsonl"}


def clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def probe_entities(lang):
    ents = set()
    p = os.path.join(MC, PROBE[lang])
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            e = (json.loads(line).get("meta") or {}).get("entity")
            if e:
                ents.add(common.norm_entity(e, lang))
    return {e for e in ents if e}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ar", "hi"], required=True)
    ap.add_argument("--min_idioms", type=int, default=4)
    ap.add_argument("--max_show", type=int, default=12)
    ap.add_argument("--include_probe", action="store_true",
                    help="do NOT hold out the symbolism-probe entities")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lang = args.lang
    out_dir = os.environ.get("CULTURE_T2_DIR", f"{DATA}/t2_{lang}")
    out_name = args.out or (
        f"{lang}_entity_symbolism{'_all' if args.include_probe else ''}.json")

    kb = common.load_kb(lang)
    ent_rows = defaultdict(list)
    for r in kb:
        figs = [clean(f) for f in common.flatten(r["fig"]) if clean(f)]
        if not figs:
            continue
        for e in r["entities"]:
            ent_rows[e].append((r["idiom"], figs[0]))

    held = set() if args.include_probe else probe_entities(lang)
    print(f"[probe] {len(probe_entities(lang))} probe entities, "
          f"{'INCLUDED' if args.include_probe else 'excluded'}")

    ents = sorted((e for e, v in ent_rows.items()
                   if len(v) >= args.min_idioms and e not in held),
                  key=lambda e: -len(ent_rows[e]))
    print(f"[entities] {len(ents)} with >= {args.min_idioms} idioms")

    prompts = []
    for e in ents:
        rows = ent_rows[e][: args.max_show]
        block = "\n".join(f"- {i} :: {t[:220]}" for i, t in rows)
        prompts.append(PROMPTS[lang].format(entity=e, idioms=block))

    resp = local_llm.generate(prompts, f"{lang}_entity_symbolism", max_tokens=160)
    out = {}
    for e, r in zip(ents, resp):
        t = clean(r)
        # drop refusals / English leakage / over-long output (same filter as the ar pass)
        if len(t) < 20 or len(t) > 600:
            continue
        if sum(ch.isascii() and ch.isalpha() for ch in t) > 0.3 * len(t):
            continue
        out[e] = {"symbolism": t, "n_idioms": len(ent_rows[e])}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_name)
    json.dump({"lang": lang, "include_probe": args.include_probe,
               "excluded_probe_entities": sorted(held), "entities": out},
              open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[write] {path}: {len(out)}/{len(ents)} kept")
    for e in ents[:3]:
        if e in out:
            print(f"  {e}: {out[e]['symbolism'][:180]}")


if __name__ == "__main__":
    main()
