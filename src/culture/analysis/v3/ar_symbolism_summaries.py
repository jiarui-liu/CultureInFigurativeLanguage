#!/usr/bin/env python3
"""Write, for each Arabic entity, one or two sentences stating what it symbolizes in Arabic
proverbs -- grounded only in the proverbs of \\dataname{} that contain it.

This is the generator the paper calls for at the end of \\S5: ``a notes generator aimed at
evaluative associations rather than definitions''. The existing meaning tags state what an
expression means; these statements say what an *image* stands for, which is the layer
\\S5.4 finds in idioms and absent from the culture corpus.

Two design choices keep the downstream experiment honest:
  * the model sees only the idioms of that entity and their figurative meanings, and is told
    to say nothing the idioms do not support (the same instruction, and the same failure mode,
    as the \\S4 entity summaries);
  * every entity used by the Arabic symbolism probe is **excluded**, so the probe stays a test
    of generalisation to entities the training text never discusses.

    PYTHONPATH=src:src/culture/analysis/v2 python ar_symbolism_summaries.py
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
OUT = os.environ.get("CULTURE_T2_DIR", f"{DATA}/t2_ar")

PROMPT = """Below are Arabic proverbs that all mention the same concrete noun, together with the \
figurative meaning each proverb has according to Arabic lexicographers.

Noun: {entity}

Proverbs:
{idioms}

In ONE or TWO short sentences, written in Modern Standard Arabic, state what this noun \
typically stands for or connotes in Arabic proverbs: the qualities, roles, or judgements it is \
used to express. Write only what the proverbs above support; if they do not agree on anything, \
say so briefly. Do not invent proverbs, do not list the proverbs, and do not explain any single \
proverb. Reply with the Arabic sentences only."""


def clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_idioms", type=int, default=4)
    ap.add_argument("--max_show", type=int, default=12)
    ap.add_argument("--out", default="ar_entity_symbolism.json")
    args = ap.parse_args()

    kb = common.load_kb("ar")
    ent_rows = defaultdict(list)
    for r in kb:
        figs = [clean(f) for f in common.flatten(r["fig"]) if clean(f)]
        if not figs:
            continue
        for e in r["entities"]:
            ent_rows[e].append((r["idiom"], figs[0]))

    # entities the symbolism probe asks about -- excluded so the probe stays held out
    probe_ents = set()
    p = f"{MC}/symbolism_v2_ar_letter.jsonl"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            e = (json.loads(line).get("meta") or {}).get("entity")
            if e:
                probe_ents.add(common.norm_entity(e, "ar"))
    print(f"[probe] excluding {len(probe_ents)} probe entities")

    ents = sorted((e for e, v in ent_rows.items()
                   if len(v) >= args.min_idioms and e not in probe_ents),
                  key=lambda e: -len(ent_rows[e]))
    print(f"[entities] {len(ents)} with >= {args.min_idioms} idioms")

    prompts = []
    for e in ents:
        rows = ent_rows[e][: args.max_show]
        block = "\n".join(f"- {i} :: {t[:220]}" for i, t in rows)
        prompts.append(PROMPT.format(entity=e, idioms=block))

    resp = local_llm.generate(prompts, "ar_entity_symbolism", max_tokens=160)
    out = {}
    for e, r in zip(ents, resp):
        t = clean(r)
        # drop refusals / English leakage / over-long output
        if len(t) < 20 or len(t) > 600:
            continue
        if sum(ch.isascii() and ch.isalpha() for ch in t) > 0.3 * len(t):
            continue
        out[e] = {"symbolism": t, "n_idioms": len(ent_rows[e])}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, args.out)
    json.dump({"excluded_probe_entities": sorted(probe_ents), "entities": out},
              open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[write] {path}: {len(out)}/{len(ents)} kept")
    for e in ents[:3]:
        if e in out:
            print(f"  {e}: {out[e]['symbolism'][:180]}")


if __name__ == "__main__":
    main()
