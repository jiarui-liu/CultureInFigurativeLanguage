#!/usr/bin/env python3
"""T2: three token-matched Arabic corpora that differ only in what is appended to the document.

The paper's meaning tags state what an expression means; \\S5.4 shows that what idioms carry,
and what the culture corpus lacks, is instead a *symbolic and evaluative* layer, and the paper
ends by naming ``a notes generator aimed at evaluative associations rather than definitions''
as the next step. This builds that condition and its controls:

  ar_t2_untagged  the idiom-bearing documents, nothing appended
  ar_t2_dict      the same documents, same order, with the existing meaning tag
  ar_t2_sym       the same documents, same order, with an entity-symbolism tag built from
                  `ar_symbolism_summaries.py`

Because the three corpora are written from one pass over the same documents and truncated at
the same token budget, the only difference between the arms is the appended block. Entities
used by the Arabic symbolism probe are excluded from the summaries, so the probe remains a test
of generalisation rather than of recall.

    PYTHONPATH=src:src/culture/analysis/v2 python build_ar_t2.py
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import common  # noqa: E402

DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")
SRC = f"{DATA}/train_ar"
OUT = os.environ.get("CULTURE_T2_DIR", f"{DATA}/t2_ar")
DICT_HEADER = "\n\nالمعاني الاصطلاحية للتعابير الواردة في النص:"
SYM_HEADER = "\n\nما ترمز إليه عناصر الأمثال الواردة في النص:"
CPT = 3.14
SHARD = 20000
MAX_ENTITIES = 4


class Writer:
    def __init__(self, name):
        self.dir = os.path.join(OUT, name)
        os.makedirs(self.dir, exist_ok=True)
        self.i = self.n = self.chars = 0
        self.f = None

    def write(self, text):
        if self.f is None or self.n % SHARD == 0:
            if self.f:
                self.f.close()
            self.f = open(os.path.join(self.dir, f"train_{self.i:05d}.jsonl"), "w",
                          encoding="utf-8")
            self.i += 1
        self.f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.n += 1
        self.chars += len(text)

    def close(self):
        if self.f:
            self.f.close()
        return {"docs": self.n, "chars": self.chars, "est_tokens": int(self.chars / CPT)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=float, default=1.05e9)
    ap.add_argument("--summaries", default=os.path.join(OUT, "ar_entity_symbolism.json"))
    args = ap.parse_args()

    sym = json.load(open(args.summaries, encoding="utf-8"))["entities"]
    print(f"[sym] {len(sym)} entities with a symbolism statement")
    kb = common.load_kb("ar")
    by_idiom = {r["idiom"]: r for r in kb}

    w_un, w_dict, w_sym = Writer("ar_t2_untagged"), Writer("ar_t2_dict"), Writer("ar_t2_sym")
    # the symbolism corpus is the budget-limiting one: it is the longest of the three, so
    # stop when *it* reaches the target and let the other two be shorter in characters but
    # identical in documents.
    tagged = 0
    target_chars = args.target_tokens * CPT
    for fp in sorted(glob.glob(f"{SRC}/*.jsonl")):
        if w_sym.chars >= target_chars:
            break
        for line in open(fp, encoding="utf-8"):
            if w_sym.chars >= target_chars:
                break
            try:
                o = json.loads(line)
            except Exception:
                continue
            full = o.get("text", "")
            body = full.split(DICT_HEADER)[0]
            if len(body) < 200:
                continue
            m = o.get("matched_idioms", [])
            if isinstance(m, str):
                try:
                    m = ast.literal_eval(m)
                except Exception:
                    m = []
            ents, seen = [], set()
            for idiom in m:
                for e in (by_idiom.get(idiom, {}).get("entities") or []):
                    if e not in seen and e in sym:
                        seen.add(e)
                        ents.append(e)
            if ents:
                tagged += 1
                block = "\n".join(f"- {e}: {sym[e]['symbolism']}" for e in ents[:MAX_ENTITIES])
                sym_text = body + SYM_HEADER + "\n" + block
            else:
                sym_text = body
            w_un.write(body)
            w_dict.write(full)
            w_sym.write(sym_text)

    report = {"ar_t2_untagged": w_un.close(), "ar_t2_dict": w_dict.close(),
              "ar_t2_sym": w_sym.close(), "docs_with_symbolism_tag": tagged,
              "max_entities": MAX_ENTITIES}
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "build_report.json"), "w"), indent=1)
    print("[t2]", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
