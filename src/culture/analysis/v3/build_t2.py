#!/usr/bin/env python3
"""T2, generalised: token-matched corpora that differ only in what is appended.

`build_ar_t2.py` wrote the Arabic triple (untagged / dict / sym) with the symbolism-probe
entities held out of the statements.  This script does the same for either language and
lets the sym arm be written from any summary file, which is what the two follow-ups of the
pass-3 log need:

  hi triple       --lang hi --arms untagged,dict,sym
  ar no-holdout   --lang ar --arms sym --summaries ar_entity_symbolism_all.json
                  --sym_name ar_t2_symall --max_docs 410669

``--max_docs`` is how the no-holdout arm is kept comparable to the existing triple: the
corpora are written from one pass over the source shards in the same order, so stopping at
the same document count covers exactly the same documents.  Its token count is then slightly
higher than `ar_t2_sym`'s, because more documents carry a tag; the training budget is fixed in
steps, so both arms still read the same number of tokens.

    PYTHONPATH=src:src/culture/analysis/v2 python build_t2.py --lang hi
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

# per language: source shards, the meaning-tag header that marks the end of the body, the
# header introducing the symbolism block, and measured characters per Qwen3.5 token.
LANG = {
    "ar": dict(src=f"{DATA}/train_ar",
               dict_header="\n\nالمعاني الاصطلاحية للتعابير الواردة في النص:",
               sym_header="\n\nما ترمز إليه عناصر الأمثال الواردة في النص:",
               cpt=3.14),
    "hi": dict(src=f"{DATA}/train",
               dict_header="\n\nलोकोक्तियों के अर्थ:",
               sym_header="\n\nलोकोक्तियों में आए तत्त्व किसके प्रतीक हैं:",
               cpt=1.56),
}
SHARD = 20000
MAX_ENTITIES = 4


class Writer:
    def __init__(self, out, name):
        self.dir = os.path.join(out, name)
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

    def close(self, cpt):
        if self.f:
            self.f.close()
        return {"docs": self.n, "chars": self.chars, "est_tokens": int(self.chars / cpt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ar", "hi"], required=True)
    ap.add_argument("--arms", default="untagged,dict,sym")
    ap.add_argument("--target_tokens", type=float, default=1.05e9)
    ap.add_argument("--max_docs", type=int, default=0, help="0 = no document cap")
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--sym_name", default=None, help="corpus dir for the sym arm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    lang = args.lang
    cfg = LANG[lang]
    out = args.out or os.environ.get("CULTURE_T2_DIR", f"{DATA}/t2_{lang}")
    summaries = args.summaries or os.path.join(out, f"{lang}_entity_symbolism.json")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    names = {"untagged": f"{lang}_t2_untagged", "dict": f"{lang}_t2_dict",
             "sym": args.sym_name or f"{lang}_t2_sym"}

    sym = {}
    if "sym" in arms:
        sym = json.load(open(summaries, encoding="utf-8"))["entities"]
        print(f"[sym] {len(sym)} entities with a symbolism statement ({summaries})")
    kb = common.load_kb(lang)
    by_idiom = {r["idiom"]: r for r in kb}

    w = {a: Writer(out, names[a]) for a in arms}
    # the symbolism corpus is the longest of the three, so it is the one the token budget is
    # measured on; with --max_docs the document count decides instead.
    budget_arm = "sym" if "sym" in arms else arms[0]
    target_chars = args.target_tokens * cfg["cpt"]
    tagged = 0

    def done():
        if args.max_docs:
            return w[budget_arm].n >= args.max_docs
        return w[budget_arm].chars >= target_chars

    for fp in sorted(glob.glob(f"{cfg['src']}/*.jsonl")):
        if done():
            break
        for line in open(fp, encoding="utf-8"):
            if done():
                break
            try:
                o = json.loads(line)
            except Exception:
                continue
            full = o.get("text", "")
            body = full.split(cfg["dict_header"])[0]
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
            if "untagged" in w:
                w["untagged"].write(body)
            if "dict" in w:
                w["dict"].write(full)
            if "sym" in w:
                if ents:
                    tagged += 1
                    block = "\n".join(
                        f"- {e}: {sym[e]['symbolism']}" for e in ents[:MAX_ENTITIES])
                    w["sym"].write(body + cfg["sym_header"] + "\n" + block)
                else:
                    w["sym"].write(body)

    report = {names[a]: w[a].close(cfg["cpt"]) for a in arms}
    report.update({"lang": lang, "docs_with_symbolism_tag": tagged,
                   "max_entities": MAX_ENTITIES, "summaries": summaries,
                   "max_docs": args.max_docs})
    os.makedirs(out, exist_ok=True)
    rp = os.path.join(out, args.report or "build_report.json")
    json.dump(report, open(rp, "w"), indent=1)
    print("[t2]", json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
