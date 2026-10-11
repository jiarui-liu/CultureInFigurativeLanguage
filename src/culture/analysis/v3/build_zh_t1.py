#!/usr/bin/env python3
"""Build the three Chinese corpora for T1: is the cost of Culture-CPT caused by excluding
idiom-bearing documents?

§5.3 of the paper reports that at 9B, Chinese \\culturecpt{} *lowers* ChID, CCPM, CMMLU and the
symbolism probe, and attributes the loss to the design restriction that the culture corpus
contain no \\dataname{} idiom: 90.2% of the most culture-specific Chinese documents contain a
chengyu, so removing them leaves an unnatural slice of the language.  That is an argument, not
an experiment.  The experiment is to select culture-rich text the same way *without* the
restriction and train a matched arm on it.

Three token-matched corpora, all without meaning tags:
  zh_t1_random       documents sampled uniformly from the same web pool (the control)
  zh_t1_culture_free the restricted corpus: highest-scoring documents that contain no idiom
                     (the paper's \\culturecpt{} selection, rebuilt at the smaller budget)
  zh_t1_culture_all  the unrestricted corpus: highest-scoring documents of the union of the
                     idiom-free pool and the idiom-bearing pool, scored by the same classifier

`zh_t1_culture_all` is what a culture-selection pipeline would produce if it did not have to
keep the two conditions disjoint, so the contrast `culture_all - culture_free` isolates the
cost of the restriction from the effect of culture-rich text.

    PYTHONPATH=src python build_zh_t1.py --target_tokens 1.1e9
"""
from __future__ import annotations

import argparse
import glob
import gzip
import heapq
import json
import os
import pickle
import random

import numpy as np

DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")
RANKED = f"{DATA}/bidir/full/zh/culture_docs_ranked.jsonl.gz"   # idiom-free, score-sorted
IDIOM_DOCS = f"{DATA}/train_zh_untagged"                        # idiom-bearing, tags stripped
RANDOM_DOCS = f"{DATA}/train_zh_unfiltered"
CLF = f"{DATA}/bidir/clf/zh.pkl"
OUT = os.environ.get("CULTURE_T1_DIR", f"{DATA}/t1_zh")
EMB = os.environ.get("EMB_MODEL", "/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3-Embedding-0.6B")

CPT = 1.76          # characters per Qwen3.5 token, measured on the Chinese pool (build_arms.py)
SHARD = 20000       # documents per output shard


class Writer:
    def __init__(self, name):
        self.dir = os.path.join(OUT, name)
        os.makedirs(self.dir, exist_ok=True)
        self.i, self.n, self.chars, self.f = 0, 0, 0, None

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


def iter_jsonl(d, limit=None):
    files = sorted(glob.glob(f"{d}/*.jsonl")) + sorted(glob.glob(f"{d}/*.jsonl.gz"))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    n = 0
    for fp in files:
        op = gzip.open if fp.endswith(".gz") else open
        with op(fp, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                if len(t) < 200:
                    continue
                yield t
                n += 1
                if limit and n >= limit:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=float, default=1.1e9)
    ap.add_argument("--pool", type=int, default=1_200_000,
                    help="idiom-bearing documents to score with the culture classifier")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()
    target_chars = args.target_tokens * CPT
    os.makedirs(OUT, exist_ok=True)
    report = {"target_tokens": args.target_tokens, "pool": args.pool}

    # ---------------- 1. random control
    w = Writer("zh_t1_random")
    for t in iter_jsonl(RANDOM_DOCS):
        w.write(t)
        if w.chars >= target_chars:
            break
    report["zh_t1_random"] = w.close()
    print("[random]", report["zh_t1_random"])

    # ---------------- 2. restricted culture corpus (idiom-free, top of the ranked file)
    w = Writer("zh_t1_culture_free")
    free_cut = None
    with gzip.open(RANKED, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            t = o.get("text", "")
            if len(t) < 200:
                continue
            w.write(t)
            free_cut = float(o.get("score", 0))
            if w.chars >= target_chars:
                break
    report["zh_t1_culture_free"] = w.close()
    report["zh_t1_culture_free"]["score_cutoff"] = free_cut
    print("[culture_free]", report["zh_t1_culture_free"])

    # ---------------- 3. score the idiom-bearing pool with the same classifier
    from sentence_transformers import SentenceTransformer
    ridge = pickle.load(open(CLF, "rb"))["ridge"]
    m = SentenceTransformer(EMB, device="cuda", model_kwargs={"torch_dtype": "bfloat16"})
    m.max_seq_length = args.max_tokens

    # keep only as many idiom-bearing documents as could possibly be needed
    keep = int(target_chars / 1500) + 50000
    top = []            # min-heap of (score, counter, text)
    buf, cnt, scored = [], 0, 0
    for t in iter_jsonl(IDIOM_DOCS, limit=args.pool):
        buf.append(t)
        if len(buf) >= args.batch:
            E = m.encode([x[:6000] for x in buf], batch_size=args.batch,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False).astype(np.float32)
            for s, x in zip(ridge.predict(E), buf):
                cnt += 1
                if len(top) < keep:
                    heapq.heappush(top, (float(s), cnt, x))
                elif s > top[0][0]:
                    heapq.heapreplace(top, (float(s), cnt, x))
            scored += len(buf)
            buf = []
            if scored % 100000 < args.batch:
                print(f"[score] {scored} idiom-bearing docs, "
                      f"current cutoff {top[0][0]:.3f}" if top else "")
    if buf:
        E = m.encode([x[:6000] for x in buf], batch_size=args.batch,
                     normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float32)
        for s, x in zip(ridge.predict(E), buf):
            cnt += 1
            heapq.heappush(top, (float(s), cnt, x))
        scored += len(buf)
    report["scored_idiom_docs"] = scored
    idiom_scored = sorted(top, key=lambda z: -z[0])
    print(f"[score] done: {scored} docs, best {idiom_scored[0][0]:.3f}, "
          f"median kept {idiom_scored[len(idiom_scored) // 2][0]:.3f}")

    # ---------------- 4. unrestricted culture corpus: merge the two pools by score
    w = Writer("zh_t1_culture_all")
    n_idiom = n_free = 0
    gi = iter(idiom_scored)
    cur_i = next(gi, None)
    with gzip.open(RANKED, "rt", encoding="utf-8") as f:
        cur_f = None
        for line in f:
            o = json.loads(line)
            if len(o.get("text", "")) < 200:
                continue
            cur_f = (float(o.get("score", 0)), o["text"])
            while cur_i is not None and cur_i[0] >= cur_f[0]:
                w.write(cur_i[2])
                n_idiom += 1
                cur_i = next(gi, None)
                if w.chars >= target_chars:
                    break
            if w.chars >= target_chars:
                break
            w.write(cur_f[1])
            n_free += 1
            if w.chars >= target_chars:
                break
    while w.chars < target_chars and cur_i is not None:
        w.write(cur_i[2])
        n_idiom += 1
        cur_i = next(gi, None)
    report["zh_t1_culture_all"] = w.close()
    report["zh_t1_culture_all"].update(
        {"n_idiom_bearing": n_idiom, "n_idiom_free": n_free,
         "share_idiom_bearing": n_idiom / max(1, n_idiom + n_free)})
    print("[culture_all]", report["zh_t1_culture_all"])

    json.dump(report, open(os.path.join(OUT, "build_report.json"), "w"), indent=1)
    print("[write]", os.path.join(OUT, "build_report.json"))


if __name__ == "__main__":
    main()
