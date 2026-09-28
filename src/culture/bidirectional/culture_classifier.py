#!/usr/bin/env python3
"""Culture-specificity classifier (FineWeb-Edu style) and pool scoring.

train:  embed the LLM-annotated documents with Qwen3-Embedding-0.6B (first
        --max_tokens tokens, mean-pooled by sentence-transformers), fit a ridge
        regression to the 0-5 scores on 80% and report held-out Spearman rho and
        precision/recall/F1 at score >= 3 on the other 20%; then refit on all data.
score:  embed every pool document of a shard range and write
        <out_dir>/<shard>.scores.jsonl  {"id", "score", "n_idioms", "n_chars"}

Usage:
  python -m culture.bidirectional.culture_classifier train --annot $B/annot/ar.jsonl --model_out $B/clf/ar.pkl
  python -m culture.bidirectional.culture_classifier score --model_in $B/clf/ar.pkl \
      --pool_dir $B/pool/ar --out_dir $B/scores/ar --part 0 --nparts 4
"""
import argparse
import glob
import gzip
import json
import os
import pickle

import numpy as np

EMB = os.environ.get("EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")  # fetched from the Hub into $HF_HOME


def embedder(max_tokens):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB, device="cuda", model_kwargs={"torch_dtype": "bfloat16"})
    m.max_seq_length = max_tokens
    return m


def embed(m, texts, bs=64):
    return m.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False).astype(np.float32)


def cmd_train(a):
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.metrics import precision_recall_fscore_support
    rows = [json.loads(l) for l in open(a.annot, encoding="utf-8")]
    rows = [r for r in rows if r.get("score") is not None]
    y = np.array([r["score"] for r in rows], float)
    m = embedder(a.max_tokens)
    X = embed(m, [r["text"] for r in rows])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    best = None
    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0]:
        reg = Ridge(alpha=alpha).fit(X[tr], y[tr])
        p = reg.predict(X[te])
        rho = spearmanr(p, y[te]).correlation
        pr, rc, f1, _ = precision_recall_fscore_support(y[te] >= 3, p >= 3, average="binary", zero_division=0)
        res = {"alpha": alpha, "spearman": float(rho), "precision@3": float(pr), "recall@3": float(rc),
               "f1@3": float(f1), "n_train": int(len(tr)), "n_test": int(len(te)),
               "label_dist": {int(k): int((y == k).sum()) for k in range(6)}}
        if best is None or res["f1@3"] > best["f1@3"]:
            best = res
    reg = Ridge(alpha=best["alpha"]).fit(X, y)
    os.makedirs(os.path.dirname(a.model_out), exist_ok=True)
    pickle.dump({"ridge": reg, "max_tokens": a.max_tokens, "eval": best}, open(a.model_out, "wb"))
    json.dump(best, open(a.model_out.replace(".pkl", ".eval.json"), "w"), indent=1)
    print(json.dumps(best))


def cmd_score(a):
    clf = pickle.load(open(a.model_in, "rb"))
    m = embedder(clf["max_tokens"])
    files = sorted(glob.glob(os.path.join(a.pool_dir, "*.jsonl.gz")))[a.part::a.nparts]
    os.makedirs(a.out_dir, exist_ok=True)
    for f in files:
        shard = os.path.basename(f)[:-len(".jsonl.gz")]
        out = os.path.join(a.out_dir, f"{shard}.scores.jsonl")
        if os.path.exists(out):
            continue
        ids, texts, nid, nch = [], [], [], []
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                ids.append(d["id"])
                texts.append(d["text"][:4 * clf["max_tokens"]])
                nid.append(len(d["idioms"]))
                nch.append(len(d["text"]))
        # sort by length for efficient batching, then restore order
        order = np.argsort([len(t) for t in texts])
        E = np.zeros((len(texts), 1024), np.float32)
        for s in range(0, len(order), 4096):
            sel = order[s:s + 4096]
            E[sel] = embed(m, [texts[i] for i in sel], bs=128)
        sc = clf["ridge"].predict(E)
        tmp = out + ".tmp"
        with open(tmp, "w") as fo:
            for i in range(len(ids)):
                fo.write(json.dumps({"id": ids[i], "score": round(float(sc[i]), 4),
                                     "n_idioms": nid[i], "n_chars": nch[i]}) + "\n")
        os.replace(tmp, out)
        print("scored", shard, len(ids), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--annot")
    ap.add_argument("--model_out")
    ap.add_argument("--model_in")
    ap.add_argument("--pool_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=512)
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_score(a)


if __name__ == "__main__":
    main()
