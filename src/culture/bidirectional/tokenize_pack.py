#!/usr/bin/env python3
"""Tokenize a jsonl(.gz) corpus and pack it into fixed-length blocks.

Each document is tokenized, an EOS token is appended, all documents are
concatenated in file order and cut into ``seq_len`` blocks (the trailing partial
block is dropped). Output is a flat uint32 ``.bin`` memmap plus a ``.json``
manifest with the number of blocks, tokens, and documents.

``--max_tokens`` stops once that many tokens have been packed, which is how the
token budget of every training arm is fixed exactly.

Usage:
  python -m culture.bidirectional.tokenize_pack --inputs a.jsonl.gz b.jsonl.gz \
      --tokenizer <model> --out <prefix> --seq_len 4096 --max_tokens 400000000
"""
import argparse
import gzip
import json
import os
from multiprocessing import Pool

import numpy as np
from transformers import AutoTokenizer

_TOK = None


def _init(path):
    global _TOK
    _TOK = AutoTokenizer.from_pretrained(path)


def _encode(texts):
    ids = _TOK(texts, add_special_tokens=False)["input_ids"]
    eos = _TOK.eos_token_id
    return [x + [eos] for x in ids]


def iter_texts(paths, field):
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line).get(field)
                if t:
                    yield t


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, help="output prefix (writes .bin and .json)")
    ap.add_argument("--field", default="text")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".bin.tmp"
    n_tok = n_doc = 0
    with open(tmp, "wb") as fo, Pool(a.workers, initializer=_init, initargs=(a.tokenizer,)) as pool:
        for docs in pool.imap(_encode, batched(iter_texts(a.inputs, a.field), 256), chunksize=1):
            stop = False
            for ids in docs:
                if a.max_tokens and n_tok + len(ids) > a.max_tokens:
                    ids = ids[: a.max_tokens - n_tok]
                    stop = True
                np.asarray(ids, dtype=np.uint32).tofile(fo)
                n_tok += len(ids)
                n_doc += 1
                if stop:
                    break
            if stop:
                pool.terminate()
                break
    n_blocks = n_tok // a.seq_len
    with open(tmp, "r+b") as fo:
        fo.truncate(n_blocks * a.seq_len * 4)
    os.replace(tmp, a.out + ".bin")
    man = {"inputs": a.inputs, "tokenizer": a.tokenizer, "seq_len": a.seq_len,
           "n_blocks": n_blocks, "n_tokens": n_blocks * a.seq_len, "n_docs": n_doc}
    json.dump(man, open(a.out + ".json", "w"), indent=2)
    print(json.dumps(man))


if __name__ == "__main__":
    main()
