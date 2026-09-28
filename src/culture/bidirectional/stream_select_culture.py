#!/usr/bin/env python3
"""Build a full-scale (9B-study) Culture corpus by streaming raw web files.

For every raw file (downloaded one at a time from the Hub, processed, then deleted,
so disk use stays small), documents pass the same gates as the pool
(``build_pool``), documents containing any IdiomAtlas idiom are dropped, and the
remaining documents are scored with the culture classifier. Documents scoring at
least ``--min_score`` are kept with their score. ``finalize`` then ranks all kept
documents by score, cuts the ranking at the token budget of the corresponding 9B
Idiom-CPT corpus, shuffles, and writes LLaMA-Factory ``{"text": ...}`` shards.

Usage:
  # score (parallelise with --part/--nparts; one GPU per process)
  python -m culture.bidirectional.stream_select_culture score --lang zh \
      --repo opencsg/Fineweb-Edu-Chinese-V2.1 --pattern '4_5/*.parquet' \
      --clf $B/clf/zh.pkl --min_score 2.0 --out_dir $B/full/zh --part 0 --nparts 4
  # or score an existing local pool
  python -m culture.bidirectional.stream_select_culture score --lang hi --pool_dir $B/pool/hi ...
  python -m culture.bidirectional.stream_select_culture finalize --lang zh --out_dir $B/full/zh \
      --budget_tokens 7800000000 --chars_per_token 1.76
"""
import argparse
import fnmatch
import glob
import gzip
import json
import os
import pickle
import random
import shutil

import numpy as np

_G = {}


def _gate_init(lang):
    from culture.bidirectional.build_pool import make_matcher
    _G["m"] = make_matcher(lang)
    if lang == "ar":
        from culture.training.mC4.quality_ar import reject_reason
        _G["rr"], _G["lo"], _G["hi"] = reject_reason, 300, 25_000
    else:
        _G["rr"], _G["lo"], _G["hi"] = None, 150, 100_000


def _gate(texts):
    """Keep idiom-free documents that pass the pool's length/quality gates."""
    out = []
    for t in texts:
        if not t or not (_G["lo"] <= len(t) <= _G["hi"]):
            continue
        if _G["rr"] is not None and _G["rr"](t):
            continue
        if _G["m"].match(t):
            continue
        out.append(t)
    return out


def _chunks(it, n=5000):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def cmd_score(a):
    from culture.bidirectional.build_pool import iter_raw
    from culture.bidirectional.culture_classifier import embed, embedder
    clf = pickle.load(open(a.clf, "rb"))
    m = embedder(clf["max_tokens"])
    os.makedirs(a.out_dir, exist_ok=True)

    if a.pool_dir:
        units = sorted(glob.glob(os.path.join(a.pool_dir, "*.jsonl.gz")))
    else:
        from huggingface_hub import HfApi
        files = [f.path for f in HfApi().list_repo_tree(a.repo, repo_type="dataset", recursive=True,
                                                        path_in_repo=os.path.dirname(a.pattern) or None)
                 if hasattr(f, "size") and fnmatch.fnmatch(f.path, a.pattern)]
        units = sorted(files)
    # optionally split each parquet file into row-group slices (unit = (file, slice))
    units = [(u, j) for u in units for j in range(a.rg_split)]
    units = units[a.part::a.nparts]
    for u, j in units:
        tag = os.path.basename(u).replace(".jsonl.gz", "").replace(".parquet", "").replace(".json.gz", "")
        tag = f"{a.source_tag}_{tag}" if a.source_tag else tag
        if a.rg_split > 1:
            tag = f"{tag}_rg{j}of{a.rg_split}"
        out = os.path.join(a.out_dir, f"kept_{tag}.jsonl.gz")
        if os.path.exists(out):
            continue
        local, tmpdir = u, None
        if not a.pool_dir:
            from huggingface_hub import hf_hub_download
            tmpdir = os.path.join(a.tmp_dir, f"dl_{a.part}")
            local = hf_hub_download(a.repo, u, repo_type="dataset", local_dir=tmpdir)
        texts = []
        if a.pool_dir:
            with gzip.open(local, "rt", encoding="utf-8") as fh:
                for l in fh:
                    d = json.loads(l)
                    if not d["idioms"]:
                        texts.append(d["text"])
        else:
            from multiprocessing import Pool
            with Pool(a.workers, initializer=_gate_init, initargs=(a.lang,)) as pool:
                for kept in pool.imap(_gate, _chunks(iter_raw(local, a.rg_split, j)), chunksize=1):
                    texts.extend(kept)
        n_in = len(texts)
        order = np.argsort([len(t) for t in texts])
        sc = np.zeros(len(texts), np.float32)
        for s in range(0, len(order), 4096):
            sel = order[s:s + 4096]
            E = embed(m, [texts[i][:4 * clf["max_tokens"]] for i in sel], bs=128)
            sc[sel] = clf["ridge"].predict(E)
        tmp = out + ".tmp"
        kept = 0
        with gzip.open(tmp, "wt", encoding="utf-8") as fo:
            for t, s in zip(texts, sc):
                if s >= a.min_score:
                    fo.write(json.dumps({"text": t, "score": round(float(s), 4)}, ensure_ascii=False) + "\n")
                    kept += 1
        os.replace(tmp, out)
        json.dump({"unit": u, "slice": j, "idiom_free_docs": n_in, "kept": kept,
                   "score_hist": np.histogram(sc, bins=[-10, 0, 1, 2, 3, 4, 10])[0].tolist()},
                  open(out.replace(".jsonl.gz", ".stats.json"), "w"))
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        print("unit", u, n_in, kept, flush=True)


def cmd_finalize(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out_dir, "kept_*.jsonl.gz"))):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for i, l in enumerate(fh):
                d = json.loads(l)
                rows.append((d["score"], f, i, len(d["text"])))
    rows.sort(key=lambda r: -r[0])
    need = a.budget_tokens * a.chars_per_token
    tot, keep = 0, set()
    for s, f, i, n in rows:
        keep.add((f, i))
        tot += n
        if tot >= need:
            cutoff = s
            break
    else:
        cutoff = rows[-1][0] if rows else None
    docs, ranked = [], []
    for f in sorted({f for f, _ in keep}):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for i, l in enumerate(fh):
                if (f, i) in keep:
                    d = json.loads(l)
                    docs.append(d["text"])
                    ranked.append({"id": f"{os.path.basename(f)}:{i}", "text": d["text"], "score": d["score"]})
    ranked.sort(key=lambda d: -d["score"])
    with gzip.open(os.path.join(a.out_dir, "culture_docs_ranked.jsonl.gz"), "wt", encoding="utf-8") as fo:
        for d in ranked:  # input for culture_notes.py (Culture+notes corpus)
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    random.Random(0).shuffle(docs)
    od = os.path.join(a.out_dir, "train")
    os.makedirs(od, exist_ok=True)
    per = a.docs_per_shard
    for k in range(0, len(docs), per):
        with open(os.path.join(od, f"train_{k // per:05d}.jsonl"), "w", encoding="utf-8") as fo:
            for t in docs[k:k + per]:
                fo.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    st = {"candidates": len(rows), "selected": len(docs), "chars": tot,
          "est_tokens": tot / a.chars_per_token, "score_cutoff": cutoff}
    json.dump(st, open(os.path.join(a.out_dir, "finalize_stats.json"), "w"), indent=1)
    print(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "finalize"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--repo")
    ap.add_argument("--pattern")
    ap.add_argument("--pool_dir")
    ap.add_argument("--source_tag", default="")
    ap.add_argument("--clf")
    ap.add_argument("--min_score", type=float, default=2.0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tmp_dir", default="/scratch/jiaruil5/stream_tmp")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--budget_tokens", type=float, default=0)
    ap.add_argument("--chars_per_token", type=float, default=1.0)
    ap.add_argument("--docs_per_shard", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rg_split", type=int, default=1)
    a = ap.parse_args()
    cmd_score(a) if a.cmd == "score" else cmd_finalize(a)


if __name__ == "__main__":
    main()
