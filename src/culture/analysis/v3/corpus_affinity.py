#!/usr/bin/env python3
"""Two corpus-side analyses the paper currently argues for rather than measures.

A3  Culture-score profile of every training corpus.  The paper selects culture-rich text
    with a ridge classifier on Qwen3-Embedding features, and reports in one direction only
    how often culture-rich documents contain an idiom.  The reverse question decides whether
    "Idiom-CPT" is secretly a culture-CPT: how culture-rich is the idiom corpus, on the
    paper's own classifier?  We score a sample of every arm's corpus with the arm-matched
    classifier and report the distribution.

A2  Corpus -> benchmark affinity.  §5 explains the one reliable culture gain by saying
    ArabCulture is "closest to the content of the culture corpus", and never measures
    closeness.  We embed every benchmark item and a sample of each corpus, score each
    benchmark by its affinity to each corpus, and leave the regression of the observed
    per-benchmark deltas on the affinity contrast to `affinity_vs_transfer.py` (CPU).

    PYTHONPATH=src:src/culture/analysis/v2 python corpus_affinity.py
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DATA = os.environ.get(
    "CULTURE_DATA_ROOT",
    "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data",
)
ITEMS = os.environ.get("CULTURE_ITEMS_DIR", f"{DATA}/bidir/items")
CLF = os.environ.get("CULTURE_CLF_DIR", f"{DATA}/bidir/clf")
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3",
)
EMB = os.environ.get("EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# arm -> corpus directory per language.  `unfiltered` = Random-CPT, the reference arm.
# The Hindi tagged corpus is the unsuffixed `train/`; zh and ar follow `train_<lang>`.
CORPORA = {
    "random": {"ar": "train_ar_unfiltered", "hi": "train_hi_unfiltered",
               "zh": "train_zh_unfiltered"},
    "idiom": {"ar": "train_ar", "hi": "train", "zh": "train_zh"},
    "idiom_untagged": {"ar": "train_ar_untagged", "hi": "train_hi_untagged",
                       "zh": "train_zh_untagged"},
    "culture": {"ar": "train_ar_culture", "hi": "train_hi_culture",
                "zh": "train_zh_culture"},
    "culturenotes": {"ar": "train_ar_culturenotes"},
}
LANGS = ["ar", "hi", "zh"]

TASKS_BY_LANG = {
    "ar": ["arabculture", "alyah", "dzirieval", "arabic_cultural_qa", "global_piqa_ar",
           "arabicmmlu", "kinayat_meaning", "ar_figurative",
           "idiomatlas_mc_ar_seen", "idiomatlas_mc_ar_unseen", "symbolism_v2_ar_letter"],
    "hi": ["milu", "global_piqa", "mabl", "global_piqa_hi", "global_piqa_hi_cultural",
           "parambench_hi_culture", "parambench_hi_other",
           "idiomatlas_mc_hi_seen", "idiomatlas_mc_hi_unseen", "symbolism_v2_hi_letter"],
    "zh": ["ccpm", "cmmlu", "chengyu_bench", "global_piqa_zh", "global_piqa_zh_cultural",
           "idiomatlas_mc_zh_seen", "idiomatlas_mc_zh_unseen", "symbolism_v2_zh_letter"],
}

TAG_HEADERS = [
    "\n\n【成语注释】",                                      # zh meaning tag
    "\n\nالمعاني الاصطلاحية للتعابير الواردة في النص:",       # ar meaning tag
    "\n\nلوكوكتيات",                                        # (unused; defensive)
    "\n\nलोकोक्तियों के अर्थ:",                              # hi meaning tag
    "\n\nملاحظات ثقافية حول النص:",                          # ar cultural notes
]


def _iter_docs(d, limit, rng_seed=0):
    """Read up to `limit` documents, reservoir-sampled over the first 40 shards."""
    files = sorted(glob.glob(f"{d}/*.jsonl")) + sorted(glob.glob(f"{d}/*.jsonl.gz"))
    files = [f for f in files if not os.path.basename(f).startswith("_")][:40]
    rng = random.Random(rng_seed)
    res, seen = [], 0
    for fp in files:
        op = gzip.open if fp.endswith(".gz") else open
        try:
            with op(fp, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        t = json.loads(line).get("text", "")
                    except Exception:
                        continue
                    if not t or len(t) < 200:
                        continue
                    seen += 1
                    if len(res) < limit:
                        res.append(t)
                    else:
                        j = rng.randrange(seen)
                        if j < limit:
                            res[j] = t
        except Exception as e:
            print(f"[warn] {fp}: {e}")
        if seen > limit * 40:
            break
    return res


def strip_tag(text):
    """Remove the appended meaning-tag / cultural-notes block.

    The classifier was trained on untagged web text, so leaving the block in would score
    the tag rather than the document.  The headers are the ones `build_arms.py` and
    `culture_notes.py` write.
    """
    for h in TAG_HEADERS:
        i = text.find(h)
        if i > 0:
            return text[:i]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_docs", type=int, default=6000)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default="corpus_affinity.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB, device="cuda", model_kwargs={"torch_dtype": "bfloat16"})
    m.max_seq_length = args.max_tokens

    def enc(texts, bs=64):
        return m.encode(texts, batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    out = {"n_docs": args.n_docs, "culture_score": {}, "affinity": {}, "meta": {}}

    for lang in LANGS:
        clf_p = f"{CLF}/{lang}.pkl"
        ridge = pickle.load(open(clf_p, "rb"))["ridge"] if os.path.exists(clf_p) else None
        if ridge is None:
            print(f"[warn] no classifier for {lang}")

        corpus_emb = {}
        for arm, pats in CORPORA.items():
            if lang not in pats:
                continue
            d = f"{DATA}/{pats[lang]}"
            if not os.path.isdir(d):
                print(f"[skip] {lang}/{arm}: {d}")
                continue
            docs = _iter_docs(d, args.n_docs)
            if not docs:
                print(f"[skip] {lang}/{arm}: empty")
                continue
            if arm in ("idiom", "culturenotes"):
                docs = [strip_tag(t) for t in docs]
            E = enc([t[:6000] for t in docs])
            corpus_emb[arm] = E
            if ridge is not None:
                s = ridge.predict(E)
                out["culture_score"][f"{lang}/{arm}"] = {
                    "n": len(s), "mean": float(s.mean()), "sd": float(s.std()),
                    "p25": float(np.percentile(s, 25)), "median": float(np.median(s)),
                    "p75": float(np.percentile(s, 75)),
                    "share_ge3": float((s >= 3).mean()), "share_ge4": float((s >= 4).mean()),
                    "hist": np.histogram(s, bins=np.arange(-1, 6.5, 0.5))[0].tolist(),
                }
            print(f"[score] {lang}/{arm}: n={len(docs)} "
                  f"mean={out['culture_score'].get(f'{lang}/{arm}', {}).get('mean')}")

        # ---- benchmark affinity
        for task in TASKS_BY_LANG[lang]:
            p = f"{ITEMS}/{task}.jsonl"
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
            if len(rows) > 2000:
                rows = random.Random(0).sample(rows, 2000)
            texts = []
            for o in rows:
                g = o.get("gold")
                opts = o.get("options") or []
                gold = opts[g] if isinstance(g, int) and 0 <= g < len(opts) else ""
                texts.append((str(o["stem"]) + " " + str(gold)).strip()[:2000])
            Q = enc(texts)
            for arm, E in corpus_emb.items():
                S = Q @ E.T                                    # (n_items, n_docs)
                topk = np.sort(S, axis=1)[:, -args.topk:]
                out["affinity"].setdefault(task, {})[arm] = {
                    "mean_top{}".format(args.topk): float(topk.mean()),
                    "mean_max": float(S.max(axis=1).mean()),
                    "mean_centroid": float((Q @ E.mean(0)).mean()),
                    "n_items": len(rows),
                }
            out["meta"][task] = {"lang": lang, "n": len(rows)}
            print(f"[aff] {task}: " + ", ".join(
                f"{a}={out['affinity'][task][a]['mean_top%d' % args.topk]:.3f}"
                for a in corpus_emb))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(path, "w"), indent=1)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
