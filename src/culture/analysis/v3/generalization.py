#!/usr/bin/env python3
"""Does item-specific idiom learning spill over to *semantically neighbouring* idioms?

§5 establishes that meaning tags teach the idioms they list and not the idioms they do not,
and §5.6 shows the gain grows with how often a seen idiom occurs.  Both are statements about
the *listed* items.  The question they leave open is whether the knowledge is purely lexical
or whether it generalises along meaning space: an unseen proverb that means almost the same
thing as a seen one, or that is built from the same entities, should benefit if the model
learned anything more abstract than a lookup table.

For every unseen IdiomAtlas-MC item we compute
  * `nn_meaning`  cosine of its gold figurative meaning to the nearest *seen* idiom's meaning,
  * `nn_idiom`    cosine of the idiom string itself to the nearest seen idiom string,
  * `ent_overlap` how many of its entities occur in any seen idiom, and the max corpus
                  exposure of those shared entities,
and relate them to the per-item gain of each arm over Random-CPT.  A null here is itself a
result: it says the tags install a lookup table with no neighbourhood structure.

    PYTHONPATH=src:src/culture/analysis/v2 python generalization.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import common  # noqa: E402

EVAL = os.environ.get(
    "CULTURE_EVAL_RESULTS",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval",
)
DATA = os.environ.get(
    "CULTURE_DATA_DIR", "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data")
MC = f"{DATA}/eval/mc"
OUT_DIR = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")
EMB = os.environ.get("EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")

COUNT_FILES = {
    "zh": [f"{DATA}/fineweb-edu-zh-chengyu-cpt/stats/kept_idiom_counts_zh.json",
           f"{DATA}/mc4-zh-idiom-cpt/stats/kept_idiom_counts_zh.json"],
    "hi": [f"{DATA}/hi-proverbs-cpt/stats/kept_idiom_counts_hi.json"],
    "ar": [f"{DATA}/ar-amthal-cpt/stats/kept_idiom_counts_ar.json"],
}
ARMS = {"base": "base", "random": "unfiltered", "idiom_untagged": "untagged",
        "idiom_cpt": "cpt", "culture": "culture", "culture_notes": "culturenotes"}
LANGS = ("ar", "hi", "zh")


def _rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_p(a, b, reps=5000, seed=0):
    rho = spearman(a, b)
    if rho is None:
        return None, None
    rng = np.random.default_rng(seed)
    b = np.asarray(b, float)
    hits = sum(abs(spearman(a, rng.permutation(b))) >= abs(rho) for _ in range(reps))
    return rho, (hits + 1) / (reps + 1)


def load_counts(lang):
    c = defaultdict(int)
    for p in COUNT_FILES[lang]:
        if os.path.exists(p):
            for k, v in json.load(open(p, encoding="utf-8")).items():
                c[k] += int(v)
    return c


def load_records(lang, arm, task):
    p = f"{EVAL}/{lang}/{ARMS[arm]}/{task}.json"
    if not os.path.exists(p):
        return None
    return {r["qid"]: int(r.get("correct_norm", r.get("correct", 0)))
            for r in json.load(open(p, encoding="utf-8"))["records"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="generalization.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB, device="cuda", model_kwargs={"torch_dtype": "bfloat16"})
    m.max_seq_length = 128

    def enc(texts, bs=256):
        return m.encode(texts, batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    out = {}
    for lang in LANGS:
        counts = load_counts(lang)
        kb = common.load_kb(lang)
        by_idiom = {r["idiom"]: r for r in kb}
        seen_rows = [r for r in kb if counts.get(r["idiom"], 0) > 0 and r["fig"]]
        if not seen_rows:
            print(f"[skip] {lang}: no seen idioms")
            continue
        seen_mean = [common.flatten(r["fig"])[0] for r in seen_rows]
        seen_str = [r["idiom"] for r in seen_rows]
        seen_ents = set()
        for r in seen_rows:
            seen_ents |= {common.norm_entity(e, lang) for e in (r.get("entities") or [])}
        print(f"[{lang}] seen idioms {len(seen_rows)}, seen entities {len(seen_ents)}")

        E_mean, E_str = enc(seen_mean), enc(seen_str)

        rows = [json.loads(l) for l in
                open(f"{MC}/idiomatlas_mc_{lang}_unseen.jsonl", encoding="utf-8")]
        q_mean, q_str, feats = [], [], []
        for o in rows:
            idiom = (o.get("meta") or {}).get("idiom", "")
            gold = o["options"][o["gold"]].strip()
            kbrow = by_idiom.get(idiom, {})
            ents = {common.norm_entity(e, lang) for e in (kbrow.get("entities") or [])}
            q_mean.append(gold)
            q_str.append(idiom)
            feats.append({
                "qid": o["qid"], "idiom": idiom,
                "n_ent": len(ents),
                "ent_overlap": len(ents & seen_ents),
                "ent_overlap_frac": (len(ents & seen_ents) / len(ents)) if ents else 0.0,
                "len_chars": len(idiom),
                "gold_len": len(gold),
                "max_distractor_sim": max((o.get("meta") or {}).get("distractor_sims") or [0]),
            })
        Qm, Qs = enc(q_mean), enc(q_str)
        nn_mean = (Qm @ E_mean.T).max(axis=1)
        nn_str = (Qs @ E_str.T).max(axis=1)
        for f, a, b in zip(feats, nn_mean, nn_str):
            f["nn_meaning"] = float(a)
            f["nn_idiom"] = float(b)

        base = load_records(lang, "random", f"idiomatlas_mc_{lang}_unseen")
        res = {"n_unseen": len(feats), "n_seen_idioms": len(seen_rows), "arms": {}}
        for arm in ("idiom_cpt", "idiom_untagged", "culture", "culture_notes", "base"):
            rec = load_records(lang, arm, f"idiomatlas_mc_{lang}_unseen")
            if rec is None or base is None:
                continue
            qids = [f["qid"] for f in feats if f["qid"] in rec and f["qid"] in base]
            gain = np.array([rec[q] - base[q] for q in qids], float)
            fm = {q: f for f in feats for q in [f["qid"]]}
            r = {"n": len(qids), "mean_gain": float(gain.mean())}
            for key in ("nn_meaning", "nn_idiom", "ent_overlap", "ent_overlap_frac",
                        "n_ent", "len_chars", "max_distractor_sim"):
                x = np.array([fm[q][key] for q in qids], float)
                rho, p = perm_p(x, gain)
                r[key] = {"rho": rho, "p": p}
                # top vs bottom tercile of the predictor
                if x.std() > 0:
                    lo, hi = np.percentile(x, [33.3, 66.7])
                    r[key]["tercile_gain"] = [
                        float(gain[x <= lo].mean()) if (x <= lo).any() else None,
                        float(gain[(x > lo) & (x <= hi)].mean()) if ((x > lo) & (x <= hi)).any() else None,
                        float(gain[x > hi].mean()) if (x > hi).any() else None,
                    ]
            res["arms"][arm] = r
            print(f"  [{lang}/{arm}] gain={r['mean_gain']:+.3f} "
                  f"nn_meaning rho={r['nn_meaning']['rho']} p={r['nn_meaning']['p']}")
        out[lang] = res
        out[f"{lang}_items"] = feats

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, args.out)
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"[write] {p}")


if __name__ == "__main__":
    main()
