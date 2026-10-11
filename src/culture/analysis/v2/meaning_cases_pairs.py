#!/usr/bin/env python3
"""Table 3 (same meaning, different entity) for the five pairs beyond en-zh.

The paper's Table~\\ref{tab:meaning-cases} covers English-Chinese only. This builds the
same table for en-hi, en-ar, zh-hi, zh-ar and hi-ar.

Method, following the en-zh original: embed every idiom's first figurative meaning with
Qwen3-Embedding, pair idioms across the two languages whose meanings reach cosine 0.70,
and keep the pairs where the two sides reach that meaning through *different* entities --
which is the phenomenon the table exists to show. Candidate clusters are then ranked by
how asymmetric the two sides are (one language elaborating a meaning the other states
plainly is the asymmetric-lexicalisation point), and an LLM writes the shared-meaning
label and the literal glosses from the idioms alone.

pdfLaTeX with CJKutf8 can set Chinese via \\zh{} but not Devanagari or Arabic, so Hindi
and Arabic idioms are romanised with a literal gloss.

    sbatch meaning_cases_remote.slurm      # needs the embedding model on GPU
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

import common

PAIRS = [("en", "hi"), ("en", "ar"), ("zh", "hi"), ("zh", "ar"), ("hi", "ar")]
LANG_NAME = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
SIM = 0.70          # the paper's pairing threshold
MAX_PER_SIDE = 3    # idioms shown per language per row
NEAR = 0.95         # same-language expansion, as in the en-zh original
TOP_PAIRS = 4000    # highest-similarity cross-lingual pairs considered per language pair
N_ROWS = 6


def _clean(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


# The glosses share stock openings ("Used ironically to describe someone who..."), and
# keying clusters on the raw string merges every ironic idiom in the language into one
# 161-member blob. Strip the opening before clustering so the key is the meaning itself.
BOILER = re.compile(
    r"^(?:used\s+(?:ironically\s+|humorously\s+|sarcastically\s+)?(?:to\s+)?"
    r"(?:describe|refer\s+to|mock|express|indicate|say|convey|denote)?\s*"
    r"(?:someone|something|a\s+person|a\s+situation)?\s*(?:who|that|which|where)?\s*"
    r"|refers?\s+to\s+|describes?\s+|denotes?\s+|signifies?\s+"
    r"|a\s+situation\s+(?:in\s+which|where)\s+|the\s+idea\s+that\s+|to\s+)+",
    re.I)


def meaning_key(s):
    k = BOILER.sub("", _clean(s).lower()).strip(" .;:,")
    return k if len(k) >= 12 else _clean(s).lower().strip(" .")


def _gloss_cache():
    """The English paraphrases already produced by gloss.py, read straight from disk."""
    import local_llm as L
    p = os.path.join(L.CACHE_DIR, "v2_gloss_primary.jsonl")
    d = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try:
                o = json.loads(line)
                d[o["h"]] = o["r"]
            except Exception:
                pass
    return d


def load_meanings(lang, cap=None):
    """(idiom, English gloss of its meaning, entities).

    We pair on the *English paraphrase*, not the native-language meaning string, for the
    reason gloss.py exists: embedding Chinese text against Hindi text measures the
    cross-lingual embedding gap as much as the meanings. Pairing natively gave 12 zh-hi
    pairs against 1,932 en-ar, which is an artefact of script, not of the languages.
    Only idioms already glossed by the divergence analysis are used, so this adds no LLM
    calls; coverage is 20-34% of each knowledge base.
    """
    import hashlib
    import local_llm as L
    from gloss import LANG_NAME as GL, PROMPT, _clean as _gclean, _clip
    cache = _gloss_cache()
    rows = []
    for r in common.load_kb(lang):
        if not (r["fig"] and r["idiom"]):
            continue
        prompt = PROMPT.format(lang=GL[lang], idiom=r["idiom"],
                               defs=_clip(" | ".join(r["fig"]), 1500))
        k = hashlib.sha1(json.dumps(
            ["vllm", L.NAMES.get(L.PRIMARY, L.PRIMARY), prompt, 0.0]).encode()).hexdigest()
        g = _gclean(cache.get(k, ""))
        if g:
            rows.append({"idiom": r["idiom"], "mean": g[:220],
                         "native": _clean(r["fig"][0])[:220], "ents": r["entities"]})
    return rows[:cap] if cap else rows


def near_neighbours(V, thr=NEAR, step=2048):
    """Same-language idioms whose meaning is nearly identical, as index lists.

    The en-zh original builds a bilingual cluster from one cross-lingual pair by
    attaching, on each side, the idioms of that same language whose meaning is nearly
    identical to the anchor. Counting the b side by its cross-lingual matches instead
    (which an earlier version did) makes every cluster look like 1-vs-many, because the
    a side is then an anchor by construction and the b side is a 0.70 neighbourhood.
    """
    out = []
    for i0 in range(0, len(V), step):
        S = V[i0:i0 + step] @ V.T
        for r in range(S.shape[0]):
            out.append(np.flatnonzero(S[r] >= thr).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=N_ROWS)
    ap.add_argument("--sim", type=float, default=SIM)
    ap.add_argument("--out", default="meaning_cases_pairs_candidates.json")
    args = ap.parse_args()

    langs = sorted({l for p in PAIRS for l in p})
    data, vecs, near = {}, {}, {}
    for l in langs:
        data[l] = load_meanings(l)
        print(f"[embed] {l}: {len(data[l])} meanings", flush=True)
        vecs[l] = common.embed([r["mean"] for r in data[l]], batch_size=256)
        near[l] = near_neighbours(vecs[l])
        print(f"[near]  {l}: median cluster size "
              f"{int(np.median([len(x) for x in near[l]]))}", flush=True)

    out = {"method": __doc__, "sim_threshold": args.sim, "near_threshold": NEAR,
           "pairs": {}}
    for a, b in PAIRS:
        A, B = vecs[a], vecs[b]
        da, db = data[a], data[b]
        cands = []
        step = 2048
        for i0 in range(0, len(A), step):
            S = A[i0:i0 + step] @ B.T
            for i, j in np.argwhere(S >= args.sim):
                ia, jb = i0 + int(i), int(j)
                if da[ia]["ents"] and db[jb]["ents"]:
                    cands.append((float(S[i, j]), ia, jb))
        print(f"[{a}-{b}] {len(cands)} cross-lingual meaning pairs at cos>={args.sim}",
              flush=True)
        cands.sort(key=lambda t: -t[0])

        rows, seen = [], set()
        for sim, ia, jb in cands[:TOP_PAIRS]:
            ai, bi = near[a][ia], near[b][jb]
            k = (min(ai), min(bi))
            if k in seen:
                continue
            seen.add(k)
            ea = set().union(*[set(da[i]["ents"]) for i in ai])
            eb = set().union(*[set(db[j]["ents"]) for j in bi])
            if ea & eb:            # shared imagery cannot illustrate the contrast
                continue
            # keep the anchor first: it is the idiom the similarity was measured on
            ai = [ia] + [i for i in ai if i != ia]
            bi = [jb] + [j for j in bi if j != jb]
            rows.append({
                "key": da[ia]["mean"][:70].lower(), "sim": round(sim, 4),
                "n_a": len(ai), "n_b": len(bi),
                "a": [{"idiom": da[i]["idiom"], "mean": da[i]["mean"],
                       "native": da[i]["native"], "ents": da[i]["ents"]} for i in ai[:8]],
                "b": [{"idiom": db[j]["idiom"], "mean": db[j]["mean"],
                       "native": db[j]["native"], "ents": db[j]["ents"]} for j in bi[:8]],
                "entity_overlap": 0,
                "asymmetry": abs(len(ai) - len(bi)),
            })
        rows.sort(key=lambda r: (-r["sim"], -r["asymmetry"]))
        sizes = [r["n_a"] + r["n_b"] for r in rows]
        print(f"[{a}-{b}] {len(rows)} disjoint-imagery clusters, "
              f"median size {int(np.median(sizes)) if sizes else 0}", flush=True)
        out["pairs"][f"{a}-{b}"] = {
            "n_pairs": len(cands), "n_clusters": len(rows),
            "candidates": rows[: args.rows * 8],
        }

    print("wrote", common.dump(out, args.out))


if __name__ == "__main__":
    main()
