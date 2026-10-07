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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=N_ROWS)
    ap.add_argument("--sim", type=float, default=SIM)
    ap.add_argument("--out", default="meaning_cases_pairs_candidates.json")
    args = ap.parse_args()

    langs = sorted({l for p in PAIRS for l in p})
    data, vecs = {}, {}
    for l in langs:
        data[l] = load_meanings(l)
        print(f"[embed] {l}: {len(data[l])} meanings", flush=True)
        vecs[l] = common.embed([r["mean"] for r in data[l]], batch_size=256)

    out = {"method": __doc__, "sim_threshold": args.sim, "pairs": {}}
    for a, b in PAIRS:
        A, B = vecs[a], vecs[b]
        da, db = data[a], data[b]
        cands = []
        # blocked matmul: the full matrix would be ~20k x 27k
        step = 2048
        for i0 in range(0, len(A), step):
            S = A[i0:i0 + step] @ B.T
            idx = np.argwhere(S >= args.sim)
            for i, j in idx:
                ia, jb = i0 + int(i), int(j)
                ea, eb = set(da[ia]["ents"]), set(db[jb]["ents"])
                if not ea or not eb:
                    continue
                cands.append({"sim": float(S[i, j]), "a": ia, "b": jb})
        print(f"[{a}-{b}] {len(cands)} cross-lingual meaning pairs at cos>={args.sim}",
              flush=True)

        # group by the English-side (or a-side) meaning, so one row = one meaning
        groups = defaultdict(lambda: {"a": set(), "b": set(), "sim": 0.0})
        for c in cands:
            k = meaning_key(da[c["a"]]["mean"])[:70]
            g = groups[k]
            g["a"].add(c["a"])
            g["b"].add(c["b"])
            g["sim"] = max(g["sim"], c["sim"])

        rows = []
        for k, g in groups.items():
            ai, bi = sorted(g["a"]), sorted(g["b"])
            # Clusters of 20+ a side are an artefact of a vague gloss, not of one
            # language elaborating a meaning; they read as noise in the table.
            if not (1 <= len(ai) <= 12 and 1 <= len(bi) <= 12):
                continue
            ea = set().union(*[set(da[i]["ents"]) for i in ai]) if ai else set()
            eb = set().union(*[set(db[j]["ents"]) for j in bi]) if bi else set()
            rows.append({
                "key": k, "sim": round(g["sim"], 4),
                "n_a": len(ai), "n_b": len(bi),
                "a": [{"idiom": da[i]["idiom"], "mean": da[i]["mean"],
                       "native": da[i]["native"], "ents": da[i]["ents"]} for i in ai[:8]],
                "b": [{"idiom": db[j]["idiom"], "mean": db[j]["mean"],
                       "native": db[j]["native"], "ents": db[j]["ents"]} for j in bi[:8]],
                # the point of the table: the two sides use different imagery
                "entity_overlap": len(ea & eb),
                "asymmetry": abs(len(ai) - len(bi)),
            })
        # Only disjoint imagery can illustrate "same meaning, different entity"; among
        # those, prefer the tightest meaning match, then the most asymmetric cluster.
        rows = [r for r in rows if r["entity_overlap"] == 0]
        # Ranking on asymmetry first pinned every row to the size cap, so the table read
        # as six copies of one statistic; the tightest meaning match is the better order
        # and leaves the cluster sizes free to vary.
        rows.sort(key=lambda r: (-r["sim"], -r["asymmetry"]))
        out["pairs"][f"{a}-{b}"] = {
            "n_pairs": len(cands), "n_clusters": len(rows),
            "candidates": rows[: args.rows * 8],
        }

    print("wrote", common.dump(out, args.out))


if __name__ == "__main__":
    main()
