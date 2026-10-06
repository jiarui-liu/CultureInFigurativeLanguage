"""Evaluative polarity of shared entities across the four languages.

The embedding-based divergence of entity_divergence_{en_zh,multi,cross}.py is explicitly
insensitive to valence -- the paper notes that the dog, whose evaluation differs sharply between
English and Chinese, lands among the *least* divergent entities. This script measures the axis
that metric misses: for every entity that several languages share, how does each language's
idiom stock *evaluate* it?

An LLM reads up to CAP idioms (with their figurative meanings) for one entity in one language
and returns a valence in [-2, +2] plus a short characterization, judging only from the evidence
shown. A second model from a different family rescoores a subsample. We then report, per language
pair, the mean absolute valence gap, the rate of sign flips (an entity praised in one language
and disparaged in the other), and the correlation between the valence gap and the embedding
divergence index -- the last being the test of whether valence is a genuinely separate axis.

    PYTHONPATH=src:src/culture/analysis/v2 python entity_valence.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from itertools import combinations

import numpy as np

import common
import local_llm
from common import OUT, dump, entity_counter, entity_index, load_kb

LANGS = ("en", "zh", "hi", "ar")
CAP = 15
MIN_N = 5
LANG_NAME = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}

PROMPT = """Below are {n} {lang} idioms that mention the entity "{ent}" ({ent_en}), each with its
figurative meaning.

{evidence}

Judging ONLY from this evidence, how does the {lang} idiom stock evaluate "{ent_en}"?

Reply with one JSON object and nothing else:
{{"valence": <integer -2..2>, "gloss": "<3-6 words naming what it connotes>"}}

valence: -2 strongly negative (contempt, blame), -1 mildly negative, 0 neutral or mixed,
+1 mildly positive, +2 strongly positive (admiration, praise)."""


def load_anchor_groups(lang):
    """english anchor -> [native entity strings] for that language."""
    if lang == "en":
        return None  # handled by the caller: the entity is its own anchor
    p = os.path.join(OUT, f"entity_translations_{lang}_en.json")
    groups = defaultdict(list)
    with open(p, encoding="utf-8") as f:
        for e, t in json.load(f).items():
            if t and t.get("en") and t["en"] != "none":
                groups[t["en"]].append(e)
    return groups


def parse_obj(resp):
    m = re.search(r"\{.*?\}", resp or "", re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        v = int(o["valence"])
    except Exception:
        return None
    if not -2 <= v <= 2:
        return None
    return {"valence": v, "gloss": str(o.get("gloss", ""))[:80]}


def build_tasks(args):
    rng = random.Random(0)
    tasks = []  # (lang, anchor, prompt)
    for lang in LANGS:
        kb = load_kb(lang)
        idx = entity_index(kb)
        groups = load_anchor_groups(lang)
        if groups is None:
            cnt = entity_counter(kb)
            groups = {e: [e] for e, _ in cnt.most_common(args.top_en)}
        for anchor, natives in groups.items():
            ids = sorted({j for e in natives for j in idx.get(e, []) if kb[j]["fig"]})
            if len(ids) < MIN_N:
                continue
            pick = rng.sample(ids, CAP) if len(ids) > CAP else ids
            lines = []
            for j in pick:
                meaning = re.sub(r"\s+", " ", kb[j]["fig"][0])[:160]
                lines.append(f"- {kb[j]['idiom']} — {meaning}")
            ev = "\n".join(lines)
            tasks.append(
                {
                    "lang": lang,
                    "anchor": anchor,
                    "n_idioms": len(ids),
                    "prompt": PROMPT.format(
                        n=len(pick), lang=LANG_NAME[lang], ent=natives[0],
                        ent_en=anchor, evidence=ev[:4000],
                    ),
                }
            )
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_en", type=int, default=500)
    ap.add_argument("--agree_n", type=int, default=200)
    ap.add_argument("--out", default="entity_valence.json")
    # See culture_layer_taxonomy.py: one model per process.
    ap.add_argument("--skip_second", action="store_true")
    args = ap.parse_args()

    tasks = build_tasks(args)
    print(f"[tasks] {len(tasks)} (lang, entity) pairs")
    for l in LANGS:
        print(f"   {l}: {sum(1 for t in tasks if t['lang'] == l)}")

    resp = local_llm.generate(
        [t["prompt"] for t in tasks], "entity_valence_primary", max_tokens=64
    )
    val = defaultdict(dict)  # lang -> anchor -> {valence, gloss}
    bad = 0
    for t, r in zip(tasks, resp):
        o = parse_obj(r)
        if o is None:
            bad += 1
            continue
        o["n_idioms"] = t["n_idioms"]
        val[t["lang"]][t["anchor"]] = o
    print(f"[parse] unparsed={bad}")

    # Reliability: second family on a subsample.
    pa, pb = [], []
    if not args.skip_second:
        rng = random.Random(1)
        sub = rng.sample(range(len(tasks)), min(args.agree_n, len(tasks)))
        sec = local_llm.generate(
            [tasks[i]["prompt"] for i in sub], "entity_valence_second",
            model=local_llm.THIRD, max_tokens=64,
        )
        for i, r in zip(sub, sec):
            o2 = parse_obj(r)
            o1 = val[tasks[i]["lang"]].get(tasks[i]["anchor"])
            if o2 and o1:
                pa.append(o1["valence"])
                pb.append(o2["valence"])
    rel = {}
    if len(pa) > 5:
        a, b = np.array(pa, float), np.array(pb, float)
        rel = {
            "n": len(pa),
            "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "exact_agree": round(float(np.mean(a == b)), 4),
            "sign_agree": round(float(np.mean(np.sign(a) == np.sign(b))), 4),
            "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 4),
        }

    # Cross-language comparison.
    pairs = {}
    for x, y in combinations(LANGS, 2):
        shared = sorted(set(val[x]) & set(val[y]))
        if len(shared) < 10:
            continue
        vx = np.array([val[x][a]["valence"] for a in shared], float)
        vy = np.array([val[y][a]["valence"] for a in shared], float)
        flip = (np.sign(vx) * np.sign(vy)) < 0
        gap = np.abs(vx - vy)
        order = np.argsort(-gap)
        pairs[f"{x}-{y}"] = {
            "n_entities": len(shared),
            "mean_abs_gap": round(float(gap.mean()), 4),
            "sign_flip_rate": round(float(flip.mean()), 4),
            "pearson_r": round(float(np.corrcoef(vx, vy)[0, 1]), 4)
            if vx.std() and vy.std() else None,
            "mean_valence_a": round(float(vx.mean()), 3),
            "mean_valence_b": round(float(vy.mean()), 3),
            "largest_gaps": [
                {
                    "entity": shared[i],
                    x: {"v": int(vx[i]), "gloss": val[x][shared[i]]["gloss"]},
                    y: {"v": int(vy[i]), "gloss": val[y][shared[i]]["gloss"]},
                }
                for i in order[:25]
            ],
            "sign_flips": [
                {
                    "entity": shared[i],
                    x: {"v": int(vx[i]), "gloss": val[x][shared[i]]["gloss"]},
                    y: {"v": int(vy[i]), "gloss": val[y][shared[i]]["gloss"]},
                }
                for i in np.where(flip)[0]
            ],
        }

    # Is valence a different axis from the embedding divergence? (en-zh, where per-entity exists)
    vs_div = None
    try:
        import csv

        p = os.path.join(OUT, "entity_divergence_en_zh_per_entity.csv")
        div = {
            r["entity_en"]: float(r["gloss_centroid_div"])
            for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["gloss_centroid_div"]
        }
        common_e = sorted(set(div) & set(val["en"]) & set(val["zh"]))
        if len(common_e) > 10:
            d = np.array([div[a] for a in common_e])
            g = np.array(
                [abs(val["en"][a]["valence"] - val["zh"][a]["valence"]) for a in common_e],
                float,
            )
            vs_div = {
                "n": len(common_e),
                "pearson_r_divergence_vs_valencegap": round(
                    float(np.corrcoef(d, g)[0, 1]), 4
                ),
                "note": "near zero => valence is an axis the embedding metric does not capture",
            }
    except Exception as e:
        vs_div = {"error": str(e)}

    out = {
        "method": __doc__,
        "cap_idioms": CAP,
        "min_idioms": MIN_N,
        "primary_model": local_llm.NAMES.get(local_llm.PRIMARY, local_llm.PRIMARY),
        "second_model": local_llm.NAMES.get(local_llm.THIRD, local_llm.THIRD),
        "reliability": rel,
        "n_scored": {l: len(val[l]) for l in LANGS},
        "pairs": pairs,
        "valence_vs_embedding_divergence": vs_div,
        "per_entity": {l: val[l] for l in LANGS},
    }
    print("wrote", dump(out, args.out))
    for k, v in pairs.items():
        print(f"  {k}: gap={v['mean_abs_gap']:.2f} flips={v['sign_flip_rate']:.3f} "
              f"n={v['n_entities']}")
    if vs_div:
        print("  valence vs embedding divergence:", vs_div)


if __name__ == "__main__":
    main()
