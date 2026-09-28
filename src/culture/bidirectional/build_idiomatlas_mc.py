#!/usr/bin/env python3
"""Build IdiomAtlas-MC: a KB-derived idiom-meaning selection test for hi / zh / ar.

Each item gives an idiom and four candidate figurative meanings: the gold meaning
(the first figurative meaning in IdiomAtlas) and three distractors, which are the
first meanings of other idioms of the same language chosen to be topically related
but not paraphrases (embedding cosine to the gold in [lo, hi]) and of comparable
length. Options are scored as continuations (acc_norm).

Items are split by corpus exposure:
  seen   - the idiom occurs in the idiom-annotated CPT corpus, so its meaning
           appears in the meaning tags that Idiom-CPT trains on;
  unseen - the idiom never occurs in any scanned web document, so no training arm
           has seen it or its meaning; this split tests generalization.

Usage:
  python -m culture.bidirectional.build_idiomatlas_mc --lang zh --out_dir $B/eval_data/mc
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
D = REPO / "culture/data"
KB = {
    "zh": D / "idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl",
    "hi": D / "idioms/hi/idioms_merged_llm_formatted_figurative_only.jsonl",
    "ar": D / "idioms/ar/idioms_merged_llm_formatted.jsonl",
}
TEMPLATE = {
    "zh": "成语「{idiom}」的意思是：",
    "hi": "लोकोक्ति \"{idiom}\" का अर्थ है:",
    "ar": "معنى المثل «{idiom}» هو:",
}
EMB = "/data/group_data/r3lit_culture_pretrain/models/Qwen/Qwen3-Embedding-0.6B"


def seen_counts(lang, extra):
    """Return (seen: set of idioms in the CPT corpus, ever: set of idioms seen in any scan)."""
    if lang == "zh":
        kept = json.load(open(D / "fwe_corpus/zh/kept_idiom_counts_zh.json"))
        mc4 = json.load(open(D / "mc4_filtered/idiom_doc_counts_zh.json"))["idiom_doc_counts"]
        return set(kept), set(kept) | {k for k, v in mc4.items() if v > 0}
    if lang == "hi":
        kept = json.load(open(D / "mc4_corpus/hi/kept_idiom_counts_hi.json"))
        return set(kept), set(kept)
    # ar: counts from the local filter_and_tag_ar run(s) + pool scans (see --ar_counts)
    seen, ever = set(), set()
    for p in extra:
        c = json.load(open(p))
        seen |= set(c.get("seen", []))
        ever |= set(c.get("ever", []))
    return seen, ever | seen


def units(lang, s):
    """Comparison units for lexical overlap: characters for Chinese, words (>=2 chars) otherwise."""
    if lang == "zh":
        return {c for c in s if "\u4e00" <= c <= "\u9fff"}
    if lang == "ar":
        from culture.data_processing.ar_idioms.normalize import normalize_ar
        s = normalize_ar(s)
    return {w for w in re.findall(r"\w+", s) if len(w) >= 2}


def overlap(lang, idiom, meaning):
    """Share of the idiom's units that also occur in the meaning."""
    u = units(lang, idiom)
    return len(u & units(lang, meaning)) / max(1, len(u))


def clean_meaning(lang, m):
    m = re.sub(r"\s+", " ", str(m)).strip()
    if lang == "ar":
        from culture.training.mC4.filter_and_tag_ar import _gloss
        m = _gloss(m, cap=200)
    return m[:200].rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["zh", "hi", "ar"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_split", type=int, default=600)
    ap.add_argument("--lo", type=float, default=0.35)
    ap.add_argument("--hi", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ar_counts", nargs="*", default=[])
    ap.add_argument("--max_overlap", type=float, default=0.25,
                    help="drop items whose gold (or any distractor) shares more than this "
                         "share of the idiom's words/characters, removing lexical shortcuts")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = [json.loads(l) for l in open(KB[a.lang], encoding="utf-8") if l.strip()]
    items = []
    for r in rows:
        o = r["output"]
        fm = o.get("figurative_meanings")
        if not fm or fm == "NAN":
            continue
        fm = fm if isinstance(fm, list) else [fm]
        m = clean_meaning(a.lang, fm[0])
        idiom = o["idiom"].strip()
        if len(m) < 4 or not idiom or idiom in m:
            continue
        items.append({"idiom": idiom, "meaning": m})
    # one entry per idiom string
    uniq = {}
    for it in items:
        uniq.setdefault(it["idiom"], it)
    items = list(uniq.values())
    seen, ever = seen_counts(a.lang, a.ar_counts)
    for it in items:
        it["split"] = "seen" if it["idiom"] in seen else ("unseen" if it["idiom"] not in ever else "other")
    print(a.lang, len(items), {s: sum(it["split"] == s for it in items) for s in ["seen", "unseen", "other"]})

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB, device="cuda")
    E = model.encode([it["meaning"] for it in items], batch_size=256, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    lens = np.array([len(it["meaning"]) for it in items])

    out = []
    for it in items:
        it["ov"] = overlap(a.lang, it["idiom"], it["meaning"])
    for split in ["seen", "unseen"]:
        idx = [i for i, it in enumerate(items) if it["split"] == split and it["ov"] <= a.max_overlap]
        rng.shuffle(idx)
        for i in idx:
            if sum(1 for x in out if x["meta"]["split"] == split) >= a.n_per_split:
                break
            sims = E @ E[i]
            ok = np.where((sims >= a.lo) & (sims <= a.hi) & (lens >= 0.5 * lens[i]) &
                          (lens <= 2.0 * lens[i]))[0]
            ok = [j for j in ok if j != i and items[j]["meaning"] != items[i]["meaning"]
                  and overlap(a.lang, items[i]["idiom"], items[j]["meaning"]) <= a.max_overlap]
            if len(ok) < 3:
                continue
            ds = rng.sample(ok, 3)
            opts = [items[i]["meaning"]] + [items[j]["meaning"] for j in ds]
            order = list(range(4))
            rng.shuffle(order)
            opts = [opts[k] for k in order]
            gold = order.index(0)
            out.append({"qid": f"idiomatlas_{a.lang}/{split}/{len(out)}",
                        "context": TEMPLATE[a.lang].format(idiom=items[i]["idiom"]),
                        "options": [" " + x for x in opts], "gold": gold,
                        "meta": {"idiom": items[i]["idiom"], "split": split,
                                 "distractor_idioms": [items[j]["idiom"] for j in ds],
                                 "distractor_sims": [round(float(E[i] @ E[j]), 3) for j in ds]}})
    os.makedirs(a.out_dir, exist_ok=True)
    fn = Path(a.out_dir) / f"idiomatlas_mc_{a.lang}.jsonl"
    with open(fn, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    # also write per-split files so they can be scored as separate tasks
    for split in ["seen", "unseen"]:
        with open(Path(a.out_dir) / f"idiomatlas_mc_{a.lang}_{split}.jsonl", "w", encoding="utf-8") as f:
            for x in out:
                if x["meta"]["split"] == split:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
    # lexical-overlap baseline: pick the option sharing most units with the idiom
    hit = 0
    for x in out:
        ov = [overlap(a.lang, x["meta"]["idiom"], o) for o in x["options"]]
        hit += int(max(range(4), key=lambda k: (ov[k], -k)) == x["gold"])
    print("wrote", fn, len(out), "overlap-baseline acc", round(hit / max(1, len(out)), 3))


if __name__ == "__main__":
    main()
