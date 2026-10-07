#!/usr/bin/env python3
"""IdiomAtlas-MC statistics for the 9B checkpoints: accuracies, paired contrasts, and the
familiarity analysis (which wrong option Idiom-CPT picks on unseen items).
Writes docs/paper_stats/v2/idiomatlas_9b.json."""
import json

import numpy as np

from culture.evaluation.compute_cis import paired

EVAL = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval"
MC = "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc"
D = "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data"
SEEN = {"ar": set(json.load(open(f"{D}/ar-amthal-cpt/stats/kept_idiom_counts_ar.json"))),
        "hi": set(json.load(open(f"{D}/hi-proverbs-cpt/stats/kept_idiom_counts_hi.json"))),
        "zh": set(json.load(open(f"{D}/fineweb-edu-zh-chengyu-cpt/stats/kept_idiom_counts_zh.json")))}
rng = np.random.default_rng(0)
out = {}
for L in ["ar", "hi", "zh"]:
    for split in ["seen", "unseen"]:
        recs = {}
        for r in ["base", "unfiltered", "untagged", "cpt"]:
            try:
                recs[r] = json.load(open(f"{EVAL}/{L}/{r}/idiomatlas_mc_{L}_{split}.json"))["records"]
            except FileNotFoundError:
                pass
        meta = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{MC}/idiomatlas_mc_{L}_{split}.jsonl")}
        # align every arm on the same qid order, so the paired bootstrap pairs the same item
        qids = sorted(set.intersection(*[{x["qid"] for x in v} for v in recs.values()]))
        acc = {r: np.array([int({x["qid"]: x for x in v}[q]["correct_norm"]) for q in qids])
               for r, v in recs.items()}
        row = {r: round(100 * a.mean(), 1) for r, a in acc.items()}
        def c(a, b):
            d, lo, hi, _, _, p = paired(acc[a], acc[b], rng)
            return (round(100 * d, 1), [round(100 * lo, 1), round(100 * hi, 1)], round(p, 4))
        row["cpt-unfiltered"] = c("cpt", "unfiltered")
        row["cpt-base"] = c("cpt", "base")
        if "untagged" in acc:
            row["cpt-untagged"] = c("cpt", "untagged")
            row["untagged-unf"] = c("untagged", "unfiltered")
        if split == "unseen":
            for r in ["unfiltered", "cpt"]:
                wrong = seenpick = 0
                avail = 0.0
                for x in recs[r]:
                    m = meta[x["qid"]]
                    pred, gold = int(x["pred_norm"]), m["gold"]
                    didi = m["meta"]["distractor_idioms"]
                    opt = dict(zip([k for k in range(4) if k != gold], didi))
                    avail += sum(d in SEEN[L] for d in didi) / 3
                    if pred != gold:
                        wrong += 1
                        seenpick += opt[pred] in SEEN[L]
                row[f"{r}_wrong_pick_seen_share"] = round(seenpick / max(1, wrong), 3)
                row["seen_distractor_share"] = round(avail / len(recs[r]), 3)
        out[f"{L}_{split}"] = row
        print(L, split, row)
json.dump(out, open("docs/paper_stats/v2/idiomatlas_9b.json", "w"), indent=1)
