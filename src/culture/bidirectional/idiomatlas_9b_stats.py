#!/usr/bin/env python3
"""IdiomAtlas-MC statistics for the 9B checkpoints: accuracies, paired contrasts, and the
familiarity analysis (which wrong option Idiom-CPT picks on unseen items).
Writes docs/paper_stats/v2/idiomatlas_9b.json."""
import json

import numpy as np

from culture.evaluation.compute_cis import paired

B = "/data/group_data/r3lit_culture_pretrain/culture/bidir"
D = "culture/data"
SEEN = {"ar": set(json.load(open(f"{B}/ar_idiom_counts.json"))["seen"]),
        "hi": set(json.load(open(f"{D}/mc4_corpus/hi/kept_idiom_counts_hi.json"))),
        "zh": set(json.load(open(f"{D}/fwe_corpus/zh/kept_idiom_counts_zh.json")))}
rng = np.random.default_rng(0)
out = {}
for L in ["ar", "hi", "zh"]:
    for split in ["seen", "unseen"]:
        recs = {}
        for r in ["base", "unfiltered", "untagged", "cpt"]:
            try:
                recs[r] = json.load(open(f"{B}/eval9b/{L}/{r}/idiomatlas_mc_{L}_{split}.json"))["records"]
            except FileNotFoundError:
                pass
        meta = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{B}/eval_data/mc/idiomatlas_mc_{L}_{split}.jsonl")}
        acc = {r: np.array([int(x["correct_norm"]) for x in v]) for r, v in recs.items()}
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
