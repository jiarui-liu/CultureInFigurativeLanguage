#!/usr/bin/env python3
"""Assemble the document lists of every training arm of the bidirectional study.

Each arm is written as a jsonl.gz in the order in which it will be packed; the
exact token budget is imposed afterwards by ``tokenize_pack --max_tokens``, so
every list only needs to hold at least the budget (``--oversample`` x budget).

  random          uniform sample of the pool (same gates as the culture arms)
  culture_docs    idiom-free pool documents ranked by culture score (best first)
  idiom_tagged    random sample of the 9B idiom-annotated corpus (docs + meaning tags)
  idiom_untagged  the same documents in the same order with the tags stripped
  (culture_notes is assembled from culture_notes.py output with --cmd notes_arm)

Also writes <out_dir>/arm_stats.json: documents, characters, idiom density per arm,
and the idiom rate among the top culture-scored documents *before* the idiom-free
filter versus the pool base rate (how often culture-rich text carries idioms).

Usage:
  python -m culture.bidirectional.build_arms --lang ar --budget 300000000 --out_dir $B/arms/ar
  python -m culture.bidirectional.build_arms --lang ar --cmd notes_arm --out_dir $B/arms/ar
"""
import argparse
import ast
import glob
import gzip
import json
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
B = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
TAGGED = {
    "zh": sorted(glob.glob(str(REPO / "culture/data/fwe_tagged/zh/*.json.gz"))),
    "hi": sorted(glob.glob(str(REPO / "culture/data/mc4_tagged/hi/*.json.gz"))),
    "ar": sorted(glob.glob(str(B / "hf9b/data/ar-amthal-cpt/data/tagged_*.json.gz"))),
}
SPLIT = {
    "zh": "\n\n【成语注释】",
    "ar": "\n\nالمعاني الاصطلاحية للتعابير الواردة في النص:",
}
# characters per Qwen3.5 token, measured on 2k pool documents per language (see --measure)
CPT = {"zh": 1.76, "hi": 1.58, "ar": 3.14}


def strip_tags(lang, d):
    if lang == "hi":
        return d["text"][: int(d["original_text_chars"])]
    return d["text"].split(SPLIT[lang])[0]


def matched(d):
    m = d.get("matched_idioms", [])
    return ast.literal_eval(m) if isinstance(m, str) else m


def write(path, docs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def load_scores(lang):
    sc = {}
    for f in glob.glob(str(B / f"scores/{lang}/*.scores.jsonl")):
        for l in open(f):
            o = json.loads(l)
            sc[o["id"]] = o
    return sc


def iter_pool(lang):
    for f in sorted(glob.glob(str(B / f"pool/{lang}/*.jsonl.gz"))):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for l in fh:
                yield json.loads(l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--cmd", default="base", choices=["base", "culture", "notes_arm", "full", "full_notes"],
                    help="base: random + idiom arms (no classifier needed); culture: culture_docs")
    ap.add_argument("--budget", type=int, default=300_000_000)
    ap.add_argument("--oversample", type=float, default=1.3)
    ap.add_argument("--notes_docs_factor", type=float, default=1.0,
                    help="culture docs sent to the notes generator, as a multiple of the budget")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    need_chars = a.budget * CPT[a.lang] * a.oversample

    if a.cmd == "full":
        # full-scale (9B) Culture corpus from the already scored pool: idiom-free docs ranked
        # by culture score up to --budget tokens, shuffled, written as {"text"} shards
        scores = load_scores(a.lang)
        ranked = [o for o in sorted(scores.values(), key=lambda o: -o["score"]) if o["n_idioms"] == 0]
        need = a.budget * CPT[a.lang]
        tot, keep = 0, {}
        for o in ranked:
            keep[o["id"]] = o["score"]
            tot += o["n_chars"]
            if tot >= need:
                break
        docs = [{"id": d["id"], "text": d["text"]} for d in iter_pool(a.lang) if d["id"] in keep]
        rng.shuffle(docs)
        od = out / "train"
        od.mkdir(parents=True, exist_ok=True)
        for k in range(0, len(docs), 50000):
            with open(od / f"train_{k // 50000:05d}.jsonl", "w", encoding="utf-8") as fo:
                for d in docs[k:k + 50000]:
                    fo.write(json.dumps({"text": d["text"]}, ensure_ascii=False) + "\n")
        write(out / "culture_docs_ranked.jsonl.gz",
              sorted(docs, key=lambda d: -keep[d["id"]]))  # input for culture_notes.py
        st = {"selected": len(docs), "chars": tot, "est_tokens": tot / CPT[a.lang],
              "score_cutoff": min(keep.values()), "pool_docs": len(scores),
              "pool_idiom_free": len(ranked)}
        json.dump(st, open(out / "full_stats.json", "w"), indent=1)
        print(st)
        return

    if a.cmd == "full_notes":
        # full-scale Culture+notes corpus: every selected document with its cultural notes appended
        notes = {}
        for f in sorted(glob.glob(str(out / "notes/notes_*.jsonl.gz"))):
            for l in gzip.open(f, "rt", encoding="utf-8"):
                o = json.loads(l)
                notes[o["id"]] = o["text"]
        docs = [notes.get(json.loads(l)["id"], json.loads(l)["text"])
                for l in gzip.open(out / "culture_docs_ranked.jsonl.gz", "rt", encoding="utf-8")]
        rng.shuffle(docs)
        od = out / "train_notes"
        od.mkdir(parents=True, exist_ok=True)
        for k in range(0, len(docs), 50000):
            with open(od / f"train_{k // 50000:05d}.jsonl", "w", encoding="utf-8") as fo:
                for t in docs[k:k + 50000]:
                    fo.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print({"docs": len(docs), "with_notes": len(notes)})
        return

    if a.cmd == "notes_arm":
        notes = {}
        for f in sorted(glob.glob(str(out / "notes/notes_*.jsonl.gz"))):
            for l in gzip.open(f, "rt", encoding="utf-8"):
                o = json.loads(l)
                notes[o["id"]] = o
        docs, miss = [], 0
        for l in gzip.open(out / "culture_docs.jsonl.gz", "rt", encoding="utf-8"):
            d = json.loads(l)
            if d["id"] in notes:
                docs.append({"id": d["id"], "text": notes[d["id"]]["text"]})
            else:
                miss += 1
                break  # keep a strict prefix of the ranking
        write(out / "culture_notes.jsonl.gz", docs)
        print("culture_notes docs", len(docs), "stopped at first doc without notes:", miss)
        return

    stats_path = out / "arm_stats.json"
    stats = json.load(open(stats_path)) if stats_path.exists() else {}
    if a.cmd == "base":
        # random arm: every pool document kept independently with prob. p (seeded);
        # p is set from the document count and the mean length of the first shard
        # (single pass over the pool; the budget is enforced later by tokenize_pack)
        n_docs = 0
        for f in sorted(glob.glob(str(B / f"pool/{a.lang}/*.stats.json"))):
            n_docs += json.load(open(f))["kept"]
        first = sorted(glob.glob(str(B / f"pool/{a.lang}/*.jsonl.gz")))[0]
        lens = [len(json.loads(l)["text"]) for l in gzip.open(first, "rt", encoding="utf-8")]
        tot_chars = n_docs * sum(lens) / len(lens)
        p = min(1.0, 1.1 * need_chars / tot_chars)
        rand_docs = [{"id": d["id"], "text": d["text"], "n_idioms": len(d["idioms"])}
                     for d in iter_pool(a.lang) if rng.random() < p]
        rng.shuffle(rand_docs)
        write(out / "random.jsonl.gz", rand_docs)
        stats["pool"] = {"docs": n_docs, "chars": tot_chars}
        stats["random"] = {"docs": len(rand_docs),
                           "idiom_doc_rate": sum(d["n_idioms"] > 0 for d in rand_docs) / len(rand_docs)}
        # ---------------- idiom arms (9B idiom-annotated corpus) ----------------
        files = TAGGED[a.lang][:]
        rng.shuffle(files)
        tag_docs, tot = [], 0
        for f in files:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for l in fh:
                    d = json.loads(l)
                    tag_docs.append(d)
                    tot += len(d["text"])
            if tot >= need_chars:
                break
        rng.shuffle(tag_docs)
        write(out / "idiom_tagged.jsonl.gz", [{"text": d["text"], "idioms": matched(d)} for d in tag_docs])
        write(out / "idiom_untagged.jsonl.gz", [{"text": strip_tags(a.lang, d)} for d in tag_docs])
        stats["idiom_tagged"] = {"docs": len(tag_docs), "files": len(files)}


        json.dump(stats, open(stats_path, "w"), indent=1)
        print(json.dumps(stats, indent=1))
        return

    # culture arm: idiom-free pool documents ranked by classifier score
    scores = load_scores(a.lang)
    print(a.lang, "scored pool docs", len(scores))
    ranked = sorted(scores.values(), key=lambda o: -o["score"])
    tot, top_all = 0, []
    for o in ranked:
        top_all.append(o)
        tot += o["n_chars"]
        if tot >= need_chars:
            break
    tot, cul = 0, []
    for o in ranked:
        if o["n_idioms"]:
            continue
        cul.append(o)
        tot += o["n_chars"]
        if tot >= need_chars * a.notes_docs_factor:
            break
    base_rate = sum(o["n_idioms"] > 0 for o in scores.values()) / len(scores)
    stats["pool_scores"] = {"docs": len(scores), "idiom_doc_rate": base_rate,
                            "mean_score": sum(o["score"] for o in scores.values()) / len(scores),
                            "share_score_ge3": sum(o["score"] >= 3 for o in scores.values()) / len(scores)}
    stats["top_culture_before_idiom_filter"] = {
        "docs": len(top_all), "idiom_doc_rate": sum(o["n_idioms"] > 0 for o in top_all) / len(top_all),
        "score_cutoff": top_all[-1]["score"]}
    stats["culture"] = {"docs": len(cul), "score_cutoff": cul[-1]["score"],
                        "mean_score": sum(o["score"] for o in cul) / len(cul), "idiom_doc_rate": 0.0}
    rank_of = {o["id"]: k for k, o in enumerate(cul)}
    cul_docs = [None] * len(cul)
    rand_ids = set()
    if (out / "random.jsonl.gz").exists():
        for l in gzip.open(out / "random.jsonl.gz", "rt", encoding="utf-8"):
            rand_ids.add(json.loads(l)["id"])
        rs = [scores[i]["score"] for i in rand_ids if i in scores]
        stats["random"]["mean_score"] = sum(rs) / max(1, len(rs))
    for d in iter_pool(a.lang):
        k = rank_of.get(d["id"])
        if k is not None:
            cul_docs[k] = {"id": d["id"], "text": d["text"], "score": cul[k]["score"]}
    write(out / "culture_docs.jsonl.gz", [d for d in cul_docs if d])
    json.dump(stats, open(stats_path, "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
