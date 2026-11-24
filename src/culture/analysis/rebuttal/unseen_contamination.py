#!/usr/bin/env python3
"""Are the idioms of the unseen-idiom test sets absent from the documents the 9B models trained on?

Scans the actual 9B continued-pretraining corpora, as released in the HF dataset
Jerry9999/CultureInFigurativeLanguage:
  Idiom-Text corpus  data/train_{ar,hi,zh}_untagged   (Idiom+Meaning = same documents + meaning blocks)
  Random-CPT corpus  data/train_{ar,hi,zh}_unfiltered  (*.jsonl.gz; docs_written matches Table 13)
Culture-CPT corpora are not in the release and are not scanned.

Test sets and what is matched:
  idiom keys : IdiomAtlas-MC unseen (ar/hi/zh), MIDI-ar / MIDI-zh figurative items, Hindi proverbs,
               Chinese xiehouyu (first half, the part shown to the model)
  text keys  : AR-Figurative question text, MABL-hi start phrase, MIDI context sentence
Matching follows the corpus builders: Arabic uses normalize_ar / normalize_idiom_for_matching and
substring matching (as ArabicIdiomMatcher); Hindi strips punctuation and requires word boundaries;
Chinese strips whitespace and punctuation. Aho-Corasick over all keys at once.
  exact : the whole normalized idiom (or sentence) occurs in the document
  loose : a contiguous run of >= max(3, ceil(0.75 n)) of the idiom's n words occurs (n >= 4 words;
          Chinese: >= ceil(0.75 n) characters for idioms of >= 6 characters). Over-inclusive by design.

Subcommands:
  patterns                       build the key file (needs the bidir env for AR-Figurative)
  scan --lang L --corpus C --part i --nparts k --procs p    download + scan shards on node-local disk
  report --out JSON              aggregate, per-item flags, recompute unseen contrasts without hits
"""
import argparse
import glob
import gzip
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
MC = ROOT / "eval_data/mc"
WORK = ROOT / "rebuttal_e4"
REPO_ID = "Jerry9999/CultureInFigurativeLanguage"
CORPUS = {"idiom": "data/train_{L}_untagged", "random": "data/train_{L}_unfiltered"}
SCRATCH = Path(os.environ.get("E4_SCRATCH", "/scratch/jiaruil5/e4"))

PUNCT = re.compile(r"[\s‌‍,.;:!?،؛؟।॥\"'“”‘’«»\-–—_()\[\]{}…/\\|*~`]+")
ZH_STRIP = re.compile(r"[\s　，。、；：！？“”‘’「」『』《》〈〉（）()【】\[\]—…·,.;:!?'\"\-~]+")


def norm(L, s):
    s = str(s or "")
    if L == "ar":
        from culture.data_processing.ar_idioms.normalize import normalize_ar
        return normalize_ar(s)
    s = unicodedata.normalize("NFC", s)
    if L == "zh":
        return ZH_STRIP.sub("", s)
    return " " + PUNCT.sub(" ", s).strip() + " "          # hi: padded for word boundaries


def norm_key(L, s):
    if L == "ar":
        from culture.data_processing.ar_idioms.normalize import normalize_idiom_for_matching
        return normalize_idiom_for_matching(str(s))
    return norm(L, s)


def loose_keys(L, k):
    """Contiguous sub-spans of the normalized key covering >= 75% of it."""
    if L == "zh":
        n = len(k)
        if n < 6:
            return []
        m = math.ceil(0.75 * n)
        return sorted({k[i:i + m] for i in range(n - m + 1)} - {k})
    w = k.split()
    n = len(w)
    if n < 4:
        return []
    m = max(3, math.ceil(0.75 * n))
    out = {" ".join(w[i:i + m]) for i in range(n - m + 1)}
    return sorted((" " + x + " " if L == "hi" else x) for x in out if x)


def text_key(L, s):
    """A distinctive verbatim span of a benchmark sentence (as the paper's overlap check)."""
    k = norm(L, s).strip()
    cap = 20 if L == "zh" else 40
    if len(k) > 60:
        return k[-cap:]
    return (" " + k + " ") if L == "hi" else k


# --------------------------------------------------------------------------- patterns
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def first_line(ctx):
    return ctx.split("\n")[0]


def build_patterns():
    items = []   # {lang, set, qid, kind, raw, key, loose}

    def add(L, set_, qid, kind, raw):
        k = norm_key(L, raw) if kind == "idiom" else text_key(L, raw)
        if L != "hi":
            k = k.strip()
        if len(k.strip()) < (2 if L == "zh" else 3):
            return
        items.append({"lang": L, "set": set_, "qid": qid, "kind": kind, "raw": raw, "key": k,
                      "loose": loose_keys(L, k.strip() if L != "hi" else k) if kind == "idiom" else []})

    for L in ["ar", "hi", "zh"]:
        for x in load_jsonl(MC / f"idiomatlas_mc_{L}_unseen.jsonl"):
            add(L, f"idiomatlas_mc_{L}_unseen", x["qid"], "idiom", x["meta"]["idiom"])
    for L in ["ar", "zh"]:
        for x in load_jsonl(MC / f"midi_{L}.jsonl"):
            if x["meta"].get("use") != "figurative":
                continue
            add(L, f"midi_{L}", x["qid"], "idiom", x["meta"]["idiom"])
            add(L, f"midi_{L}:sentence", x["qid"], "text", first_line(x["context"]))
    for x in load_jsonl(MC / "hindi_proverbs.jsonl"):
        add("hi", "hindi_proverbs", x["qid"], "idiom", x["meta"]["expr"])
    for x in load_jsonl(MC / "xiehouyu_zh.jsonl"):
        add("zh", "xiehouyu_zh", x["qid"], "idiom", x["meta"]["expr"])
    import csv
    for i, r in enumerate(csv.DictReader(open(ROOT / "eval_data/hi/mabl_hi.csv", encoding="utf-8"))):
        add("hi", "mabl_hi", f"mabl/{i}", "text", r["startphrase"])
    try:
        from culture.evaluation.tasks_ar import load_ar_figurative
        t = load_ar_figurative(ar_data_dir=str(ROOT / "eval_data/ar/raw"))
        for ex in t.examples:
            add("ar", "ar_figurative", ex.qid, "text", ex.context.split("\nالإجابة:")[0])
    except Exception as e:  # pragma: no cover
        print("AR-Figurative not loaded:", repr(e), file=sys.stderr)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "patterns.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(Counter((it["lang"], it["set"]) for it in items))


# --------------------------------------------------------------------------- scan
_A = None


def _init(L):
    global _A
    import ahocorasick
    keys = set()
    for it in load_jsonl(WORK / "patterns.jsonl"):
        if it["lang"] == L:
            keys.add(it["key"]); keys.update(it["loose"])
    _A = ahocorasick.Automaton()
    for k in keys:
        _A.add_word(k, k)
    _A.make_automaton()


def _scan_file(args):
    L, path = args
    cnt, docs, chars = Counter(), 0, 0
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                t = json.loads(line)["text"]
            except Exception:
                continue
            docs += 1; chars += len(t)
            h = norm(L, t)
            for k in {v for _, v in _A.iter(h)}:
                cnt[k] += 1
    return path, docs, chars, cnt


def scan(L, corpus, part, nparts, procs):
    from huggingface_hub import HfApi, hf_hub_download
    d = CORPUS[corpus].format(L=L)
    files = sorted(f.path for f in HfApi().list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=d)
                   if f.path.endswith((".jsonl", ".jsonl.gz")))
    files = files[part::nparts]
    out = WORK / f"scan_{L}_{corpus}_{part}of{nparts}.json"
    if out.exists():
        print("exists", out); return
    loc = SCRATCH / f"{L}_{corpus}_{part}"
    loc.mkdir(parents=True, exist_ok=True)
    total, tdocs, tchars, done = Counter(), 0, 0, []
    with Pool(procs, initializer=_init, initargs=(L,)) as pool:
        # download in batches of `procs` files, scan them in parallel, delete
        for b in range(0, len(files), procs):
            batch = [hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=str(loc)) for f in files[b:b + procs]]
            for p, n, c, cnt in pool.imap_unordered(_scan_file, [(L, p) for p in batch]):
                total.update(cnt); tdocs += n; tchars += c; done.append(os.path.relpath(p, loc))
                os.remove(p)
            print(f"{L} {corpus} {len(done)}/{len(files)} files, {tdocs} docs", flush=True)
    json.dump({"lang": L, "corpus": corpus, "hf_dir": d, "files": done, "docs": tdocs, "chars": tchars,
               "key_doc_counts": dict(total)}, open(out, "w"), ensure_ascii=False)


# --------------------------------------------------------------------------- report
def report(out):
    import numpy as np
    from culture.analysis.rebuttal import idiomtext_vs_random as M
    pats = load_jsonl(WORK / "patterns.jsonl")
    scans = defaultdict(lambda: {"docs": 0, "chars": 0, "files": 0, "cnt": Counter()})
    for p in glob.glob(str(WORK / "scan_*.json")):
        s = json.load(open(p))
        a = scans[(s["lang"], s["corpus"])]
        a["docs"] += s["docs"]; a["chars"] += s["chars"]; a["files"] += len(s["files"]); a["cnt"].update(s["key_doc_counts"])
    res = {"corpora_scanned": {f"{L}:{c}": {"hf_dir": CORPUS[c].format(L=L), "docs": v["docs"], "chars": v["chars"],
                                            "files": v["files"]} for (L, c), v in scans.items()},
           "sets": {}, "contrasts_after_dropping": {}}
    flags = {}
    for it in pats:
        L = it["lang"]
        r = {}
        for c in CORPUS:
            cnt = scans[(L, c)]["cnt"]
            r[f"{c}_exact_docs"] = cnt.get(it["key"], 0)
            r[f"{c}_loose_docs"] = max([cnt.get(k, 0) for k in it["loose"]] + [r[f"{c}_exact_docs"]])
        flags[(it["set"], it["qid"])] = r
        s = res["sets"].setdefault(it["set"], {"lang": L, "kind": it["kind"], "n": 0, "examples": []})
        s["n"] += 1
        for c in CORPUS:
            for m in ["exact", "loose"]:
                s[f"{c}_{m}_items"] = s.get(f"{c}_{m}_items", 0) + int(r[f"{c}_{m}_docs"] > 0)
        s["any_exact_items"] = s.get("any_exact_items", 0) + int(r["idiom_exact_docs"] + r["random_exact_docs"] > 0)
        s["any_loose_items"] = s.get("any_loose_items", 0) + int(r["idiom_loose_docs"] + r["random_loose_docs"] > 0)
        if r["idiom_exact_docs"] + r["random_exact_docs"] > 0 and len(s["examples"]) < 8:
            s["examples"].append({"raw": it["raw"][:80], **r})
    import statistics
    for set_, s in res["sets"].items():
        for c in CORPUS:
            v = [r[f"{c}_exact_docs"] for (s_, q), r in flags.items() if s_ == set_ and r[f"{c}_exact_docs"] > 0]
            s[f"{c}_exact_docs_median_among_hits"] = statistics.median(v) if v else 0
            s[f"{c}_exact_docs_total"] = sum(v)
        if set_.startswith(("idiomatlas", "hindi", "xiehouyu", "midi")) and ":" not in set_:
            nw = [len(it["key"].split()) for it in pats if it["set"] == set_]
            s["items_with_>=2_words"] = sum(x >= 2 for x in nw)
            s["any_exact_items_with_>=2_words"] = sum(
                1 for it in pats if it["set"] == set_ and len(it["key"].split()) >= 2
                and flags[(set_, it["qid"])]["idiom_exact_docs"] + flags[(set_, it["qid"])]["random_exact_docs"] > 0)
    for s in res["sets"].values():
        for k in list(s):
            if k.endswith("_items"):
                s[k.replace("_items", "_share")] = round(s[k] / max(1, s["n"]), 4)
    # recompute contrasts on items with 9B per-item records, dropping matched items
    task_of = {"idiomatlas_mc_ar_unseen": ("ar", "idiomatlas_mc_ar_unseen"),
               "idiomatlas_mc_hi_unseen": ("hi", "idiomatlas_mc_hi_unseen"),
               "idiomatlas_mc_zh_unseen": ("zh", "idiomatlas_mc_zh_unseen"),
               "midi_ar": ("ar", "midi_ar"), "midi_zh": ("zh", "midi_zh"),
               "hindi_proverbs": ("hi", "hindi_proverbs"), "xiehouyu_zh": ("zh", "xiehouyu_zh")}
    rng = np.random.default_rng(7)
    for set_, (L, task) in task_of.items():
        recs = {r: M.load(L, r, task) for r in ["unfiltered", "untagged", "cpt"]}
        if not all(recs.values()):
            continue
        for rule in ["exact", "loose"]:
            hit = {q for (s_, q), r in flags.items() if s_ in (set_, f"{set_}:sentence")
                   and r[f"idiom_{rule}_docs"] + r[f"random_{rule}_docs"] > 0}
            qids = sorted(set.intersection(*[set(v) for v in recs.values()]) - hit)
            row = {"n_kept": len(qids), "n_dropped": len(hit)}
            if len(qids) < 10:
                res["contrasts_after_dropping"][f"{set_} | drop {rule} hits"] = row
                continue
            acc = {r: np.array([v[q] for q in qids]) for r, v in recs.items()}
            idx = rng.integers(0, len(qids), size=(10000, len(qids)))
            for t, c in [("untagged", "unfiltered"), ("cpt", "unfiltered"), ("cpt", "untagged")]:
                d = acc[t] - acc[c]; bs = d[idx].mean(1); p, w, l = M.mcnemar(acc[t], acc[c])
                row[f"{M.RUN[t]} - {M.RUN[c]}"] = {"delta": round(100 * d.mean(), 1),
                    "ci95": [round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)],
                    "mcnemar_p": round(p, 4)}
            res["contrasts_after_dropping"][f"{set_} | drop {rule} hits"] = row
    res["notes"] = {
        "idiomatlas_unseen_definition": "build_idiomatlas_mc.py: unseen = KB idiom absent from every scanned "
            "document; 'ever' = kept idiom-corpus counts plus (zh) mC4 counts and (ar) pool-scan counts; for hi, "
            "'ever' is only the kept idiom corpus, so the Random-CPT documents were not part of the definition.",
        "idiom_plus_meaning": "Idiom+Meaning trains on the Idiom-Text documents plus meaning blocks; blocks list only "
            "matched KB idioms, so an idiom absent from the Idiom-Text documents is also absent from its blocks.",
        "not_scanned": "Culture-CPT corpora (not in the HF release).",
    }

    def _rare(c):
        try:
            c.encode("gb2312"); return False
        except UnicodeEncodeError:
            return True
    zh_un = [it["raw"] for it in pats if it["set"] == "idiomatlas_mc_zh_unseen"]
    res["notes"]["zh_unseen_rare_char_share"] = {
        "n": len(zh_un), "with_non_gb2312_char": sum(any(_rare(c) for c in s) for s in zh_un),
        "comment": "most Chinese unseen items carry a rare or corrupted character (e.g. 金鴗擘海 for 金翅擘海, "
                   "银样閖枪头 for 银样镴枪头), i.e. they are unseen largely because the KB spelling is non-standard"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    for k, s in res["sets"].items():
        print(f"{k:28s} n={s['n']:5d} idiom-corpus exact {s['idiom_exact_items']:4d} loose {s['idiom_loose_items']:4d} | "
              f"random exact {s['random_exact_items']:4d} loose {s['random_loose_items']:4d}")
    for k, v in res["contrasts_after_dropping"].items():
        print(k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("patterns")
    s = sp.add_parser("scan")
    s.add_argument("--lang"); s.add_argument("--corpus", choices=list(CORPUS))
    s.add_argument("--part", type=int, default=0); s.add_argument("--nparts", type=int, default=1)
    s.add_argument("--procs", type=int, default=8)
    r = sp.add_parser("report"); r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "patterns":
        build_patterns()
    elif a.cmd == "scan":
        scan(a.lang, a.corpus, a.part, a.nparts, a.procs)
    else:
        report(a.out)
