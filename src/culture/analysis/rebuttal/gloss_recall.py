#!/usr/bin/env python3
"""How much of the seen-idiom gain of Idiom+Meaning is recall of the appended dictionary meaning?

1. blocks  : read the Idiom+Meaning training corpora (HF Jerry9999/CultureInFigurativeLanguage
             data/train_ar, data/train (Hindi), data/train_zh), parse every appended meaning block and
             keep, per idiom, the number of documents whose block lists it and the first block entry text.
2. analyze : for the seen items of IdiomAtlas-MC (ar/hi/zh) and the Arabic external seen benchmarks
             (Jawaher, Absher meaning, Absher cultural interpretation):
               containment = share of the gold option's character 4-grams found in the block entry the
                             model read for that idiom (1.0 = verbatim);
               distractor exposure = whether the idioms whose meanings serve as distractors also had
                             their meanings appended in training (so a "familiar gloss" does not single
                             out the gold);
             then Idiom+Meaning minus Idiom-Text / Random-CPT per containment tertile and per
             distractor-exposure group, paired bootstrap 95% CIs, and the pass-7 option-only control
             (idiom removed from the prompt) restated from docs/paper_stats/v7/swap_stats.json.
"""
import argparse
import glob
import json
import os
import re
import unicodedata
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
MC = ROOT / "eval_data/mc"
WORK = ROOT / "rebuttal_e5"
REPO_ID = "Jerry9999/CultureInFigurativeLanguage"
TAGGED = {"ar": "data/train_ar", "hi": "data/train", "zh": "data/train_zh"}
HEADER = {"ar": "المعاني الاصطلاحية للتعابير الواردة في النص:", "hi": "लोकोक्तियों के अर्थ:", "zh": "【成语注释】"}
SCRATCH = Path(os.environ.get("E5_SCRATCH", "/scratch/jiaruil5/e5"))
REPO = Path(__file__).resolve().parents[4]


def key(L, s):
    s = str(s or "").strip()
    if L == "ar":
        from culture.data_processing.ar_idioms.normalize import normalize_idiom_for_matching
        return normalize_idiom_for_matching(s)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def parse_block(L, text):
    """[(idiom surface, entry text)] of the meaning block appended to one document."""
    i = text.rfind(HEADER[L])
    if i < 0:
        return []
    lines = text[i + len(HEADER[L]):].strip("\n").split("\n")
    out = []
    if L == "ar":
        cur = None
        for ln in lines:
            if ln.startswith("- "):
                cur = [ln[2:].strip(), []]; out.append(cur)
            elif cur is not None and ln.startswith("  "):
                cur[1].append(ln.strip())
        return [(a, "\n".join(b)) for a, b in out]
    sep = " — " if L == "hi" else "："
    for ln in lines:
        if sep in ln:
            a, b = ln.split(sep, 1)
            out.append((a.strip(), b.strip()))
    return out


def _blocks_file(args):
    L, path = args
    seen = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                t = json.loads(line)["text"]
            except Exception:
                continue
            for a, b in parse_block(L, t):
                k = key(L, a)
                if k in seen:
                    seen[k][0] += 1
                    if b != seen[k][1] and len(seen[k]) < 4:
                        seen[k].append(b)
                else:
                    seen[k] = [1, b]
    return path, seen


def blocks(L, part, nparts, procs):
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f.path for f in HfApi().list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=TAGGED[L])
                   if f.path.endswith(".jsonl"))[part::nparts]
    out = WORK / f"blocks_{L}_{part}of{nparts}.json"
    if out.exists():
        return
    loc = SCRATCH / f"{L}_{part}"
    loc.mkdir(parents=True, exist_ok=True)
    agg = {}
    with Pool(procs) as pool:
        for b in range(0, len(files), procs):
            batch = [hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=str(loc)) for f in files[b:b + procs]]
            for p, seen in pool.imap_unordered(_blocks_file, [(L, p) for p in batch]):
                for k, v in seen.items():
                    if k in agg:
                        agg[k][0] += v[0]
                        for e in v[1:]:
                            if e not in agg[k][1:] and len(agg[k]) < 4:
                                agg[k].append(e)
                    else:
                        agg[k] = v
                os.remove(p)
            print(L, b + len(batch), "/", len(files), len(agg), flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump({"lang": L, "files": len(files), "entries": agg}, open(out, "w"), ensure_ascii=False)


# --------------------------------------------------------------------------- analysis
def grams(L, s, n=4):
    s = key(L, s) if L == "ar" else re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()
    s = re.sub(r"[\s\W_]+", " ", s) if L != "zh" else re.sub(r"[\s\W_]+", "", s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))} if s else set()


def containment(L, option, entries):
    g = grams(L, option)
    if not g or not entries:
        return 0.0
    return max(len(g & grams(L, e)) / len(g) for e in entries)


def load_blocks(L):
    agg = {}
    for p in glob.glob(str(WORK / f"blocks_{L}_*.json")):
        for k, v in json.load(open(p))["entries"].items():
            if k in agg:
                agg[k][0] += v[0]; agg[k] += [e for e in v[1:] if e not in agg[k][1:]][:3]
            else:
                agg[k] = v
    if L == "hi":  # blocks name the matched variant; index them under the KB canonical form as well
        for l in open(REPO / "culture/data/hi_idioms/idiom_variants_hi.jsonl", encoding="utf-8"):
            v = json.loads(l)
            c = key(L, v["canonical"])
            for s in v.get("variants", []):
                e = agg.get(key(L, s))
                if e and c not in agg:
                    agg[c] = e
                elif e and agg[c] is not e:
                    agg[c] = [agg[c][0] + e[0]] + (agg[c][1:] + [x for x in e[1:] if x not in agg[c][1:]])[:3]
    return agg


def analyze(out):
    import numpy as np
    from culture.analysis.rebuttal import idiomtext_vs_random as M
    rng = np.random.default_rng(11)
    res = {"sets": {}, "option_only_control": {}}
    sets = [("ar", "idiomatlas_mc_ar_seen", "idiom"), ("hi", "idiomatlas_mc_hi_seen", "idiom"),
            ("zh", "idiomatlas_mc_zh_seen", "idiom"), ("ar", "jawaher_ar", "expr"),
            ("ar", "absher_meaning", "term"), ("ar", "absher_cultural_interpretation", "term"),
            ("ar", "idiomatlas_mc_ar_unseen", "idiom"), ("hi", "idiomatlas_mc_hi_unseen", "idiom")]
    B = {L: load_blocks(L) for L in ["ar", "hi", "zh"]}
    # zh blocks come from a 1-in-6 shard sample; exposure (idiom listed in some block) is taken from the corpus
    # builders' kept-idiom counts, which list every idiom that got a block
    KEPT = {"zh": set()}
    for f in ["fineweb-edu-zh-chengyu-cpt", "mc4-zh-idiom-cpt"]:
        KEPT["zh"] |= {key("zh", k) for k in json.load(open(ROOT / f"hf9b/data/{f}/stats/kept_idiom_counts_zh.json"))}

    def has_block(L, idiom):
        k = key(L, idiom)
        return bool(B[L].get(k)) or k in KEPT.get(L, ())
    item_rows = {}

    def contrast(acc, mask):
        out = {}
        q = np.where(mask)[0]
        if len(q) < 10:
            return {"n": int(len(q))}
        idx = rng.integers(0, len(q), size=(10000, len(q)))
        out["n"] = int(len(q))
        out["acc"] = {M.RUN[r]: round(100 * acc[r][q].mean(), 1) for r in acc}
        for t, c in [("cpt", "untagged"), ("cpt", "unfiltered"), ("untagged", "unfiltered")]:
            if t in acc and c in acc:
                d = (acc[t] - acc[c])[q]; bs = d[idx].mean(1)
                out[f"{M.RUN[t]} - {M.RUN[c]}"] = [round(100 * d.mean(), 1), round(100 * np.percentile(bs, 2.5), 1),
                                                  round(100 * np.percentile(bs, 97.5), 1)]
        return out

    for L, task, fld in sets:
        items = {json.loads(l)["qid"]: json.loads(l) for l in open(MC / f"{task}.jsonl", encoding="utf-8")}
        recs = {r: M.load(L, r, task) for r in ["base", "unfiltered", "untagged", "cpt"]}
        recs = {r: v for r, v in recs.items() if v}
        qids = sorted(set(items) & set.intersection(*[set(v) for v in recs.values()]))
        acc = {r: np.array([v[q] for q in qids]) for r, v in recs.items()}
        cont, exposed, dexp, dcont = [], [], [], []
        for q in qids:
            it = items[q]; m = it["meta"]
            idiom = m.get(fld) or m.get("idiom")
            e = B[L].get(key(L, idiom))
            gold = it["options"][it["gold"]]
            cont.append(containment(L, gold, e[1:]) if e else (np.nan if has_block(L, idiom) else 0.0))
            exposed.append(int(has_block(L, idiom)))
            di = m.get("distractor_idioms")
            if di:
                de = [B[L].get(key(L, d)) for d in di]
                dexp.append(sum(has_block(L, d) for d in di))
                others = [o for k, o in enumerate(it["options"]) if k != it["gold"]]
                # options are shuffled; distractor text i belongs to distractor idiom i in build order
                dcont.append(float(np.mean([max(containment(L, o, x[1:]) for o in others) for x in de if x])) if any(de) else np.nan)
            item_rows[(task, q)] = {"containment": round(cont[-1], 3), "idiom_in_blocks": exposed[-1],
                                    "block_docs": e[0] if e else 0}
        cont = np.array(cont, float); exposed = np.array(exposed)
        cv = cont[~np.isnan(cont)]
        row = {"lang": L, "n": len(qids), "idiom_has_block_share": round(float(exposed.mean()), 3),
               "gold_containment": {"n_with_block_text": int(len(cv)), "mean": round(float(cv.mean()), 3),
                                    "median": round(float(np.median(cv)), 3),
                                    "share_ge_0.9": round(float((cv >= 0.9).mean()), 3),
                                    "share_lt_0.5": round(float((cv < 0.5).mean()), 3)},
               "all": contrast(acc, np.ones(len(qids), bool))}
        # tertiles of containment (ties broken by rank) among items whose idiom had a block
        cont = np.nan_to_num(cont, nan=-1.0)   # -1 = block exists but its text was not in the zh sample
        if exposed.sum() >= 30:
            order = np.argsort(cont + 1e-9 * np.arange(len(cont)))
            ranks = np.empty(len(cont)); ranks[order] = np.arange(len(cont))
            t = np.floor(3 * ranks / len(cont)).astype(int)
            row["by_containment_tertile"] = {}
            for k in range(3):
                mk = t == k
                row["by_containment_tertile"][f"T{k + 1}"] = {"containment_range": [round(float(cont[mk].min()), 3), round(float(cont[mk].max()), 3)],
                                                             **contrast(acc, mk)}
            row["containment_lt_0.5"] = contrast(acc, cont < 0.5)
            row["containment_ge_0.9"] = contrast(acc, cont >= 0.9)
            row["idiom_without_block"] = contrast(acc, exposed == 0)
            row["block_but_gold_not_verbatim(lt_0.5)"] = contrast(acc, (exposed == 1) & (cont >= 0) & (cont < 0.5))
            row["block_and_gold_verbatim(ge_0.9)"] = contrast(acc, (exposed == 1) & (cont >= 0.9))
        # option-only version of the same items (idiom removed), pass-7 records
        opt = {}
        for r in ["unfiltered", "untagged", "cpt"]:
            p = ROOT / "eval9b_v7" / L / r / f"{task}_optonly.json"
            if p.exists():
                j = json.load(open(p)); mode = j.get("metrics", {}).get("score_mode", "continuation")
                kk = "correct_norm" if mode == "continuation" else "correct"
                d = {x["qid"]: int(x[kk]) for x in j["records"]}
                if all(q in d for q in qids):
                    opt[r] = np.array([d[q] for q in qids])

        def opt_contrast(mask):
            q = np.where(mask)[0]
            if len(opt) < 2 or "cpt" not in opt or len(q) < 10:
                return None
            idx = rng.integers(0, len(q), size=(10000, len(q)))
            o = {}
            for t, c in [("cpt", "unfiltered"), ("cpt", "untagged")]:
                if t in opt and c in opt:
                    full = (acc[t] - acc[c])[q]; oo = (opt[t] - opt[c])[q]
                    bo, bn = oo[idx].mean(1), (full - oo)[idx].mean(1)
                    o[f"{M.RUN[t]} - {M.RUN[c]}"] = {
                        "gain_full": round(100 * full.mean(), 1),
                        "gain_option_only": [round(100 * oo.mean(), 1), round(100 * np.percentile(bo, 2.5), 1), round(100 * np.percentile(bo, 97.5), 1)],
                        "gain_needing_idiom": [round(100 * (full - oo).mean(), 1), round(100 * np.percentile(bn, 2.5), 1), round(100 * np.percentile(bn, 97.5), 1)]}
            return o
        if opt:
            row["option_only_all"] = opt_contrast(np.ones(len(qids), bool))
        if dexp:
            dexp = np.array(dexp)
            if opt:
                row["option_only_by_distractor_exposure"] = {"0_of_3_glossed": opt_contrast(dexp == 0),
                                                             ">=1_of_3_glossed": opt_contrast(dexp >= 1)}
            row["distractor_idioms_with_block"] = {"mean_of_3": round(float(dexp.mean()), 2),
                                                   "share_items_all3": round(float((dexp == 3).mean()), 3),
                                                   "share_items_none": round(float((dexp == 0).mean()), 3),
                                                   "distractor_text_containment_mean": round(float(np.nanmean(dcont)), 3)}
            row["by_distractor_exposure"] = {"0_of_3_glossed": contrast(acc, dexp == 0),
                                             "1-2_of_3_glossed": contrast(acc, (dexp >= 1) & (dexp <= 2)),
                                             "3_of_3_glossed": contrast(acc, dexp == 3),
                                             ">=2_of_3_glossed": contrast(acc, dexp >= 2)}
        res["sets"][task] = row
        print(task, json.dumps({k: v for k, v in row.items() if k in ("n", "idiom_has_block_share", "gold_containment",
                                                                       "distractor_idioms_with_block")}, ensure_ascii=False))
    # Kinayat-Meaning (no 9B records on babel): the eval loader's own verbatim flag, 2B records
    try:
        r = json.load(open(ROOT / "eval2b/ar/i_random/kinayat_meaning.json"))["records"]
        res["kinayat_meaning_answer_string_in_training_share"] = round(sum(bool(x.get("answer_string_in_training")) for x in r) / len(r), 3)
    except Exception:
        pass
    # option-only control (pass 7)
    sw = json.load(open(REPO / "docs/paper_stats/v7/swap_stats.json"))
    for task, v in sw.items():
        if not isinstance(v, dict):
            continue
        for arm, a in v.items():
            if isinstance(a, dict) and "acc" in a and "gain_optonly" in a["acc"]:
                g = a["acc"]
                res["option_only_control"][f"{task} | {M.RUN.get(arm, arm)}"] = {
                    k: {"delta": round(100 * g[k]["delta"], 1), "ci95": [round(100 * c, 1) for c in g[k]["ci"]], "n": g[k]["n"]}
                    for k in ["gain_original", "gain_optonly", "needs_expression"] if k in g}
    res["option_only_control_source"] = "docs/paper_stats/v7/swap_stats.json (src/culture/analysis/v7/swap_stats.py); gains vs Random-CPT; optonly = idiom removed from the prompt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("blocks"); s.add_argument("--lang"); s.add_argument("--part", type=int, default=0)
    s.add_argument("--nparts", type=int, default=1); s.add_argument("--procs", type=int, default=8)
    r = sp.add_parser("analyze"); r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "blocks":
        blocks(a.lang, a.part, a.nparts, a.procs)
    else:
        analyze(a.out)
