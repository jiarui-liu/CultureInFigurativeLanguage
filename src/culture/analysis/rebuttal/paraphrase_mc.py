#!/usr/bin/env python3
"""IdiomAtlas-MC with every option paraphrased.

If the Idiom+Meaning gain on seen idioms only reflects recall of the dictionary text appended during
training, it should shrink when no option repeats that text. An LLM rewrites EVERY option (gold and
distractors alike) in the same language, keeping the meaning but changing the wording; gold index and
option order are unchanged.

  generate : rewrite all unique option strings (Qwen3.5-27B-FP8, vLLM; temperature 0, then up to two
             retries at 0.7 with a stricter prompt when the rewrite stays too close to the original by
             chrF); writes <mc_para>/idiomatlas_mc_{L}_{split}_para.jsonl and a log of all attempts
  stats    : original vs paraphrased contrasts from per-item records (paired bootstrap)
  export   : 30 random items per language, original vs paraphrase, for a native-speaker spot check
"""
import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

B = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
MC = B / "eval_data/mc"
PARA = B / "eval_data/mc_para"
EVAL, EVALP = B / "eval9b", B / "eval9b_para"
LANG = {"zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
RUN = {"base": "Base", "unfiltered": "Random-CPT", "untagged": "Idiom-Text", "cpt": "Idiom+Meaning"}
MAX_CHRF = 45.0  # retry when the rewrite is closer than this to the original (chrF, 0-100)

P1 = """Rewrite the following {lang} text so that it means exactly the same thing but uses different words and a different sentence structure. Keep it in {lang}, keep roughly the same length, do not add or remove information, and do not explain anything.

Text: {text}

Rewritten text:"""
P2 = """Paraphrase the {lang} text below. Requirements: same meaning, same language ({lang}), similar length, no added or removed information. Avoid reusing these words or phrases from the original wherever a synonym exists: {words}. Output only the paraphrase.

Text: {text}

Paraphrase:"""


def chrf(hyp, ref, n=6, beta=2.0):
    """Character n-gram F-score (chrF), whitespace removed, 0-100."""
    h, r = re.sub(r"\s+", "", hyp), re.sub(r"\s+", "", ref)
    ps, rs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        m = sum((hc & rc).values())
        ps.append(m / sum(hc.values())); rs.append(m / sum(rc.values()))
    if not ps:
        return 0.0
    p, rr = np.mean(ps), np.mean(rs)
    return 0.0 if p + rr == 0 else 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


def units(lang, s):
    if lang == "zh":
        return {c for c in s if "一" <= c <= "鿿"}
    if lang == "ar":
        from culture.data_processing.ar_idioms.normalize import normalize_ar
        s = normalize_ar(s)
    return {w for w in re.findall(r"\w+", s) if len(w) >= 2}


def clean(t):
    t = (t or "").strip().split("\n")[0].strip().strip('"“”«»「」').strip()
    return re.sub(r"^(Rewritten text|Paraphrase)\s*[:：]\s*", "", t)


def load(lang, split):
    return [json.loads(l) for l in open(MC / f"idiomatlas_mc_{lang}_{split}.jsonl", encoding="utf-8")]


def generate(a):
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.92,
              max_num_seqs=128, enable_prefix_caching=True)
    PARA.mkdir(parents=True, exist_ok=True)
    log = open(PARA / f"paraphrase_log_{'_'.join(a.langs)}.jsonl", "w", encoding="utf-8")
    for lang in a.langs:
        items = {s: load(lang, s) for s in ["seen", "unseen"]}
        texts = sorted({o.strip() for s in items for x in items[s] for o in x["options"]})
        best = {t: None for t in texts}
        todo = list(texts)
        for att in range(3):
            if not todo:
                break
            if att == 0:
                msgs = [[{"role": "user", "content": P1.format(lang=LANG[lang], text=t)}] for t in todo]
                sp = SamplingParams(temperature=0.0, max_tokens=300, seed=0)
            else:
                def words(t):
                    w = sorted(units(lang, t), key=len, reverse=True)[:12]
                    return ", ".join(w) if lang != "zh" else "、".join(w)
                msgs = [[{"role": "user", "content": P2.format(lang=LANG[lang], text=t, words=words(t))}] for t in todo]
                sp = SamplingParams(temperature=0.7, max_tokens=300, seed=att)
            try:
                outs = llm.chat(msgs, sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
            except TypeError:
                outs = llm.chat(msgs, sp, use_tqdm=True)
            nxt = []
            for t, o in zip(todo, outs):
                p = clean(o.outputs[0].text)
                c = chrf(p, t)
                ok_len = 0.4 <= len(p) / max(1, len(t)) <= 2.5 and len(p) >= 2
                log.write(json.dumps({"lang": lang, "orig": t, "attempt": att, "para": p, "chrf": round(c, 1),
                                      "len_ok": ok_len}, ensure_ascii=False) + "\n")
                if ok_len and (best[t] is None or c < best[t][1]):
                    best[t] = (p, c)
                if best[t] is None or best[t][1] > MAX_CHRF:
                    nxt.append(t)
            todo = nxt
            print(lang, "attempt", att, "still close/invalid", len(todo), flush=True)
        n_fallback = 0
        for split, xs in items.items():
            with open(PARA / f"idiomatlas_mc_{lang}_{split}_para.jsonl", "w", encoding="utf-8") as f:
                for x in xs:
                    opts = []
                    for o in x["options"]:
                        b = best[o.strip()]
                        if b is None:
                            n_fallback += 1
                            opts.append(o)
                        else:
                            opts.append(" " + b[0])
                    ov = [len(units(lang, x["meta"]["idiom"]) & units(lang, o)) / max(1, len(units(lang, x["meta"]["idiom"])))
                          for o in opts]
                    y = dict(x, qid=x["qid"], options=opts,
                             meta={**x["meta"], "orig_options": x["options"],
                                   "para_chrf": [round(best[o.strip()][1], 1) if best[o.strip()] else 100.0 for o in x["options"]],
                                   "idiom_overlap": [round(v, 2) for v in ov]})
                    f.write(json.dumps(y, ensure_ascii=False) + "\n")
        cs = [v[1] for v in best.values() if v]
        print(lang, "unique options", len(texts), "median chrF", round(float(np.median(cs)), 1),
              "share>MAX", round(float(np.mean([c > MAX_CHRF for c in cs])), 3), "fallback", n_fallback, flush=True)
    log.close()


def recs(root, lang, run, task):
    p = root / lang / run / f"{task}.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    mode = j.get("metrics", {}).get("score_mode", "continuation")
    k = "correct_norm" if mode == "continuation" else "correct"
    return {x["qid"]: int(x[k]) for x in j["records"]}


def boot(d, rng, B=10000):
    idx = rng.integers(0, len(d), size=(B, len(d))); bs = d[idx].mean(1)
    return [round(100 * d.mean(), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


def stats(a):
    rng = np.random.default_rng(11)
    out = {}
    for lang in ["ar", "hi", "zh"]:
        for split in ["seen", "unseen"]:
            t = f"idiomatlas_mc_{lang}_{split}"
            O = {r: recs(EVAL, lang, r, t) for r in RUN}
            P = {r: recs(EVALP, lang, r, t + "_para") for r in RUN}
            O = {r: v for r, v in O.items() if v}; P = {r: v for r, v in P.items() if v}
            if "unfiltered" not in P:
                continue
            q = sorted(set.intersection(*[set(v) for v in list(O.values()) + list(P.values())]))
            row = {"n": len(q), "acc_orig": {RUN[r]: round(100 * np.mean([O[r][i] for i in q]), 1) for r in O},
                   "acc_para": {RUN[r]: round(100 * np.mean([P[r][i] for i in q]), 1) for r in P}}
            for x, y in [("cpt", "unfiltered"), ("cpt", "untagged"), ("untagged", "unfiltered"), ("cpt", "base")]:
                if x in P and y in P and x in O and y in O:
                    do = np.array([O[x][i] - O[y][i] for i in q], float)
                    dp = np.array([P[x][i] - P[y][i] for i in q], float)
                    row[f"{RUN[x]} - {RUN[y]}"] = {"orig": boot(do, rng), "para": boot(dp, rng),
                                                   "para_minus_orig": boot(dp - do, rng),
                                                   "retained_share": round(float(dp.mean() / do.mean()), 3) if do.mean() else None}
            out[f"{lang}_{split}"] = row
            print(lang, split, json.dumps(row, ensure_ascii=False))
    lg = [json.loads(l) for fn in sorted(PARA.glob("paraphrase_log_*.jsonl")) for l in open(fn, encoding="utf-8")]
    best = {}
    for r in lg:
        k = (r["lang"], r["orig"])
        if r["len_ok"] and (k not in best or r["chrf"] < best[k]):
            best[k] = r["chrf"]
    out["paraphrase_chrf"] = {L: {"n": len(v), "median": round(float(np.median(v)), 1), "share_gt_45": round(float(np.mean(np.array(v) > 45)), 3)}
                              for L in ["ar", "hi", "zh"] for v in [[c for (l, _), c in best.items() if l == L]] if v}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


def export(a):
    rng = random.Random(5)
    for lang in ["ar", "hi", "zh"]:
        xs = []
        for split in ["seen", "unseen"]:
            p = PARA / f"idiomatlas_mc_{lang}_{split}_para.jsonl"
            if p.exists():
                xs += [json.loads(l) for l in open(p, encoding="utf-8")]
        pick = rng.sample(xs, min(30, len(xs)))
        with open(Path(a.out_dir) / f"paraphrase_check_{lang}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["row", "idiom", "option", "original", "paraphrase", "chrF",
                        "same_meaning (yes/partly/no)", "natural_language (yes/no)", "notes"])
            for i, x in enumerate(pick, 1):
                for k, (o, p, c) in enumerate(zip(x["meta"]["orig_options"], x["options"], x["meta"]["para_chrf"])):
                    w.writerow([i, x["meta"]["idiom"] if k == 0 else "", "ABCD"[k], o.strip(), p.strip(), c, "", "", ""])
        print("wrote", lang, len(pick))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "stats", "export"])
    ap.add_argument("--model")
    ap.add_argument("--langs", nargs="+", default=["ar", "hi", "zh"])
    ap.add_argument("--out", default="docs/paper_stats/rebuttal/paraphrase_mc.json")
    ap.add_argument("--out_dir", default="docs/arr_oct_rebuttal/annotation")
    a = ap.parse_args()
    {"generate": generate, "stats": stats, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
