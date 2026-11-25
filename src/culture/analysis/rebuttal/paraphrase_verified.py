#!/usr/bin/env python3
"""Paraphrased IdiomAtlas-MC restricted to rewrites a second model judges meaning-preserving.

The rewrites in paraphrase_mc.py come from Qwen3.5-27B; some drift in meaning (one Arabic option was even
reversed), which adds noise for every arm. Gemma-4-26B (a different model family) judges each
(original option, rewritten option) pair: same / partly / different. Contrasts are then recomputed on the
paraphrased items whose gold option, or all four options, are judged "same". No new scoring of the 9B arms is needed.

  judge : <B>/rebuttal_x2/judge_<name>.jsonl (unique pairs, resumable)
  stats : docs/paper_stats/rebuttal/paraphrase_verified.json
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from culture.analysis.rebuttal.paraphrase_mc import EVAL, EVALP, PARA, RUN, boot, recs

W = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir/rebuttal_x2")
LANG = {"zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
SPLITS = {"ar": ["seen", "unseen"], "hi": ["seen", "unseen"], "zh": ["seen", "unseen"]}

PROMPT = """Two {lang} texts are given. Text B is meant to be a paraphrase of text A. Does B mean the same thing as A?

A: {a}
B: {b}

"same" = B expresses the same meaning as A (wording may differ completely; small stylistic changes are fine);
"partly" = B keeps part of the meaning but adds, drops or shifts something important;
"different" = B means something else, or the opposite.
Return JSON only: {{"label": "same" | "partly" | "different"}}"""


def items(lang, split):
    return [json.loads(l) for l in open(PARA / f"idiomatlas_mc_{lang}_{split}_para.jsonl", encoding="utf-8")]


def all_pairs():
    P = {}
    for L, ss in SPLITS.items():
        for s in ss:
            for x in items(L, s):
                for o, p in zip(x["meta"]["orig_options"], x["options"]):
                    P[(L, o.strip(), p.strip())] = 1
    return list(P)


def parse(t):
    m = re.search(r'"label"\s*:\s*"(same|partly|different)"', t or "")
    if m:
        return m.group(1)
    m = re.search(r"\b(same|partly|different)\b", (t or "").lower())
    return m.group(1) if m else None


def judge(a):
    from vllm import LLM, SamplingParams
    W.mkdir(parents=True, exist_ok=True)
    fn = W / f"judge_{a.name}.jsonl"
    done = {tuple(json.loads(l)["key"]) for l in open(fn, encoding="utf-8")} if fn.exists() else set()
    todo = [k for k in all_pairs() if k not in done]
    random.Random(0).shuffle(todo)
    print("pairs todo", len(todo), flush=True)
    if not todo:
        return
    extra = {"quantization": a.quant} if a.quant else {}
    llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.92,
              max_num_seqs=128, enable_prefix_caching=True, **extra)
    sp = SamplingParams(temperature=0.0, max_tokens=30, seed=0)
    for k in range(0, len(todo), 2000):
        ch = todo[k:k + 2000]
        msgs = [[{"role": "user", "content": PROMPT.format(lang=LANG[L], a=o[:800], b=p[:800])}] for L, o, p in ch]
        try:
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            outs = llm.chat(msgs, sp, use_tqdm=False)
        with open(fn, "a", encoding="utf-8") as f:
            for key, o in zip(ch, outs):
                t = o.outputs[0].text
                f.write(json.dumps({"key": list(key), "label": parse(t), "raw": t[:120]}, ensure_ascii=False) + "\n")
        print("judged", k + len(ch), flush=True)


def stats(a):
    lab = {}
    for name in a.judges:
        fn = W / f"judge_{name}.jsonl"
        if fn.exists():
            lab[name] = {tuple(r["key"]): r["label"] for r in (json.loads(l) for l in open(fn, encoding="utf-8"))}
    rng = np.random.default_rng(12)
    out = {"judges": list(lab), "label_share": {}, "results": {}}
    for name, J in lab.items():
        for L in SPLITS:
            v = [J[k] for k in J if k[0] == L]
            out["label_share"][f"{name}:{L}"] = {c: round(v.count(c) / max(1, len(v)), 3) for c in ["same", "partly", "different", None]}
    for L, ss in SPLITS.items():
        for s in ss:
            xs = items(L, s)
            t = f"idiomatlas_mc_{L}_{s}"
            O = {r: recs(EVAL, L, r, t) for r in RUN}
            Pp = {r: recs(EVALP, L, r, t + "_para") for r in RUN}
            if not all(O.values()) or not all(Pp.values()):
                continue
            for name, J in list(lab.items()) + ([("both", None)] if len(lab) == 2 else []):
                def ok(x, which):
                    pairs = [(L, o.strip(), p.strip()) for o, p in zip(x["meta"]["orig_options"], x["options"])]
                    if which == "gold":
                        pairs = [pairs[x["gold"]]]
                    if J is None:
                        return all(all(l.get(k) == "same" for l in lab.values()) for k in pairs)
                    return all(J.get(k) == "same" for k in pairs)
                for which in ["gold", "all4"]:
                    q = [x["qid"] for x in xs if ok(x, which) and all(x["qid"] in O[r] and x["qid"] in Pp[r] for r in RUN)]
                    row = {"n_kept": len(q), "share_kept": round(len(q) / len(xs), 3)}
                    if len(q) >= 20:
                        row["acc_para"] = {RUN[r]: round(100 * np.mean([Pp[r][i] for i in q]), 1) for r in RUN}
                        row["acc_orig"] = {RUN[r]: round(100 * np.mean([O[r][i] for i in q]), 1) for r in RUN}
                        for x_, y_ in [("cpt", "unfiltered"), ("cpt", "untagged"), ("untagged", "unfiltered")]:
                            dp = np.array([Pp[x_][i] - Pp[y_][i] for i in q], float)
                            do = np.array([O[x_][i] - O[y_][i] for i in q], float)
                            row[f"{RUN[x_]} - {RUN[y_]}"] = {"orig_same_items": boot(do, rng), "para": boot(dp, rng)}
                    out["results"][f"{L}_{s}|{name}|{which}"] = row
                    print(L, s, name, which, json.dumps(row, ensure_ascii=False)[:300], flush=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge", "stats"])
    ap.add_argument("--model"); ap.add_argument("--name"); ap.add_argument("--quant")
    ap.add_argument("--judges", nargs="+", default=["gemma", "qwen27b"])
    ap.add_argument("--out", default="docs/paper_stats/rebuttal/paraphrase_verified.json")
    a = ap.parse_args()
    {"judge": judge, "stats": stats}[a.cmd](a)


if __name__ == "__main__":
    main()
