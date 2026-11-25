#!/usr/bin/env python3
"""Free-generation meaning test for the four 9B arms (no answer options, so no option wording to match).

Each model sees a fixed 3-shot prompt and writes the figurative meaning of an idiom in one English sentence
(greedy, <= 64 new tokens, stops at newline). Two LLM judges, blind to the arm, grade each explanation against
the dictionary meaning: correct / partial / wrong.

Items: the idioms of IdiomAtlas-MC seen/unseen (zh, hi, ar) and the Hindi proverbs set (meta.clean marks the 490
proverbs that occur in no training document, from the contamination scan in unseen_contamination.py).

  prepare  : <W>/items.jsonl
  generate : <W>/gen_<lang>_<run>.jsonl          (vLLM, one model per call)
  judge    : <W>/judge_<name>.jsonl             (unique (item, explanation) pairs, resumable)
  stats    : accuracy per arm, paired-bootstrap contrasts with McNemar and Holm, pooled unseen contrast
"""
import argparse
import ast
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

B = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
MC = B / "eval_data/mc"
W = B / "rebuttal_x1"
REPO = Path(__file__).resolve().parents[4]
KB = {
    "zh": REPO / "culture/data/idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl",
    "hi": REPO / "culture/data/idioms/hi/idioms_merged_llm_formatted_figurative_only.jsonl",
    "ar": B / "hf9b/data/idioms/ar/idioms_merged_llm_formatted.jsonl",
}
LANG = {"zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
RUN = {"base": "Base", "unfiltered": "Random-CPT", "untagged": "Idiom-Text", "cpt": "Idiom+Meaning"}
SETS = {"zh": ["seen", "unseen"], "hi": ["seen", "unseen", "proverbs"], "ar": ["seen", "unseen"]}

# fixed shots; none of these idioms occurs in any test set used here
SHOTS = {
    "zh": [("对牛弹琴", "To speak to someone who cannot understand or appreciate what is said; to address the wrong audience."),
           ("井底之蛙", "A person with a very narrow outlook who knows little of the wider world."),
           ("杯弓蛇影", "To be frightened by imaginary dangers; needless suspicion and fear.")],
    "hi": [("नाच न जाने आँगन टेढ़ा", "Blaming circumstances or tools for one's own lack of skill."),
           ("उल्टा चोर कोतवाल को डाँटे", "The wrongdoer accusing the innocent person who caught him."),
           ("थोथा चना बाजे घना", "Those with little substance or knowledge make the most noise.")],
    "ar": [("رجع بخفي حنين", "He came back empty-handed, having failed to get what he went for."),
           ("يد واحدة لا تصفق", "One person alone cannot achieve much; cooperation is needed."),
           ("الوقت كالسيف إن لم تقطعه قطعك", "Time must be used well, or it will work against you.")],
}
HEAD = "Explain the figurative meaning of each {lang} idiom or proverb in one short English sentence.\n\n"


def prompt(lang, idiom):
    s = HEAD.format(lang=LANG[lang])
    for i, m in SHOTS[lang]:
        s += f"{LANG[lang]} idiom: {i}\nMeaning: {m}\n\n"
    return s + f"{LANG[lang]} idiom: {idiom}\nMeaning:"


def clean_meaning(t):
    t = str(t).strip()
    if t.startswith("[") and t.endswith("]"):
        try:
            v = ast.literal_eval(t)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    return [t]


def kb_meanings(lang):
    out = {}
    for l in open(KB[lang], encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        o = r["output"] if isinstance(r["output"], dict) else ast.literal_eval(r["output"])
        fig = o.get("figurative_meanings") or []
        fig = fig if isinstance(fig, list) else [fig]
        k = str(o.get("idiom") or r.get("idiom")).strip()
        out.setdefault(k, [str(x)[:400] for x in fig if str(x).strip() and str(x) != "NAN"][:3])
    return out


def clean_hindi_proverbs():
    cnt = defaultdict(Counter)
    for p in glob.glob(str(B / "rebuttal_e4/scan_hi_*.json")):
        s = json.load(open(p))
        cnt[s["corpus"]].update(s["key_doc_counts"])
    clean = set()
    for l in open(B / "rebuttal_e4/patterns.jsonl", encoding="utf-8"):
        it = json.loads(l)
        if it["set"] == "hindi_proverbs" and all(c.get(it["key"], 0) == 0 for c in cnt.values()):
            clean.add(it["qid"])
    return clean


def prepare(a):
    W.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(W / "items.jsonl", "w", encoding="utf-8") as f:
        for L, sets in SETS.items():
            kb = kb_meanings(L)
            for s in sets:
                if s == "proverbs":
                    clean = clean_hindi_proverbs()
                    xs = [json.loads(l) for l in open(MC / "hindi_proverbs.jsonl", encoding="utf-8")]
                    for x in xs:
                        f.write(json.dumps({"id": x["qid"], "lang": L, "set": "hindi_proverbs", "idiom": x["meta"]["expr"],
                                            "ref": [x["options"][x["gold"]].strip()], "clean": x["qid"] in clean},
                                           ensure_ascii=False) + "\n"); n += 1
                    continue
                for x in (json.loads(l) for l in open(MC / f"idiomatlas_mc_{L}_{s}.jsonl", encoding="utf-8")):
                    idiom = x["meta"]["idiom"]
                    ref = clean_meaning(x["options"][x["gold"]])
                    for m in kb.get(idiom, []):
                        for c in clean_meaning(m):
                            if c not in ref:
                                ref.append(c)
                    f.write(json.dumps({"id": x["qid"], "lang": L, "set": f"idiomatlas_{s}", "idiom": idiom,
                                        "ref": ref[:4]}, ensure_ascii=False) + "\n"); n += 1
    print("items", n)


def generate(a):
    from vllm import LLM, SamplingParams
    items = [json.loads(l) for l in open(W / "items.jsonl", encoding="utf-8")]
    llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=2048, gpu_memory_utilization=0.9,
              max_num_seqs=128, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=64, stop=["\n"], seed=0)
    for L in a.langs:
        xs = [x for x in items if x["lang"] == L]
        outs = llm.generate([prompt(L, x["idiom"]) for x in xs], sp, use_tqdm=False)
        with open(W / f"gen_{L}_{a.run}.jsonl", "w", encoding="utf-8") as f:
            for x, o in zip(xs, outs):
                f.write(json.dumps({"id": x["id"], "run": a.run, "gen": o.outputs[0].text.strip()}, ensure_ascii=False) + "\n")
        print(L, a.run, len(xs), "examples:", [o.outputs[0].text.strip()[:80] for o in outs[:3]], flush=True)


JUDGE = """You are grading whether a short English explanation gives the figurative meaning of a {lang} idiom or proverb. Use the dictionary meaning(s) as the reference and your own knowledge of {lang}.

Idiom: {idiom}
Dictionary meaning(s): {ref}
Explanation to grade: {gen}

Grades:
"correct" = the explanation conveys the figurative meaning (any one of the listed senses is enough; wording may differ completely);
"partial" = points in the right direction but is clearly vague, incomplete or partly wrong;
"wrong" = a different meaning, only a literal paraphrase of the words, empty, or nonsense.
Return JSON only: {{"grade": "correct" | "partial" | "wrong"}}"""


def parse_grade(t):
    m = re.search(r'"grade"\s*:\s*"(correct|partial|wrong)"', t or "")
    if m:
        return m.group(1)
    m = re.search(r"\b(correct|partial|wrong)\b", (t or "").lower())
    return m.group(1) if m else None


def pairs():
    items = {x["id"]: x for x in (json.loads(l) for l in open(W / "items.jsonl", encoding="utf-8"))}
    out = {}
    for p in sorted(W.glob("gen_*.jsonl")):
        for l in open(p, encoding="utf-8"):
            g = json.loads(l)
            out[(g["id"], g["gen"])] = items[g["id"]]
    return items, out


def run_judge(a, todo, fmt, fn):
    """todo: list of (key, payload); fmt(payload) -> prompt. Appends {"key", "grade", "raw"} to fn."""
    from vllm import LLM, SamplingParams
    extra = {"quantization": a.quant} if a.quant else {}
    llm = LLM(model=a.model, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.92,
              max_num_seqs=128, enable_prefix_caching=True, **extra)
    sp = SamplingParams(temperature=0.0, max_tokens=40, seed=0)
    for k in range(0, len(todo), 2000):
        ch = todo[k:k + 2000]
        msgs = [[{"role": "user", "content": fmt(p)}] for _, p in ch]
        try:
            outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            outs = llm.chat(msgs, sp, use_tqdm=False)
        with open(fn, "a", encoding="utf-8") as f:
            for (key, _), o in zip(ch, outs):
                t = o.outputs[0].text
                f.write(json.dumps({"key": key, "grade": parse_grade(t), "raw": t[:200]}, ensure_ascii=False) + "\n")
        print("judged", k + len(ch), "/", len(todo), flush=True)


def judge(a):
    _, P = pairs()
    fn = W / f"judge_{a.name}.jsonl"
    done = {tuple(json.loads(l)["key"]) for l in open(fn, encoding="utf-8")} if fn.exists() else set()
    todo = [(list(k), (k, it)) for k, it in P.items() if k not in done]
    random.Random(0).shuffle(todo)  # arms interleaved; the judge never sees which arm wrote what
    print("pairs", len(P), "todo", len(todo), flush=True)

    def fmt(p):
        (iid, gen), it = p
        return JUDGE.format(lang=LANG[it["lang"]], idiom=it["idiom"],
                            ref=json.dumps(it["ref"], ensure_ascii=False), gen=gen if gen else "(empty)")
    if todo:
        run_judge(a, todo, fmt, fn)


def boot(d, rng, n=10000):
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return round(100 * d.mean(), 1), [round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]


def stats(a):
    from culture.analysis.rebuttal.idiomtext_vs_random import holm, mcnemar
    items, _ = pairs()
    J = {}
    for name in a.judges:
        fn = W / f"judge_{name}.jsonl"
        if fn.exists():
            J[name] = {tuple(r["key"]): r["grade"] for r in (json.loads(l) for l in open(fn, encoding="utf-8"))}
    gens = defaultdict(dict)  # run -> id -> gen
    for p in W.glob("gen_*.jsonl"):
        for l in open(p, encoding="utf-8"):
            g = json.loads(l)
            gens[g["run"]][g["id"]] = g["gen"]
    rng = np.random.default_rng(20261125)
    res = {"n_items": len(items), "judges": list(J), "shots": SHOTS, "results": {}}
    subsets = []
    for L, sets in SETS.items():
        for s in sets:
            st = "hindi_proverbs" if s == "proverbs" else f"idiomatlas_{s}"
            subsets.append((L, st, st, lambda x: True))
            if s == "proverbs":
                subsets.append((L, st, "hindi_proverbs_clean", lambda x: x.get("clean")))
    for name, grades in J.items():
        for level, ok in [("correct", {"correct"}), ("correct_or_partial", {"correct", "partial"})]:
            tab = {}
            for L, st, label, keep in subsets:
                ids = [i for i, x in items.items() if x["lang"] == L and x["set"] == st and keep(x)
                       and all(i in gens[r] and (i, gens[r][i]) in grades for r in RUN)]
                if not ids:
                    continue
                acc = {r: np.array([int(grades[(i, gens[r][i])] in ok) for i in ids]) for r in RUN}
                row = {"n": len(ids), "acc": {RUN[r]: round(100 * v.mean(), 1) for r, v in acc.items()}}
                for x, y in [("cpt", "unfiltered"), ("cpt", "untagged"), ("untagged", "unfiltered"), ("untagged", "base"), ("cpt", "base")]:
                    d, ci = boot((acc[x] - acc[y]).astype(float), rng)
                    row[f"{RUN[x]} - {RUN[y]}"] = {"delta": d, "ci95": ci, "mcnemar_p": round(mcnemar(acc[x], acc[y])[0], 4)}
                tab[f"{L}:{label}"] = row
            # Holm within each contrast over the per-language tests of one family
            for fam in ["idiomatlas_seen", "idiomatlas_unseen"]:
                for c in ["Idiom+Meaning - Random-CPT", "Idiom+Meaning - Idiom-Text", "Idiom-Text - Random-CPT"]:
                    ks = [k for k in tab if k.endswith(":" + fam)]
                    adj = holm([tab[k][c]["mcnemar_p"] for k in ks])
                    for k, p in zip(ks, adj):
                        tab[k][c]["holm_p"] = round(p, 4)
            # pooled unseen: language-stratified bootstrap over the three IdiomAtlas unseen splits + clean Hindi proverbs
            for pool_name, parts in [("pooled_idiomatlas_unseen", [f"{L}:idiomatlas_unseen" for L in SETS]),
                                     ("pooled_unseen_with_clean_hindi_proverbs",
                                      [f"{L}:idiomatlas_unseen" for L in SETS] + ["hi:hindi_proverbs_clean"])]:
                prow = {}
                for x, y in [("untagged", "unfiltered"), ("cpt", "unfiltered"), ("cpt", "untagged")]:
                    ds = []
                    for k in parts:
                        L, lab = k.split(":")
                        st = "hindi_proverbs" if lab.startswith("hindi") else lab
                        ids = [i for i, it in items.items() if it["lang"] == L and it["set"] == st
                               and (lab != "hindi_proverbs_clean" or it.get("clean"))
                               and all(i in gens[r] and (i, gens[r][i]) in grades for r in RUN)]
                        ds.append(np.array([int(grades[(i, gens[x][i])] in ok) - int(grades[(i, gens[y][i])] in ok) for i in ids], float))
                    est = np.mean([d.mean() for d in ds])  # equal weight per test
                    bs = np.mean([d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1) for d in ds], axis=0)
                    prow[f"{RUN[x]} - {RUN[y]}"] = {"delta": round(100 * est, 1),
                                                    "ci95": [round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)],
                                                    "tests": parts}
                tab[pool_name] = prow
            res["results"][f"{name}:{level}"] = tab
    if len(J) == 2:
        a_, b_ = list(J.values())
        common = [k for k in a_ if k in b_ and a_[k] and b_[k]]
        labs = ["correct", "partial", "wrong"]
        cm = Counter((a_[k], b_[k]) for k in common)
        po = sum(cm[(l, l)] for l in labs) / len(common)
        pa = Counter(a_[k] for k in common); pb = Counter(b_[k] for k in common)
        pe = sum(pa[l] * pb[l] for l in labs) / len(common) ** 2
        bin_agree = np.mean([(a_[k] == "correct") == (b_[k] == "correct") for k in common])
        res["judge_agreement"] = {"n": len(common), "raw": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3),
                                  "binary_correct_raw": round(float(bin_agree), 3)}
    for r in RUN:
        res.setdefault("examples", {})[RUN[r]] = {L: [(items[i]["idiom"], gens[r].get(i, "")) for i in
                                                     [x for x in items if items[x]["lang"] == L][:3]] for L in SETS}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    for k, t in res["results"].items():
        print("==", k)
        for s, row in t.items():
            print(" ", s, json.dumps(row, ensure_ascii=False)[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "generate", "judge", "stats"])
    ap.add_argument("--model"); ap.add_argument("--run"); ap.add_argument("--name"); ap.add_argument("--quant")
    ap.add_argument("--langs", nargs="+", default=["zh", "hi", "ar"])
    ap.add_argument("--judges", nargs="+", default=["gemma", "qwen27b"])
    ap.add_argument("--out", default="docs/paper_stats/rebuttal/free_gen.json")
    a = ap.parse_args()
    {"prepare": prepare, "generate": generate, "judge": judge, "stats": stats}[a.cmd](a)


if __name__ == "__main__":
    main()
