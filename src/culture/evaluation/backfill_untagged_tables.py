#!/usr/bin/env python3
"""Backfill the pending `$-$` Untagged cells in the paper's main tables once the
untagged backfill evals finish. Idempotent: only replaces a field that is still `$-$`.
Untagged is the 4th field (index 3) in tab:hindi and tab:chinese rows."""
import json, os, re

TEX="/storage/home/jiaruiliu/local/git-repos/OverleafCultureInFigurativeLanguage/colm2026_conference.tex"
if not os.path.exists(TEX):
    TEX="/storage/home/jiaruiliu/local/git-repos/culture-pretraining/OverleafCultureInFigurativeLanguage/colm2026_conference.tex"
EV="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval"

def summ(path):
    try: return json.load(open(path)).get("tasks",{})
    except Exception: return {}
def ppl(path,key="ppl"):
    try: return json.load(open(path)).get(key)
    except Exception: return None

def pct(x): return None if x is None else f"{x*100:.1f}"
def one(x): return None if x is None else f"{x:.1f}"
def three(x): return None if x is None else f"{x:.3f}"

# label-prefix -> value string (untagged), col index 3
hi=summ(f"{EV}/hi/untagged/en_gen/summary.json")
fills = {
 ("GSM8K (8-shot CoT, EM)",):        pct(hi.get("gsm8k",{}).get("primary")),
 ("HumanEval (0-shot, pass@1)",):    pct(hi.get("humaneval",{}).get("primary")),
}
zen=summ(f"{EV}/zh/untagged/en/summary.json"); zg=summ(f"{EV}/zh/untagged/en_gen/summary.json")
fills.update({
 ("MMLU (0-shot, acc \\%)",):            pct(zen.get("mmlu",{}).get("primary")),
 ("BoolQ (0-shot, acc \\%)",):           pct(zen.get("boolq",{}).get("primary")),
 ("GSM8K (8-shot CoT, EM \\%)",):        pct(zg.get("gsm8k",{}).get("primary")),
 ("HumanEval (0-shot, pass@1 \\%)",):    pct(zg.get("humaneval",{}).get("primary")),
 ("WikiText-103 (PPL, $\\downarrow$)",): one(ppl(f"{EV}/zh/untagged/ppl_wikitext/perplexity.json")),
 ("zh-Wikipedia$^\\ddagger$ (PPL)",):    one(ppl(f"{EV}/zh/untagged/ppl_zh_wiki/perplexity.json")),
 ("zh-chengyu, in-domain$^\\dagger$ (PPL)",): one(ppl(f"{EV}/zh/untagged/ppl_zh_chengyu/perplexity.json")),
 ("zh-chengyu, in-domain$^\\dagger$ (BPB)",): three(ppl(f"{EV}/zh/untagged/ppl_zh_chengyu/perplexity.json","bits_per_byte")),
})

t=open(TEX).read().splitlines()
changed=0
for i,line in enumerate(t):
    s=line.strip()
    for (lab,),val in fills.items():
        if val is None: continue
        if s.startswith(lab+" &") or s.startswith(lab+"&"):
            cells=line.split("&")
            if len(cells)>=6 and cells[3].strip()=="$-$":
                cells[3]=f" ${val}$ "
                t[i]="&".join(cells); changed+=1
                print(f"filled untagged: {lab} -> {val}")
if changed:
    open(TEX,"w").write("\n".join(t)+"\n")
print(f"backfilled {changed} cells")
