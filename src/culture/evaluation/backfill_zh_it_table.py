#!/usr/bin/env python3
"""Fill the Chinese table's +IT column (col index 5) from the zh-cpt-sft eval outputs.
Idempotent: only replaces cells still showing `$-$`. Category (2) zh LM PPL is not
produced by the SFT eval, so those rows stay dashed."""
import json, os

TEX="/storage/home/jiaruiliu/local/git-repos/culture-pretraining/OverleafCultureInFigurativeLanguage/colm2026_conference.tex"
EV="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/zh-cpt-sft"

def tasks(p):
    try: return json.load(open(p)).get("tasks",{})
    except Exception: return {}
def ppl(p,k="ppl"):
    try: return json.load(open(p)).get(k)
    except Exception: return None
def pct(d,k):
    v=d.get(k,{}).get("primary"); return None if v is None else f"{v*100:.1f}"

s=tasks(f"{EV}/summary.json"); en=tasks(f"{EV}/en/summary.json"); eg=tasks(f"{EV}/en_gen/summary.json")
wt=ppl(f"{EV}/en/ppl_wikitext/perplexity.json") or ppl(f"{EV}/ppl_wikitext/perplexity.json")
fills={
 "Chengyu-Bench (connotation, acc \\%)": pct(s,"chengyu_bench"),
 "ChID (cloze, acc \\%)":                pct(s,"chid"),
 "CMMLU (all 67, 5-shot, acc \\%)":      pct(s,"cmmlu"),
 "CCPM (acc$_\\text{norm}$ \\%)":        pct(s,"ccpm"),
 "MMLU (0-shot, acc \\%)":               pct(en,"mmlu"),
 "BoolQ (0-shot, acc \\%)":              pct(en,"boolq"),
 "GSM8K (8-shot CoT, EM \\%)":           pct(eg,"gsm8k"),
 "HumanEval (0-shot, pass@1 \\%)":       pct(eg,"humaneval"),
 "WikiText-103 (PPL, $\\downarrow$)":    (None if wt is None else f"{wt:.1f}"),
}
t=open(TEX).read().splitlines(); changed=0
for i,line in enumerate(t):
    s2=line.strip()
    for lab,val in fills.items():
        if val is None: continue
        if s2.startswith(lab+" &") or s2.startswith(lab+"&"):
            cells=line.split("&")
            if len(cells)>=6 and cells[5].strip().rstrip("\\").strip()=="$-$":
                has_tail = cells[5].rstrip().endswith("\\\\")
                cells[5]=f" ${val}$ " + ("\\\\" if has_tail else "")
                t[i]="&".join(cells); changed+=1
                print(f"filled zh +IT: {lab} -> {val}")
if changed: open(TEX,"w").write("\n".join(t)+"\n")
print(f"backfilled {changed} zh +IT cells")
