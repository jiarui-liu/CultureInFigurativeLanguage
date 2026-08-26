#!/usr/bin/env python3
"""Aggregate SFT eval outputs into FINAL_REPORT.md, comparing each SFT checkpoint
to the existing pre-SFT CPT-eval numbers (base / cpt)."""
import json, os, glob

EVAL="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval"

def load_summary(run_dir):
    m={}
    sp=os.path.join(run_dir,"summary.json")
    if os.path.isfile(sp):
        try:
            for t,v in json.load(open(sp)).get("tasks",{}).items():
                m[t]=v.get("primary")
        except Exception: pass
    # perplexity probes
    for pj in glob.glob(os.path.join(run_dir,"ppl_*","perplexity.json")):
        try:
            d=json.load(open(pj)); name=os.path.basename(os.path.dirname(pj))
            m[name]=d.get("bpb", d.get("ppl", d.get("perplexity")))
        except Exception: pass
    # english extras + generative
    for sub in ("en","en_gen"):
        sp=os.path.join(run_dir,sub,"summary.json")
        if os.path.isfile(sp):
            try:
                for t,v in json.load(open(sp)).get("tasks",{}).items():
                    m[f"{sub}:{t}"]=v.get("primary", v.get("acc", v.get("pass@1")))
            except Exception: pass
    for tj in glob.glob(os.path.join(run_dir,"en_gen","*.json"))+glob.glob(os.path.join(run_dir,"en","*.json")):
        b=os.path.basename(tj)
        if b=="summary.json": continue
        try:
            d=json.load(open(tj)); mm=d.get("metrics",d)
            key=("en_gen:" if "en_gen" in tj else "en:")+b[:-5]
            m.setdefault(key, mm.get("primary", mm.get("acc", mm.get("pass@1", mm.get("em")))))
        except Exception: pass
    return m

def fmt(x):
    return "—" if x is None else (f"{x:.4f}" if isinstance(x,(int,float)) else str(x))

def table(lang, runs):
    refs={r:load_summary(f"{EVAL}/{lang}/{r}") for r in ("base","cpt")}
    sfts={r:load_summary(f"{EVAL}/{lang}/{r}") for r in runs}
    tasks=[]
    for d in list(refs.values())+list(sfts.values()):
        for k in d:
            if k not in tasks: tasks.append(k)
    lines=[f"### {lang.upper()}",""]
    hdr=["task","base(CPTeval)","cpt(CPTeval)"]+runs
    lines.append("| "+" | ".join(hdr)+" |")
    lines.append("|"+"---|"*len(hdr))
    for t in sorted(tasks):
        row=[t, fmt(refs["base"].get(t)), fmt(refs["cpt"].get(t))]+[fmt(sfts[r].get(t)) for r in runs]
        lines.append("| "+" | ".join(row)+" |")
    lines.append("")
    return "\n".join(lines)

def main():
    out=["# SFT FINAL REPORT","",
         "Eval = the SAME pipeline used for the CPT checkpoints, run on the 4 SFT checkpoints.",
         "`base`/`cpt` columns are the pre-SFT CPT-eval numbers for reference (higher = better,",
         "except ppl_* which are BPB/perplexity where lower = better).",""]
    out.append(table("hi", ["hi-base-sft","hi-cpt-sft"]))
    out.append(table("ar", ["ar-base-sft","ar-cpt-sft"]))
    # manifests
    out.append("### Mixture manifests")
    for L in ("hi","ar"):
        mp=f"/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_{L}.manifest.json"
        if os.path.isfile(mp):
            d=json.load(open(mp))
            out.append(f"- **{L}**: total={d.get('total')} target={d.get('target_n')} "
                       f"english={d.get('english_n')} realized_target_ratio={d.get('realized_target_ratio')} seed={d.get('seed')}")
    rep="\n".join(out)+"\n"
    dst=os.path.join(os.path.dirname(os.path.abspath(__file__)),"FINAL_REPORT.md")
    open(dst,"w").write(rep)
    print("wrote",dst)

if __name__=="__main__":
    main()
