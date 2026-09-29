#!/usr/bin/env python3
"""latex/tables/ci_appendix.tex: every 9B contrast with paired-bootstrap CI and McNemar p (v2 report)."""
import json, os
OVERLEAF = "/home/jiaruil5/culture_pretrain/OverleafCultureInFigurativeLanguage"  # paper repo (LaTeX only)
REPO = "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage"
rep = json.load(open(os.path.join(REPO, "docs/paper_stats/v2/ci_report.json")))
v1 = json.load(open(os.path.join(REPO, "docs/paper_stats/ci_report.json")))
NAMES = {"kinayat_meaning": "Kinayat-Meaning", "chengyu_bench": "Chengyu-Bench", "ar_figurative": "AR-Figurative",
         "mabl": "MABL", "kinayat_cloze": "Kinayat-Cloze", "chid": "ChID", "alyah": "Alyah", "dzirieval": "DziriEval",
         "arabculture": "ArabCulture", "arabic_cultural_qa": "ArabicCulturalQA", "global_piqa_ar": "Global-PIQA",
         "global_piqa": "Global-PIQA", "ccpm": "CCPM", "arabicmmlu": "ArabicMMLU", "milu": "MILU", "cmmlu": "CMMLU",
         "global_piqa_ar_parallel": "Global-PIQA (parallel)"}
def cell(c):
    if not c: return "--"
    lo, hi = c["ci95"]
    s = f"{100*c['delta']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]".replace("+", "$+$").replace("-", "$-$")
    return r"\textbf{%s}" % s if (lo > 0 or hi < 0) else s
rows = []
for L in ["ar", "hi", "zh"]:
    for t, r in sorted(rep[L].items(), key=lambda kv: list(NAMES).index(kv[0]) if kv[0] in NAMES else 99):
        if t not in NAMES: continue
        c = r["contrasts"]
        unt = c.get("cpt_vs_untagged") or v1.get(L, {}).get(t, {}).get("contrasts", {}).get("cpt_vs_untagged")
        rows.append(f"{NAMES[t]} & {L} & {c['cpt_vs_unfiltered']['n']:,} & {cell(c.get('cpt_vs_base'))} & "
                    f"{cell(c.get('cpt_vs_unfiltered'))} & {cell(unt)} \\\\")
tex = r"""\begin{table*}[t]
\centering
\small
\caption{All contrasts of the 9B study: gain of \idiomcpt{} (percentage points) with paired-bootstrap 95\%% confidence intervals (10{,}000 resamples); \textbf{bold}: interval excludes zero. ArabCulture uses all 3{,}463 items (an earlier version of the analysis collapsed items with duplicate identifiers to 2{,}168).}
\label{tab:ci-appendix}
\begin{tabular}{@{}llrccc@{}}
\toprule
Benchmark & Lang. & $n$ & vs.\ Base & vs.\ \random{} & vs.\ \idiomdocs{} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % "\n".join(rows)
out = os.path.join(OVERLEAF, "latex", "tables", "ci_appendix.tex")
open(out, "w").write(tex); print("wrote", out, len(rows))
