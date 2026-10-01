#!/usr/bin/env python3
"""Generate latex/tables/main_results.tex from docs/paper_stats/ci_report.json."""
import json, os
OVERLEAF = "/home/jiaruil5/culture_pretrain/OverleafCultureInFigurativeLanguage"  # paper repo (LaTeX only)
REPO = "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage"
# v2 = recomputed from the per-item records on the HF repo with the ArabCulture qid fix.
# zh Idiom-CPT-tags records are not on the HF repo, so that run is taken from the v1 report.
rep = json.load(open(os.path.join(REPO, "docs/paper_stats/v2/ci_report.json")))
_v1 = json.load(open(os.path.join(REPO, "docs/paper_stats/ci_report.json")))
for _t, _r in _v1["zh"].items():
    if _t in rep["zh"] and "untagged" not in rep["zh"][_t]["per_run"] and "untagged" in _r["per_run"]:
        rep["zh"][_t]["per_run"]["untagged"] = _r["per_run"]["untagged"]
        rep["zh"][_t]["contrasts"]["cpt_vs_untagged"] = _r["contrasts"]["cpt_vs_untagged"]
OUT = os.path.join(OVERLEAF, "latex", "tables", "main_results.tex")
GROUPS = [
    ("Idiom meaning", [("ar", "kinayat_meaning", "Kinayat-Meaning"), ("zh", "chengyu_bench", "Chengyu-Bench$^{\\dagger}$")]),
    ("Figurative inference", [("ar", "ar_figurative", "AR-Figurative"), ("hi", "mabl", "MABL")]),
    ("Idiom cloze", [("ar", "kinayat_cloze", "Kinayat-Cloze"), ("zh", "chid", "ChID")]),
    ("Culture", [("ar", "alyah", "Alyah"), ("ar", "dzirieval", "DziriEval"), ("ar", "arabculture", "ArabCulture"),
                 ("ar", "arabic_cultural_qa", "ArabicCulturalQA"), ("ar", "global_piqa_ar", "Global-PIQA"),
                 ("hi", "global_piqa", "Global-PIQA"), ("zh", "ccpm", "CCPM")]),
    ("Regional knowledge", [("ar", "arabicmmlu", "ArabicMMLU"), ("hi", "milu", "MILU"), ("zh", "cmmlu", "CMMLU")]),
]
def delta(c):
    d = 100 * c["delta"]
    s = f"{d:+.1f}".replace("+", "$+$").replace("-", "$-$")
    return (r"\textbf{%s}$^{*}$" % s) if c["sig_0.05"] else s
rows = []
for g, tasks in GROUPS:
    rows.append(r"\multicolumn{10}{@{}l}{\textit{%s}}\\" % g)
    for lang, t, name in tasks:
        r = rep[lang][t]; a = r["per_run"]; c = r["contrasts"]
        acc = {k: 100 * a[k]["acc"] for k in ("base", "unfiltered", "untagged", "cpt")}
        best = max(acc.values())
        f = lambda k: (r"\textbf{%.1f}" % acc[k]) if abs(acc[k] - best) < 1e-9 else "%.1f" % acc[k]
        n = c["cpt_vs_unfiltered"]["n"]
        rows.append(f"{name} & {lang} & {n:,} & {f('base')} & {f('unfiltered')} & {f('untagged')} & "
                    f"\\cellcolor{{ourrow}}{f('cpt')} & {delta(c['cpt_vs_base'])} & "
                    f"{delta(c['cpt_vs_unfiltered'])} & {delta(c['cpt_vs_untagged'])} \\\\")
    rows.append(r"\midrule")
rows = rows[:-1]
tex = r"""\begin{table*}[t]
\centering
\small
\caption{Accuracy (\%) of the four training conditions. \random{}: token-matched random documents from the same sources. \idiomdocs{}: idiom-bearing documents with the meaning tags removed. \idiomcpt{}: idiom-bearing documents with meaning tags (ours, \colorsquare{ourrow}). The last three columns give the gain of \idiomcpt{} over each reference; \textbf{bold}$^{*}$ marks gains whose paired-bootstrap 95\% confidence interval excludes zero and McNemar $p<0.05$. $n$ is the number of paired test items. The comparison with \random{} isolates the effect of idioms from in-language exposure. $^{\dagger}$Chengyu-Bench (connotation) has two fixed labels, so its accuracy mostly tracks each model's label prior: after removing that prior (median-centering the log-probability difference), \idiomcpt{} and \random{} both score 95.6 ($\Delta$ $+0.0$, $[-1.3, 1.3]$; AUC 0.986 vs.\ 0.987).}
\label{tab:main}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llrcccccccc@{}}
\toprule
\multirow{2}{*}{Benchmark} & \multirow{2}{*}{Lang.} & \multirow{2}{*}{$n$} & \multirow{2}{*}{Base} & \multirow{2}{*}{\random{}} & \multirow{2}{*}{\shortstack{\idiomdocs{}}} & \multirow{2}{*}{\idiomcpt{}} & \multicolumn{3}{c}{$\Delta$ of \idiomcpt{} over} \\
\cmidrule(l){8-10}
& & & & & & & Base & Random & $-$tags \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table*}
"""
tex = tex.replace("llrcccccccc", "llrccccccc")
open(OUT, "w").write(tex); print("wrote", OUT)
