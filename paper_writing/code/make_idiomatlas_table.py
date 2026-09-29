#!/usr/bin/env python3
"""latex/tables/idiomatlas_9b.tex from docs/paper_stats/v2/idiomatlas_9b.json (9B checkpoints on IdiomAtlas-MC)."""
import json, os
OVERLEAF = "/home/jiaruil5/culture_pretrain/OverleafCultureInFigurativeLanguage"  # paper repo (LaTeX only)
REPO = "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage"
d = json.load(open(os.path.join(REPO, "docs/paper_stats/v2/idiomatlas_9b.json")))
OUT = os.path.join(OVERLEAF, "latex", "tables", "idiomatlas_9b.tex")
N = {"ar_seen": 600, "ar_unseen": 600, "hi_seen": 600, "hi_unseen": 600, "zh_seen": 600, "zh_unseen": 78}
def fmt_delta(t):
    if t is None: return "--"
    dv, ci, p = t
    s = f"{dv:+.1f}".replace("+", "$+$").replace("-", "$-$")
    return (r"\textbf{%s}$^{*}$" % s) if (ci[0] > 0 or ci[1] < 0) else s
rows = []
for L, name in [("ar", "Arabic"), ("hi", "Hindi"), ("zh", "Chinese")]:
    for sp in ["seen", "unseen"]:
        r = d[f"{L}_{sp}"]
        g = lambda k: f"{r[k]:.1f}" if k in r else "--"
        rows.append(f"{name if sp == 'seen' else ''} & {sp} & {N[L+'_'+sp]} & {g('base')} & {g('unfiltered')} & {g('untagged')} & "
                    f"\\cellcolor{{ourrow}}{g('cpt')} & {fmt_delta(r.get('cpt-unfiltered'))} & {fmt_delta(r.get('untagged-unf'))} \\\\")
    rows.append(r"\midrule")
rows = rows[:-1]
tex = r"""\begin{table}[t]
\centering
\small
\caption{IdiomAtlas-MC accuracy (\%%) of the 9B models on idioms whose meanings occur in the meaning tags (\textit{seen}) and on idioms that occur in no scanned document (\textit{unseen}); chance is 25\%%. The last two columns give the gain of \idiomcpt{} and of \idiomdocs{} over \random{}; \textbf{bold}$^{*}$: paired-bootstrap 95\%% CI excludes zero. The Chinese \idiomdocs{} checkpoint was not available for this evaluation.}
\label{tab:idiomatlas}
\resizebox{\columnwidth}{!}{%%
\begin{tabular}{@{}llrccccrr@{}}
\toprule
& Split & $n$ & Base & Rand. & $-$tags & Idiom & $\Delta_{\text{Idiom}}$ & $\Delta_{-\text{tags}}$ \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(rows)
open(OUT, "w").write(tex); print("wrote", OUT)
