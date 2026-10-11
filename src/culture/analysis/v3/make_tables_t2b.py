#!/usr/bin/env python3
"""Render the pass-4 T2 tables: the Arabic four-arm table (with the no-holdout arm) and the
Hindi triple.

Rewrites `tables/t2.tex` (now four arms) and writes `tables/t2_hi.tex`.

    PYTHONPATH=src python -m culture.analysis.v3.make_tables_t2b
"""
from __future__ import annotations

import json
import os

STATS = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")
TEX = os.environ.get(
    "CULTURE_TEX",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "OverleafCultureInFigurativeLanguage/latex")

NAME = {
    "symbolism_v2_ar_letter": "Symbolism probe", "symbolism_ar": "Symbolism probe v1",
    "kinayat_meaning": "Kinayat-Meaning", "kinayat_cloze": "Kinayat-Cloze",
    "ar_figurative": "AR-Figurative", "idiomatlas_mc_ar_seen": "IdiomAtlas-MC seen",
    "idiomatlas_mc_ar_unseen": "IdiomAtlas-MC unseen", "alyah": "Alyah",
    "arabculture": "ArabCulture", "dzirieval": "DziriEval",
    "arabic_cultural_qa": "ArabicCulturalQA", "global_piqa_ar": "Global-PIQA ar",
    "arabicmmlu": "ArabicMMLU",
    "symbolism_v2_hi_letter": "Symbolism probe", "symbolism_hi": "Symbolism probe v1",
    "idiomatlas_mc_hi_seen": "IdiomAtlas-MC seen",
    "idiomatlas_mc_hi_unseen": "IdiomAtlas-MC unseen", "mabl": "MABL",
    "global_piqa": "Global-PIQA hi (100-item slice)",
    "global_piqa_hi": "Global-PIQA hi",
    "global_piqa_hi_cultural": "Global-PIQA hi cultural",
    "global_piqa_hi_parallel4": "Global-PIQA hi parallel",
    "parambench_hi_culture": "ParamBench culture",
    "parambench_hi_other": "ParamBench other", "milu": "MILU",
}


def load(name):
    p = os.path.join(STATS, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def w(name, body):
    os.makedirs(os.path.join(TEX, "tables"), exist_ok=True)
    p = os.path.join(TEX, "tables", name)
    open(p, "w", encoding="utf-8").write(body)
    print("[write]", p)


def cell(r, k):
    v = r.get(k)
    if v is None:
        return "--"
    s = f"{100 * v['delta']:+.1f}"
    return (r"\textbf{" + s + "}") if v["sig"] else s


def table(stats, cols, header, caption, label, fname):
    """cols: list of (contrast key, column title)."""
    spec = "lr" + "r" * len(cols)
    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{" + spec + "}", r"\toprule", header, r"\midrule"]
    for t, r in stats["accuracy"].items():
        body.append(f"{NAME.get(t, t)} & {r['n']} & "
                    + " & ".join(cell(r, k) for k, _ in cols) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", caption, label, r"\end{table}"]
    w(fname, "\n".join(body) + "\n")


def main():
    ar = load("t2_ar_stats.json")
    if ar:
        table(
            ar,
            [("dict_minus_untagged", "meaning"), ("sym_minus_untagged", "symbolism"),
             ("symall_minus_untagged", "symbolism+"), ("symall_minus_sym", "+ $-$ sym")],
            r"& $n$ & meaning & symbolism & symbolism$^{+}$ & $\Delta$ \\",
            r"\caption{T2 and its no-holdout follow-up. Four Qwen3.5-9B arms trained on the "
            r"same 410,669 Arabic idiom-bearing documents for 0.63B tokens, differing only in "
            r"what is appended: nothing, the meaning tag of \S\ref{sec:data-corpus}, "
            r"statements of what the matched idioms' entities symbolize (\emph{symbolism}, "
            r"with the probe's own entities excluded), or the same statements written for "
            r"every entity including the probe's (\emph{symbolism}$^{+}$). Accuracy points "
            r"against the untagged arm, except the last column, which is "
            r"symbolism$^{+}$ minus symbolism; bold: 95\% item bootstrap excludes zero. The "
            r"meaning tags reproduce the main study's signature; in Arabic neither symbolism "
            r"arm teaches symbolism, and lifting the holdout -- which puts 82 of the probe's "
            r"98 entities into the training text -- does not change that.}",
            r"\label{tab:t2}", "t2.tex")
    hi = load("t2_hi_stats.json")
    if hi:
        table(
            hi,
            [("dict_minus_untagged", "meaning"), ("sym_minus_untagged", "symbolism")],
            r"& $n$ & meaning tags & symbolism tags \\",
            r"\caption{T2 in Hindi. Three Qwen3.5-9B arms trained on the same 228{,}413 Hindi "
            r"idiom-bearing documents for 0.63B tokens, differing only in what is appended: "
            r"nothing, the meaning tag, or statements of what the matched proverbs' entities "
            r"symbolize in Hindi (probe entities excluded). Accuracy points against the "
            r"untagged arm; bold: 95\% item bootstrap excludes zero. The two tag types "
            r"dissociate: the meaning tags buy $+17.5$ on seen idiom meanings and nothing on "
            r"the probe, the symbolism tags buy $+9.0$ on the probe and lose $4.0$ on seen "
            r"idiom meanings. The probe's 155 entities are held out of the statements, so its "
            r"gain is generalisation, not recall.}",
            r"\label{tab:t2hi}", "t2_hi.tex")


if __name__ == "__main__":
    main()
