#!/usr/bin/env python3
"""Per-benchmark table of the 2B bidirectional study.

Reads docs/paper_stats/v2/bidir_2b_i.json (culture.bidirectional.aggregate --prefix i_) and
writes latex/tables/bidir_2b.tex: for every language and benchmark, Random-CPT accuracy and
the gain of each other condition over it (points). Bold marks a gain whose Holm-adjusted
p-value (within benchmark group) is below 0.05.
"""
import json
import os
import sys

OVERLEAF = "/home/jiaruil5/culture_pretrain/OverleafCultureInFigurativeLanguage"
REPO = "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage"
VARIANTS = {
    "main": ("docs/paper_stats/v2/bidir_2b_i.json", "bidir_2b.tex", "tab:bidir", "Qwen3.5-2B", "table*"),
    "base": ("docs/paper_stats/v2/bidir_2b_base.json", "bidir_2b_base.tex", "tab:bidir-base",
             "Qwen3.5-2B-Base, pretrained only", "table*"),
}
variant = sys.argv[1] if len(sys.argv) > 1 else "main"
src_rel, out_name, label, model_name, env = VARIANTS[variant]
src = os.path.join(REPO, src_rel)
out = os.path.join(OVERLEAF, "latex", "tables", out_name)
rows = json.load(open(src))["per_task"]

ARMS = ["idiom_tagged", "idiom_untagged", "culture", "culture_notes"]
GROUP_ORDER = ["idiom_meaning", "idiom_unseen", "figurative", "cloze", "symbolism",
               "culture", "regional"]
GROUP_NAME = {"idiom_meaning": "Idiom meaning", "idiom_unseen": "Idiom meaning (unseen)",
              "figurative": "Figurative", "cloze": "Idiom cloze", "symbolism": "Symbolism",
              "culture": "Culture", "regional": "Regional"}
TASK_NAME = {
    "kinayat_meaning": "Kinayat-Meaning", "kinayat_cloze": "Kinayat-Cloze",
    "ar_figurative": "AR-Figurative", "alyah": "Alyah", "dzirieval": "DziriEval",
    "arabculture": "ArabCulture", "arabic_cultural_qa": "ArabicCulturalQA",
    "chid": "ChID", "chengyu_bench": "Chengyu-Bench (conn.)", "chengyu_bench_app": "Chengyu-Bench (app.)",
    "global_piqa_hi": "Global-PIQA", "milu": "MILU", "mabl": "MABL",
    "idiomatlas_mc_ar_seen": "IdiomAtlas-MC", "idiomatlas_mc_hi_seen": "IdiomAtlas-MC",
    "idiomatlas_mc_zh_seen": "IdiomAtlas-MC", "idiomatlas_mc_ar_unseen": "IdiomAtlas-MC",
    "idiomatlas_mc_hi_unseen": "IdiomAtlas-MC", "idiomatlas_mc_zh_unseen": "IdiomAtlas-MC",
    "symbolism_v2_ar_letter": "Symbolism probe", "symbolism_v2_hi_letter": "Symbolism probe",
    "symbolism_v2_zh_letter": "Symbolism probe", "global_piqa_ar": "Global-PIQA",
    "global_piqa": "Global-PIQA", "arabicmmlu": "ArabicMMLU", "ccpm": "CCPM", "cmmlu": "CMMLU",
}
LANG = {"ar": "Arabic", "hi": "Hindi", "zh": "Chinese"}

by = {}
for r in rows:
    if r["group"] not in GROUP_ORDER:
        continue
    by.setdefault((r["lang"], r["task"]), {"group": r["group"]})[r["arm"]] = r
    if "acc_random" in r:
        by[(r["lang"], r["task"])]["random"] = r["acc_random"]


def cell(r):
    if r is None:
        return "--"
    s = f"{100 * r['delta']:+.1f}".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if r.get("p_holm_group", 1) < 0.05 else s


lines = [r"\begin{table*}[t]", r"\centering", r"\small",
         (r"\caption{Transfer in both directions at 2B (Qwen3.5-2B): accuracy of \random{} and gain of each condition over it (points). \textbf{Bold}: Holm-adjusted $p<0.05$ within the benchmark group (paired bootstrap). $-$tags: \idiomdocs{}. Chengyu-Bench (connotation) is scored after removing each model's label prior; Chengyu-Bench (app.) is its appropriateness task.}"
          if variant == "main" else
          r"\caption{Transfer in both directions at 2B from the pretrained-only checkpoint (Qwen3.5-2B-Base); layout as in Table~\ref{tab:bidir}.}"),
         r"\label{" + label + "}",
         r"\begin{tabular}{@{}lllrrrrr@{}}", r"\toprule",
         r"Lang. & Group & Benchmark & \random{} & \idiomcpt{} & $-$tags & \culturecpt{} & \culturenotes{} \\",
         r"\midrule"]
for lang in ["ar", "hi", "zh"]:
    keys = sorted((k for k in by if k[0] == lang),
                  key=lambda k: (GROUP_ORDER.index(by[k]["group"]), k[1]))
    if not keys:
        continue
    for i, k in enumerate(keys):
        d = by[k]
        rnd = d.get("random")
        lines.append(" & ".join([LANG[lang] if i == 0 else "", GROUP_NAME[d["group"]],
                                 TASK_NAME.get(k[1], k[1].replace("_", "-")),
                                 f"{100 * rnd:.1f}" if rnd is not None else "--"]
                                + [cell(d.get(a)) for a in ARMS]) + r" \\")
    lines.append(r"\midrule" if lang != "zh" else r"\bottomrule")
if lines[-1] == r"\midrule":
    lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table*}"]
open(out, "w").write("\n".join(lines) + "\n")
print("wrote", out)
