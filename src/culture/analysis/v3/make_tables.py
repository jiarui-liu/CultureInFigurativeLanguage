#!/usr/bin/env python3
"""Render the pass-3 tables and figures into the Overleaf checkout from docs/paper_stats/v3.

    PYTHONPATH=src python -m culture.analysis.v3.make_tables
"""
from __future__ import annotations

import json
import os

import numpy as np

STATS = os.environ.get(
    "CULTURE_STATS_V3",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "CultureInFigurativeLanguage/docs/paper_stats/v3")
TEX = os.environ.get(
    "CULTURE_TEX",
    "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/"
    "OverleafCultureInFigurativeLanguage/latex")

ARM_TEX = {"random": r"\random{}", "idiom": r"\idiomcpt{}",
           "idiom_untagged": r"\idiomdocs{}", "culture": r"\culturecpt{}",
           "culturenotes": r"\culturenotes{}"}
LANG_NAME = {"ar": "Arabic", "hi": "Hindi", "zh": "Chinese"}
ARM_ORDER = ["random", "idiom", "idiom_untagged", "culture", "culturenotes"]


def load(name):
    p = os.path.join(STATS, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def w(name, body):
    os.makedirs(os.path.join(TEX, "tables"), exist_ok=True)
    p = os.path.join(TEX, "tables", name)
    open(p, "w", encoding="utf-8").write(body)
    print("[write]", p)


# --------------------------------------------------------------- corpus profile
def corpus_profile():
    d = load("corpus_affinity.json")
    if not d:
        return
    cs = d["culture_score"]
    rows = []
    for lang in ("ar", "hi", "zh"):
        for arm in ARM_ORDER:
            v = cs.get(f"{lang}/{arm}")
            if not v:
                continue
            rows.append((lang, arm, v))
    body = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Language & Corpus & Mean & Median & $\geq 3$ \\",
        r"\midrule",
    ]
    last = None
    for lang, arm, v in rows:
        lab = LANG_NAME[lang] if lang != last else ""
        last = lang
        hl = r"\rowcolor{ourrow}" if (lang == "zh" and arm == "culture") else ""
        body.append(f"{hl}{lab} & {ARM_TEX[arm]} & {v['mean']:.2f} & {v['median']:.2f} "
                    f"& {100 * v['share_ge3']:.1f}\\% \\\\")
        if arm == ARM_ORDER[-1] or (lang, arm) == rows[-1][:2]:
            pass
    body += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Culture specificity of each training corpus, scored by the paper's own "
        r"classifier (\S\ref{sec:setup}) on 6{,}000 documents per corpus, with the appended "
        r"tags and notes removed so that only the web text is scored. Idiom-bearing documents "
        r"are markedly more culture-specific than random text in all three languages. The "
        r"highlighted row is the exception that explains the Chinese results: after idiom-"
        r"bearing documents are excluded, the Chinese \culturecpt{} corpus scores \emph{below} "
        r"the Chinese idiom corpus and close to random text.}",
        r"\label{tab:corpus-profile}", r"\end{table}",
    ]
    w("corpus_profile.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- margins
def margin_table():
    d = load("margins_and_churn.json")
    if not d:
        return
    IDIOM = {"kinayat_meaning": "Kinayat-Mean.", "ar_figurative": "AR-Figurative",
             "idiomatlas_mc_ar_seen": "IA-MC ar seen",
             "idiomatlas_mc_ar_unseen": "IA-MC ar unseen",
             "idiomatlas_mc_hi_seen": "IA-MC hi seen",
             "idiomatlas_mc_hi_unseen": "IA-MC hi unseen",
             "idiomatlas_mc_zh_seen": "IA-MC zh seen",
             "chengyu_bench": "Chengyu-Bench", "mabl": "MABL", "chid": "ChID"}
    arms = ["idiom_cpt", "idiom_untagged", "culture", "culture_notes"]
    head = {"idiom_cpt": r"\idiomcpt{}", "idiom_untagged": r"\idiomdocs{}",
            "culture": r"\culturecpt{}", "culture_notes": r"\culturenotes{}"}
    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{l" + "r" * len(arms) + "}", r"\toprule",
            "Benchmark & " + " & ".join(head[a] for a in arms) + r" \\",
            r"\midrule"]
    for task, name in IDIOM.items():
        cells = []
        any_cell = False
        for a in arms:
            k = next((k for k in d["margin"] if k.endswith(f"/{task}/{a}")), None)
            if k is None:
                cells.append("--")
                continue
            v = d["margin"][k]
            any_cell = True
            s = f"{v['delta_margin']:+.2f}"
            cells.append(r"\textbf{" + s + "}" if v["sig"] else s)
        if any_cell:
            body.append(f"{name} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Change in the decision margin (log-probability of the gold option "
             r"minus the best distractor, length-normalised, in nats) against \random{} at 9B. "
             r"Bold: item-level paired-bootstrap 95\% CI excludes zero. Culture-rich text "
             r"moves the margin on idiom-meaning benchmarks in Arabic and Chinese even where "
             r"accuracy does not move.}",
             r"\label{tab:margins}", r"\end{table}"]
    w("margins.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- churn
def churn_table():
    d = load("margins_and_churn.json")
    if not d:
        return
    rows = []
    for k, v in d["churn"].items():
        lang, task, arm = k.split("/")
        if arm == "base" or v["n"] < 300:
            continue
        rows.append((abs(v["delta_acc"]), k, v))
    rows.sort()
    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\begin{tabular}{llrrrr}", r"\toprule",
            r"Benchmark & Arm & $n$ & $\Delta$acc & fixed & broken \\", r"\midrule"]
    for _, k, v in rows[:14]:
        lang, task, arm = k.split("/")
        body.append(f"{task.replace('_', chr(92) + '_')} ({lang}) & "
                    f"{arm.replace('_', chr(92) + '_')} & {v['n']} & "
                    f"{100 * v['delta_acc']:+.1f} & {100 * v['fixed'] / v['n']:.1f}\\% & "
                    f"{100 * v['broken'] / v['n']:.1f}\\% \\\\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{The fourteen arm-benchmark pairs with the smallest accuracy change at "
             r"9B, and the share of items each arm answers differently from \random{}. A null "
             r"accuracy delta is not a null model: they answer 7--15\% of the items "
             r"differently from the control, and the two directions nearly cancel.}",
             r"\label{tab:churn}", r"\end{table}"]
    w("churn.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- selection bias
def selection_bias():
    d = load("kb_corpus_benchmark.json")
    if not d:
        return
    sb = d["selection_bias"]
    conc = d.get("occurrence_concentration", {})
    types = sorted({t for v in sb.values() for t in v["type_share_attested"]})
    langs = list(sb)
    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\begin{tabular}{l" + "r" * len(langs) + "}", r"\toprule",
            r"Entity type & " + " & ".join(LANG_NAME[l] for l in langs) + r" \\",
            r"\midrule"]
    for t in types:
        cells = []
        for l in langs:
            v = sb[l]
            cells.append(f"{100 * (v['type_share_attested'].get(t, 0) - v['type_share_unattested'].get(t, 0)):+.1f}")
        body.append(t.replace("_", " ") + " & " + " & ".join(cells) + r" \\")
    body.append(r"\midrule")
    body.append("attested idioms & " + " & ".join(
        f"{100 * sb[l]['coverage']:.1f}\\%" for l in langs) + r" \\")
    body.append(r"Cram\'er's $V$ & " + " & ".join(f"{sb[l]['cramers_v']:.3f}" for l in langs) + r" \\")
    if conc:
        body.append("Gini of occurrences & " + " & ".join(
            f"{conc[l]['gini']:.2f}" if l in conc else "--" for l in langs) + r" \\")
        body.append("median docs per idiom & " + " & ".join(
            f"{conc[l]['median']:.0f}" if l in conc else "--" for l in langs) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Which idioms reach the training corpus. Each cell is the difference, in "
             r"percentage points, between the share of entity mentions of that semantic type "
             r"among web-attested idioms and among unattested ones. The two distributions differ "
             r"but weakly, so the corpus is a biased rather than a distorted sample of "
             r"\dataname{}; the Hindi corpus is tilted away from kinship and household imagery. "
             r"The last rows give the coverage and the concentration of the occurrence "
             r"distribution.}",
             r"\label{tab:selection-bias}", r"\end{table}"]
    w("selection_bias.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- entity coverage
def entity_coverage():
    d = load("kb_corpus_benchmark.json")
    if not d:
        return
    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\begin{tabular}{llrrr}", r"\toprule",
            r"Lang & Benchmark & $n$ & with entity & $\rho$ \\", r"\midrule"]
    for k, v in d["entity_coverage"].items():
        lang, task = k.split("/", 1)
        body.append(f"{lang} & {task.replace('_', chr(92) + '_')} & {v['n_items']} & "
                    f"{100 * v['share_items_with_kb_entity']:.0f}\\% & "
                    f"{v['spearman_idiom_freq_vs_benchmark_freq']:.2f} \\\\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Entity coverage of the benchmarks. ``with entity'' is the share of items "
             r"that mention at least one \dataname{} entity of the same language (surface-form "
             r"match, entities occurring in at least three idioms); $\rho$ is the Spearman "
             r"correlation between an entity's frequency across idioms and its frequency across "
             r"benchmark items. The benchmarks mention the same imagery the idioms are built "
             r"from; what differs is what they ask about it.}",
             r"\label{tab:entity-coverage}", r"\end{table}"]
    w("entity_coverage.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- capability profile
def capability():
    d = load("margins_and_churn.json")
    if not d or not d.get("perplexity"):
        return
    probes, arms = [], []
    for k in d["perplexity"]:
        lang, probe, arm = k.split("/")
        if (lang, probe) not in probes:
            probes.append((lang, probe))
        if arm not in arms:
            arms.append(arm)
    order = [a for a in ("base", "random", "idiom_cpt", "idiom_untagged",
                         "culture", "culture_notes") if a in arms]
    head = {"base": "Base", "random": r"\random{}", "idiom_cpt": r"\idiomcpt{}",
            "idiom_untagged": r"\idiomdocs{}", "culture": r"\culturecpt{}",
            "culture_notes": r"\culturenotes{}"}
    body = [r"\begin{table*}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{ll" + "r" * len(order) + "}", r"\toprule",
            r"Lang & Probe & " + " & ".join(head[a] for a in order) + r" \\", r"\midrule"]
    for lang, probe in probes:
        cells = []
        for a in order:
            v = d["perplexity"].get(f"{lang}/{probe}/{a}")
            cells.append(f"{v['ppl']:.2f}" if v and v.get("ppl") else "--")
        body.append(f"{lang} & {probe.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Perplexity of every 9B arm on the held-out language-modelling probes. "
             r"The culture arms were added in this pass; they are the arms that read the "
             r"narrowest slice of the language and had never been measured.}",
             r"\label{tab:capability}", r"\end{table*}"]
    w("capability.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- item taxonomy
CAT_SHORT = {"symbolic_evaluative": "Symb.", "social_norm_relation": "Norm",
             "material_practice": "Pract.", "factual_knowledge": "Fact",
             "language_form": "Lang.", "generic_pragmatic": "Gen."}
TASK_NAME = {
    "kinayat_meaning": "Kinayat-Meaning", "ar_figurative": "AR-Figurative",
    "mabl": "MABL (hi)", "chengyu_bench": "Chengyu-Bench",
    "idiomatlas_mc_ar_seen": "IdiomAtlas-MC ar", "idiomatlas_mc_hi_seen": "IdiomAtlas-MC hi",
    "idiomatlas_mc_zh_seen": "IdiomAtlas-MC zh",
    "symbolism_v2_ar_letter": "Symbolism ar", "symbolism_v2_hi_letter": "Symbolism hi",
    "symbolism_v2_zh_letter": "Symbolism zh",
    "alyah": "Alyah", "dzirieval": "DziriEval", "arabculture": "ArabCulture",
    "arabic_cultural_qa": "ArabicCulturalQA", "global_piqa_ar": "Global-PIQA ar",
    "global_piqa_hi": "Global-PIQA hi", "global_piqa_zh": "Global-PIQA zh",
    "ccpm": "CCPM", "arabicmmlu": "ArabicMMLU", "milu": "MILU", "cmmlu": "CMMLU",
    "parambench_hi_culture": "ParamBench-cult.", "parambench_hi_other": "ParamBench-other",
}
GROUPS = [
    ("Idiom and figurative", ["kinayat_meaning", "ar_figurative", "mabl", "chengyu_bench",
                              "idiomatlas_mc_ar_seen", "idiomatlas_mc_hi_seen",
                              "idiomatlas_mc_zh_seen"]),
    ("Symbolism probe", ["symbolism_v2_ar_letter", "symbolism_v2_hi_letter",
                         "symbolism_v2_zh_letter"]),
    ("Culture", ["alyah", "dzirieval", "arabculture", "arabic_cultural_qa",
                 "global_piqa_ar", "global_piqa_hi", "global_piqa_zh", "ccpm"]),
    ("Regional knowledge", ["arabicmmlu", "milu", "cmmlu", "parambench_hi_culture",
                            "parambench_hi_other"]),
]


def taxonomy_table():
    d = load("item_taxonomy.json")
    if not d:
        return
    dist = d["distribution"]
    cats = list(CAT_SHORT)
    body = [r"\begin{table*}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lr" + "r" * len(cats) + "}", r"\toprule",
            r"Source & $n$ & " + " & ".join(CAT_SHORT[c] for c in cats) + r" \\"]
    for gname, tasks in GROUPS:
        body.append(r"\midrule")
        body.append(r"\multicolumn{" + str(2 + len(cats)) + r"}{l}{\textit{" + gname + r"}} \\")
        for t in tasks:
            c = dist.get(t)
            if not c:
                continue
            tot = sum(c.values())
            cells = []
            for k in cats:
                v = 100 * c.get(k, 0) / tot
                s = f"{v:.0f}"
                cells.append(r"\textbf{" + s + "}" if v >= 50 else s)
            body.append(f"{TASK_NAME.get(t, t)} & {tot} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Culture-layer taxonomy applied to \emph{every} item of each benchmark "
             r"(up to 4{,}000 per source), not a 400-item sample: percentage of items in each "
             r"category under the primary annotator (Qwen3.5-27B). Columns are symbolic or "
             r"evaluative, social norms and relations, material practice, factual knowledge, "
             r"language form, and generic pragmatics. Alyah and DziriEval, the two culture "
             r"benchmarks with the largest symbolic share, are the two that \idiomcpt{} "
             r"improves most.}",
             r"\label{tab:item-taxonomy}", r"\end{table*}"]
    w("item_taxonomy.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- english retention
def english_retention():
    import glob as _glob
    root = os.environ.get(
        "CULTURE_EVAL_RESULTS",
        "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval")
    arm_dir = {"base": "base", "random": "unfiltered", "idiom_cpt": "cpt",
               "idiom_untagged": "untagged", "culture": "culture",
               "culture_notes": "culturenotes"}
    head = {"base": "Base", "random": r"\random{}", "idiom_cpt": r"\idiomcpt{}",
            "idiom_untagged": r"\idiomdocs{}", "culture": r"\culturecpt{}",
            "culture_notes": r"\culturenotes{}"}
    order = list(arm_dir)
    tasks = [("en/mmlu.json", "MMLU"), ("en/boolq.json", "BoolQ"),
             ("en_gen/gsm8k.json", "GSM8K"), ("en_gen/humaneval.json", "HumanEval"),
             ("en/ppl_wikitext/perplexity.json", "WikiText ppl")]
    rows = []
    for lang in ("ar", "hi", "zh"):
        for rel, name in tasks:
            cells, any_ = [], False
            for a in order:
                p = f"{root}/{lang}/{arm_dir[a]}/{rel}"
                if not os.path.exists(p):
                    cells.append("--")
                    continue
                d = json.load(open(p))
                v = d.get("ppl") if "perplexity" in rel else d.get("metrics", {}).get("primary")
                if v is None:
                    cells.append("--")
                    continue
                any_ = True
                cells.append(f"{v:.1f}" if "perplexity" in rel else f"{100 * v:.1f}")
            if any_:
                rows.append((lang, name, cells))
    if not rows:
        return
    body = [r"\begin{table*}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{ll" + "r" * len(order) + "}", r"\toprule",
            r"CPT lang & Task & " + " & ".join(head[a] for a in order) + r" \\", r"\midrule"]
    last = None
    for lang, name, cells in rows:
        if lang != last:
            body.append(r"\midrule" if last else "")
            last = lang
        body.append(f"{lang} & {name} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{English retention for every 9B arm (accuracy, or perplexity for "
             r"WikiText). The culture arms are added in this pass. Continued pretraining in "
             r"the target language costs English code generation in every condition, including "
             r"the token-matched control, so the cost belongs to continued pretraining and not "
             r"to any particular corpus. A dash marks an evaluation that did not complete "
             r"(the WikiText download failed on the compute nodes for the culture arms).}",
             r"\label{tab:english-retention}", r"\end{table*}"]
    w("english_retention.tex", "\n".join(body) + "\n")



# --------------------------------------------------------------- T1
T1_NAME = {"chid": "ChID (idiom cloze)", "chengyu_bench": "Chengyu-Bench",
           "chengyu_bench_app": "Chengyu-Bench app.", "cmmlu": "CMMLU",
           "ccpm": "CCPM", "global_piqa_zh": "Global-PIQA zh",
           "global_piqa_zh_cultural": "Global-PIQA zh (cult.)",
           "idiomatlas_mc_zh_seen": "IdiomAtlas-MC seen",
           "idiomatlas_mc_zh_unseen": "IdiomAtlas-MC unseen",
           "symbolism_v2_zh_letter": "Symbolism probe"}


def t1_table():
    d = load("t1_stats.json")
    if not d:
        return
    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lrrrr}", r"\toprule",
            r"& $n$ & restr. & unrestr. & restr.$-$unrestr. \\",
            r"\midrule"]
    for t, r in d["accuracy"].items():
        def c(k, bold=False):
            v = r[k]
            s = f"{100 * v['delta']:+.1f}"
            return (r"\textbf{" + s + "}") if v["sig"] else s
        body.append(f"{T1_NAME.get(t, t)} & {r['n']} & {c('free_minus_random')} & "
                    f"{c('all_minus_random')} & {c('free_minus_all')} \\\\")
    body.append(r"\midrule")
    for t, r in d["perplexity"].items():
        if not r:
            continue
        nm = {"ppl_zh_wiki": "zh Wikipedia (ppl)",
              "ppl_zh_chengyu": "held-out chengyu text (ppl)"}.get(t, t)
        base = r["random"]["ppl"]
        body.append(f"{nm} & -- & {r['culture_free']['ppl'] - base:+.2f} & "
                    f"{r['culture_all']['ppl'] - base:+.2f} & "
                    f"{r['culture_free']['ppl'] - r['culture_all']['ppl']:+.2f} \\\\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{T1: is the Chinese cost of \culturecpt{} caused by the idiom-free "
             r"restriction? Three Qwen3.5-9B arms trained from the same checkpoint on 1.05B "
             r"tokens of the same Chinese pool, differing only in the selection rule: "
             r"uniformly random, culture-rich with the restriction (restr.), and culture-rich "
             r"without it (unrestr.). Columns 3 and 4 are accuracy points against the random "
             r"arm, or perplexity differences; the last column is the effect of the "
             r"restriction alone. Bold: 95\% item bootstrap excludes zero.}",
             r"\label{tab:t1}", r"\end{table}"]
    w("t1.tex", "\n".join(body) + "\n")



# --------------------------------------------------------------- T2
T2_NAME = {"symbolism_v2_ar_letter": "Symbolism probe", "symbolism_ar": "Symbolism probe v1",
           "kinayat_meaning": "Kinayat-Meaning", "kinayat_cloze": "Kinayat-Cloze",
           "ar_figurative": "AR-Figurative", "idiomatlas_mc_ar_seen": "IdiomAtlas-MC seen",
           "idiomatlas_mc_ar_unseen": "IdiomAtlas-MC unseen", "alyah": "Alyah",
           "arabculture": "ArabCulture", "dzirieval": "DziriEval",
           "arabic_cultural_qa": "ArabicCulturalQA", "global_piqa_ar": "Global-PIQA ar",
           "arabicmmlu": "ArabicMMLU"}


def t2_table():
    d = load("t2_stats.json")
    if not d:
        return
    body = [r"\begin{table}[t]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lrrr}", r"\toprule",
            r"& $n$ & meaning tags & symbolism tags \\",
            r"\midrule"]
    for t, r in d["accuracy"].items():
        def c(k):
            v = r[k]
            s = f"{100 * v['delta']:+.1f}"
            return (r"\textbf{" + s + "}") if v["sig"] else s
        body.append(f"{T2_NAME.get(t, t)} & {r['n']} & {c('dict_minus_untagged')} & "
                    f"{c('sym_minus_untagged')} \\\\")
    body += [r"\bottomrule", r"\end{tabular}",
             r"\caption{T2: the paper's proposed next step, run. Three Qwen3.5-9B arms trained "
             r"on the same 410,669 Arabic idiom-bearing documents for 0.63B tokens, differing "
             r"only in what is appended: nothing, the meaning tag of \S\ref{sec:data-corpus}, "
             r"or statements of what the matched idioms' entities symbolize in Arabic proverbs, "
             r"written from \dataname{} with the probe's own entities excluded. Accuracy points "
             r"against the untagged arm; bold: 95\% item bootstrap excludes zero. The meaning "
             r"tags reproduce the main study's signature; the symbolism tags teach neither "
             r"symbolism nor meaning.}",
             r"\label{tab:t2}", r"\end{table}"]
    w("t2.tex", "\n".join(body) + "\n")


def main():
    corpus_profile()
    margin_table()
    churn_table()
    selection_bias()
    entity_coverage()
    capability()
    taxonomy_table()
    english_retention()
    t1_table()
    t2_table()


if __name__ == "__main__":
    main()
