#!/usr/bin/env python3
"""Build the main-text figures of the paper from the project's result files.

    python code/make_figures.py

Inputs (all read-only):
  - docs/paper_stats/ci_report.json      per-task accuracies + paired bootstrap CIs
  - the four idiom KBs (en / zh / hi / ar)

Outputs: latex/figures/fig_entities.pdf, latex/figures/fig_cpt_effects.pdf
"""
import collections
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
OVERLEAF = "/home/jiaruil5/culture_pretrain/OverleafCultureInFigurativeLanguage"  # paper repo (LaTeX only)

REPO = "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage"
DATA = os.path.join(REPO, "culture/data")
# The Arabic KB is not on this filesystem; it is downloaded from the HF dataset
# Jerry9999/CultureInFigurativeLanguage (data/idioms/ar/). Override with AR_KB=...
AR_KB = os.environ.get("AR_KB", "/data/group_data/r3lit_culture_pretrain/culture/bidir/hf9b/data/idioms/ar/"
                       "idioms_merged_llm_formatted.jsonl")  # enriched KB from the HF repo
OUT = os.path.join(OVERLEAF, "latex", "figures")

ZH_FONT = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"  # LxgwWenKai in culture/data/fonts fails to subset
font_manager.fontManager.addfont(ZH_FONT)
ZH = font_manager.FontProperties(fname=ZH_FONT)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#B0B0B0", "axes.linewidth": 0.8,
    "xtick.color": "#555555", "ytick.color": "#333333", "pdf.fonttype": 42,
})
C_EN, C_ZH, C_HI, C_AR = "#6B7280", "#EC703E", "#27B381", "#327DD8"
C_SIG, C_NS = "#EC703E", "#9AA1AB"


# --------------------------------------------------------------------------- #
# Figure: what idioms talk about in each language
# --------------------------------------------------------------------------- #
KB = {
    "en": os.path.join(DATA, "idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl"),
    "zh": os.path.join(DATA, "idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl"),
    "hi": os.path.join(DATA, "hi_idioms/idioms_hi_llm_formatted.jsonl"),
    "ar": AR_KB,
}
# Dictionary slot fillers in English headwords ("put someone in their place").
# They are artefacts of lexicographic convention, not imagery.
EN_SLOTS = {"something", "someone", "thing", "person", "place", "way"}
# English glosses, for display only (the counts are computed on the native strings).
GLOSS = {
    "zh": {"心": "heart-mind", "人": "person", "天": "sky/heaven", "风": "wind", "地": "earth",
           "言": "speech", "马": "horse", "水": "water", "日": "sun/day", "目": "eye",
           "云": "cloud", "虎": "tiger", "口": "mouth", "山": "mountain", "龙": "dragon"},
    "hi": {"घर": "ghar (home)", "धन": "dhan (wealth)", "पानी": "pānī (water)", "हाथ": "hāth (hand)",
           "पेट": "peṭ (belly)", "स्त्री": "strī (woman)", "सिर": "sir (head)", "मुँह": "mũh (mouth)",
           "आदमी": "ādmī (man)", "मनुष्य": "manuṣya (human)", "चोर": "cor (thief)",
           "व्यक्ति": "vyakti (person)", "काम": "kām (work)", "कुत्ता": "kuttā (dog)",
           "माँ": "mā̃ (mother)", "राजा": "rājā (king)", "गाँव": "gā̃v (village)"},
    "ar": {"الله": "Allāh (God)", "لله": "Allāh (God)", "ناس": "nās (people)", "عين": "ʿayn (eye)", "كلب": "kalb (dog)",
           "جمل": "jamal (camel)", "حمار": "ḥimār (donkey)", "دار": "dār (home)",
           "قلب": "qalb (heart)", "باب": "bāb (door)", "ماء": "māʾ (water)", "راس": "raʾs (head)",
           "مال": "māl (wealth)", "ارض": "arḍ (land)", "نار": "nār (fire)", "بيت": "bayt (house)"},
}


def entity_rates(lang, k=12):
    norm = lambda e: e
    if lang == "ar":
        sys.path.insert(0, os.path.join(REPO, "src/culture/data_processing/ar_idioms"))
        from normalize import normalize_ar
        # strip the article, but never from Allah (الله -> "له" would be wrong)
        norm = lambda e: (lambda n: n if n in ("الله", "لله") else re.sub(r"^(ال|لل)", "", n))(normalize_ar(e))
    counts, n = collections.Counter(), 0
    for line in open(KB[lang], encoding="utf-8"):
        out = json.loads(line).get("output") or {}
        n += 1
        ents = out.get("entities") or []
        if isinstance(ents, str):
            continue
        ents = {norm(e.lower() if lang == "en" else e) for e in ents if e and e != "NAN"}
        counts.update(e for e in ents if e and not (lang == "en" and e in EN_SLOTS))
    return [(e, 1000.0 * c / n) for e, c in counts.most_common(k)], n


def fig_entities():
    panels = [("en", "English", C_EN), ("zh", "Chinese", C_ZH), ("hi", "Hindi", C_HI), ("ar", "Arabic", C_AR)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55))
    for ax, (lang, name, color) in zip(axes, panels):
        rows, n = entity_rates(lang)
        rows = rows[::-1]
        ys = range(len(rows))
        ax.barh(ys, [r for _, r in rows], color=color, height=0.72, edgecolor="white")
        ax.set_yticks(list(ys))
        if lang == "zh":
            ax.set_yticklabels([f"{e} {GLOSS['zh'][e]}" for e, _ in rows], fontfamily=["DejaVu Serif", ZH.get_name()], fontsize=8)
        elif lang == "en":
            ax.set_yticklabels([e for e, _ in rows], fontsize=8)
        else:
            ax.set_yticklabels([GLOSS[lang][e] for e, _ in rows], fontsize=7.5)
        for y, (_, r) in zip(ys, rows):
            ax.text(r + 0.4, y, f"{r:.1f}", va="center", fontsize=6.5, color="#555555")
        ax.set_title(f"{name} ($n$={n:,})", fontsize=8.5, loc="left")
        ax.set_xlim(0, max(r for _, r in rows) * 1.22)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["left"].set_visible(False)
    fig.supxlabel("idioms containing the entity, per 1,000 idioms", fontsize=8.5, y=0.06)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_entities.pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure: effect of idiom curation against the token-matched random control
# --------------------------------------------------------------------------- #
GROUPS = [
    ("Idiom meaning", [("ar", "kinayat_meaning", "Kinayat-Meaning (ar)"),
                       ("zh", "chengyu_bench", "Chengyu-Bench (zh)")]),
    ("Figurative\ninference", [("ar", "ar_figurative", "AR-Figurative (ar)"),
                               ("hi", "mabl", "MABL (hi)")]),
    ("Idiom cloze", [("ar", "kinayat_cloze", "Kinayat-Cloze (ar)"),
                     ("zh", "chid", "ChID (zh)")]),
    ("Culture", [("ar", "alyah", "Alyah (ar)"), ("ar", "dzirieval", "DziriEval (ar)"),
                 ("ar", "arabculture", "ArabCulture (ar)"), ("ar", "arabic_cultural_qa", "ArabicCulturalQA (ar)"),
                 ("ar", "global_piqa_ar", "Global-PIQA (ar)"), ("hi", "global_piqa", "Global-PIQA (hi)"),
                 ("zh", "ccpm", "CCPM (zh)")]),
    ("Regional\nknowledge", [("ar", "arabicmmlu", "ArabicMMLU (ar)"), ("hi", "milu", "MILU (hi)"),
                             ("zh", "cmmlu", "CMMLU (zh)")]),
]


def fig_cpt_effects():
    # v2 report (ArabCulture qid fix); zh Idiom-CPT-tags taken from v1 (records not on HF)
    rep = json.load(open(os.path.join(REPO, "docs/paper_stats/v2/ci_report.json")))
    _v1 = json.load(open(os.path.join(REPO, "docs/paper_stats/ci_report.json")))
    for _t, _r in _v1["zh"].items():
        if _t in rep["zh"] and "untagged" not in rep["zh"][_t]["per_run"] and "untagged" in _r["per_run"]:
            rep["zh"][_t]["per_run"]["untagged"] = _r["per_run"]["untagged"]
            rep["zh"][_t]["contrasts"]["cpt_vs_untagged"] = _r["contrasts"]["cpt_vs_untagged"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios": [1.3, 1]})

    # (a) forest plot: Idiom-CPT minus token-matched unfiltered CPT
    y, yticks, ylabels, seps = 0, [], [], []
    for gname, tasks in GROUPS:
        top = y
        for lang, task, label in tasks:
            c = rep[lang][task]["contrasts"]["cpt_vs_unfiltered"]
            d, (lo, hi) = 100 * c["delta"], [100 * v for v in c["ci95"]]
            col = C_SIG if c["sig_0.05"] else C_NS
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round")
            ax.plot([d], [y], "o", color=col, ms=4.2)
            ax.text(hi + 0.6, y, f"{d:+.1f}" + ("*" if c["sig_0.05"] else ""), va="center",
                    fontsize=6.8, color=col)
            yticks.append(y); ylabels.append(label); y += 1
        ax.text(38.5, y - 1, gname, fontsize=7.2, va="center", ha="right",
                color="#333333", style="italic")
        seps.append(y - 0.5); y += 0.35
    for s in seps[:-1]:
        ax.axhline(s + 0.17, color="#E3E3E3", lw=0.8)
    ax.axvline(0, color="#777777", lw=0.9, ls="--")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7.4)
    ax.invert_yaxis(); ax.set_xlim(-9, 39)
    ax.set_xticks([-5, 0, 5, 10, 15, 20, 25])
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("accuracy gain of Idiom-CPT over Random-CPT (points)", fontsize=8)
    ax.set_title("(a) Does idiom curation beat generic in-language text?", fontsize=8.5, loc="left")

    # (b) decomposition of the gap into document selection and meaning tags
    dec = [("ar", "kinayat_meaning", "Kinayat-\nMeaning"), ("zh", "chengyu_bench", "Chengyu-\nBench"),
           ("ar", "ar_figurative", "AR-\nFigurative"), ("ar", "alyah", "Alyah"),
           ("hi", "global_piqa", "Global-\nPIQA (hi)")]
    sel, tag, tagsig = [], [], []
    for lang, task, _ in dec:
        r = rep[lang][task]
        a = r["per_run"]
        sel.append(100 * (a["untagged"]["acc"] - a["unfiltered"]["acc"]))
        tag.append(100 * (a["cpt"]["acc"] - a["untagged"]["acc"]))
        tagsig.append(r["contrasts"]["cpt_vs_untagged"]["sig_0.05"])
    xs = range(len(dec))
    bx.bar(xs, sel, color="#327DD8", width=0.62, label="document selection", edgecolor="white")
    bx.bar(xs, tag, bottom=sel, color="#EC703E", width=0.62, label="meaning tags", edgecolor="white")
    for x, s, t, sig in zip(xs, sel, tag, tagsig):
        bx.text(x, s / 2, f"{s:.1f}", ha="center", va="center", fontsize=6.8, color="white")
        bx.text(x, s + t + 0.35, f"+{t:.1f}" + ("*" if sig else ""), ha="center", fontsize=6.8, color="#EC703E")
    bx.set_xticks(list(xs)); bx.set_xticklabels([d[2] for d in dec], fontsize=6.2)
    bx.set_ylabel("gain over Random-CPT (points)", fontsize=8)
    bx.tick_params(axis="x", length=0); bx.tick_params(axis="y", labelsize=7.5)
    bx.yaxis.grid(True, color="#E8E8E8", lw=0.8); bx.set_axisbelow(True)
    bx.legend(frameon=False, fontsize=7.2, loc="upper right")
    bx.set_title("(b) Where does the gain come from?", fontsize=8.5, loc="left")

    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(OUT, "fig_cpt_effects.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_entities()
    fig_cpt_effects()
    print("wrote figures to", os.path.abspath(OUT))
