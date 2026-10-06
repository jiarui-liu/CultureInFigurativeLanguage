#!/usr/bin/env python3
"""Build the main-text figures of the paper from the project's result files.

    python code/make_figures.py

Inputs (all read-only):
  - docs/paper_stats/ci_report.json      per-task accuracies + paired bootstrap CIs
  - docs/paper_stats/analysis_v2/        entity typology shares + bootstrap CIs
  - the four idiom KBs (en / zh / hi / ar)

Outputs: latex/figures/fig_entities.pdf, latex/figures/fig_cpt_effects.pdf
"""
import collections
import json
import os
import re

import numpy as np
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from nativetext import ShapedFont, native_label

HERE = os.path.dirname(os.path.abspath(__file__))
# Defaults are the author's checkout; override with the env vars on other machines.
REPO = os.environ.get("CULTURE_REPO") or os.path.normpath(os.path.join(HERE, "..", ".."))
OVERLEAF = os.environ.get("OVERLEAF_REPO") or os.path.join(
    os.path.dirname(REPO), "OverleafCultureInFigurativeLanguage")  # paper repo (LaTeX only)

DATA = os.path.join(REPO, "culture/data")
STATS = os.path.join(REPO, "docs/paper_stats/analysis_v2")
# The Arabic KB may not be on this filesystem; it is downloaded from the HF dataset
# Jerry9999/CultureInFigurativeLanguage (data/idioms/ar/). Override with AR_KB=...
AR_KB = os.environ.get("AR_KB", os.path.join(DATA, "idioms/ar/idioms_merged_llm_formatted.jsonl"))
OUT = os.path.join(OVERLEAF, "latex", "figures")

FONTS = os.path.join(DATA, "fonts")


def _first_font(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of these fonts exist: {paths}")


# CJK needs no shaping, so Chinese still goes through matplotlib's own text path.
ZH_FONT = _first_font(
    "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    os.path.join(FONTS, "DroidSansFallbackFull.ttf"),
)  # LxgwWenKai in culture/data/fonts fails to subset
font_manager.fontManager.addfont(ZH_FONT)
ZH = font_manager.FontProperties(fname=ZH_FONT)
# Devanagari and Arabic do need shaping; see nativetext.py.
HI_FONT = ShapedFont(_first_font(os.path.join(FONTS, "NotoSerifDevanagari.ttf")))
AR_FONT = ShapedFont(_first_font(os.path.join(FONTS, "NotoNaskhArabic.ttf")))

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#B0B0B0", "axes.linewidth": 0.8,
    "xtick.color": "#555555", "ytick.color": "#333333", "pdf.fonttype": 42,
})
C_EN, C_ZH, C_HI, C_AR = "#6D4AA8", "#D9541E", "#1E9E6A", "#2D6FD6"
C_SIG, C_NS = "#EC703E", "#9AA1AB"
LANG_COLOR = {"en": C_EN, "zh": C_ZH, "hi": C_HI, "ar": C_AR}
LANG_NAME = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
# Marker shape is a second channel on top of hue: the green/orange pair sits in the
# 6-8 CVD separation band, and it keeps the figure readable in greyscale print.
LANG_MARKER = {"en": "o", "zh": "s", "hi": "^", "ar": "D"}
# Equal optical area: a triangle and a diamond enclose less ink than a disc or
# square at the same nominal markersize.
LANG_MS = {"en": 3.9, "zh": 3.5, "hi": 4.4, "ar": 3.6}
INK, INK_MUTED, RULE = "#222222", "#6B7280", "#E8E8E8"


# --------------------------------------------------------------------------- #
# Figure: what idioms talk about in each language
# --------------------------------------------------------------------------- #
KB = {
    "en": os.path.join(DATA, "idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl"),
    "zh": os.path.join(DATA, "idioms/zh/idioms_merged_llm_formatted_figurative_only.jsonl"),
    "hi": _first_font(os.path.join(DATA, "hi_idioms/idioms_hi_llm_formatted.jsonl"),
                      os.path.join(DATA, "idioms/hi/idioms_merged_llm_formatted_figurative_only.jsonl")),
    "ar": AR_KB,
}
# Dictionary slot fillers in English headwords ("put someone in their place").
# They are artefacts of lexicographic convention, not imagery.
EN_SLOTS = {"something", "someone", "thing", "person", "place", "way"}
# English glosses, for display only (the counts are computed on the native strings).
# The native string is what a Hindi or Arabic reader actually reads, so it is what
# the label shows; the gloss sits beside it for everyone else.
GLOSS = {
    "zh": {"心": "heart-mind", "人": "person", "天": "sky/heaven", "风": "wind", "地": "earth",
           "言": "speech", "马": "horse", "水": "water", "日": "sun/day", "目": "eye",
           "云": "cloud", "虎": "tiger", "口": "mouth", "山": "mountain", "龙": "dragon"},
    "hi": {"घर": "home", "धन": "wealth", "पानी": "water", "हाथ": "hand",
           "पेट": "belly", "स्त्री": "woman", "सिर": "head", "मुँह": "mouth",
           "आदमी": "man", "मनुष्य": "human", "चोर": "thief",
           "व्यक्ति": "person", "काम": "work", "कुत्ता": "dog",
           "माँ": "mother", "राजा": "king", "गाँव": "village"},
    "ar": {"الله": "God", "لله": "God", "ناس": "people", "عين": "eye", "كلب": "dog",
           "جمل": "camel", "حمار": "donkey", "دار": "home",
           "قلب": "heart", "باب": "door", "ماء": "water", "راس": "head",
           "مال": "wealth", "ارض": "land", "نار": "fire", "بيت": "house"},
}
# Entity keys are post-normalisation; show the citation form instead where they differ.
NATIVE_DISPLAY = {"لله": "الله", "راس": "رأس", "ماء": "ماء"}
SHAPED_FONT = {"hi": HI_FONT, "ar": AR_FONT}
# Noto Serif Devanagari and Noto Naskh Arabic sit low in the em box; these sizes
# make the native word read at roughly the same optical size as the 7.5pt gloss.
NATIVE_SIZE = {"hi": 8.5, "ar": 9.0}


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
    panels = [("en", C_EN), ("zh", C_ZH), ("hi", C_HI), ("ar", C_AR)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7))
    native_rows = []  # (ax, lang, y, native string), placed after the glosses are measured
    for ax, (lang, color) in zip(axes, panels):
        rows, n = entity_rates(lang)
        rows = rows[::-1]
        ys = range(len(rows))
        ax.barh(ys, [r for _, r in rows], color=color, height=0.72, edgecolor="white")
        ax.set_yticks(list(ys))
        if lang == "en":
            ax.set_yticklabels([e for e, _ in rows], fontsize=8)
        elif lang == "zh":
            # CJK renders correctly as ordinary text, so keep it in one label.
            ax.set_yticklabels([f"{e} {GLOSS['zh'][e]}" for e, _ in rows],
                               fontfamily=["DejaVu Serif", ZH.get_name()], fontsize=8)
        else:
            ax.set_yticklabels([GLOSS[lang][e] for e, _ in rows], fontsize=7.5)
            native_rows += [(ax, lang, y, NATIVE_DISPLAY.get(e, e)) for y, (e, _) in zip(ys, rows)]
        for y, (_, r) in zip(ys, rows):
            ax.text(r + 0.4, y, f"{r:.1f}", va="center", fontsize=6.5, color="#555555")
        ax.set_title(f"{LANG_NAME[lang]} ($n$={n:,})", fontsize=8.5, loc="left")
        ax.set_xlim(0, max(r for _, r in rows) * 1.22)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines["left"].set_visible(False)

    # The native word goes to the left of the gloss column, which means we first
    # have to know how wide the widest gloss is.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pad = {}
    for ax, lang, _, _ in native_rows:
        w = max(t.get_window_extent(rend).width for t in ax.get_yticklabels())
        pad[ax] = -(w * 72.0 / fig.dpi + ax.yaxis.majorTicks[0].get_pad() + 3.0)
    for ax, lang, y, native in native_rows:
        native_label(ax, native, SHAPED_FONT[lang], NATIVE_SIZE[lang], 0, y,
                     xycoords=("axes fraction", "data"), ha="right", dx=pad[ax],
                     color="#222222")

    fig.supxlabel("idioms containing the entity, per 1,000 idioms", fontsize=8.5, y=0.06)
    fig.tight_layout(w_pad=0.6, rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_entities.pdf"), bbox_inches="tight")
    plt.close(fig)


# The semantic-type figures were removed: analysis/v2/make_figures.py::fig_typology and
# ::fig_typology_expanded show the same type x language shares as a heatmap coloured by the
# chi-square adjusted residual, which is what latex/figures/fig_entity_types_expanded.tex
# now includes.



# --------------------------------------------------------------------------- #
# Figure: effect of idiom curation against the token-matched random control
# --------------------------------------------------------------------------- #
GROUPS = [
    ("Idiom meaning", [("ar", "kinayat_meaning", "Kinayat-Meaning (ar)"),
                       ("zh", "chengyu_bench", "Chengyu-Bench (zh)$^\\dagger$")]),
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


def _first_dir(*paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    raise FileNotFoundError(f"none of these 9B eval roots exist: {paths}")


# Per-item 9B records, laid out as <root>/<lang>/<run>/<task>.json. The babel path is the
# author's; the H100 box keeps the same tree under lustre. Override with CULTURE_HF9B=...
HF9B = os.environ.get("CULTURE_HF9B") or _first_dir(
    "/data/group_data/r3lit_culture_pretrain/culture/bidir/hf9b/eval",
    "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval",
)


def calibrated_contrast(lang, task, a="cpt", b="unfiltered", n=10000):
    """Paired contrast after removing each model's label prior (two fixed-label options)."""
    def corr(run):
        rs = json.load(open(os.path.join(HF9B, lang, run, task + ".json")))["records"]
        lp = np.array([json.loads(r["logprobs_norm"]) if isinstance(r["logprobs_norm"], str)
                       else r["logprobs_norm"] for r in rs])
        d = lp[:, 1] - lp[:, 0]
        return {r["qid"]: float(p == int(r["gold"])) for r, p in zip(rs, (d - np.median(d) > 0).astype(int))}
    A, B = corr(a), corr(b)
    q = sorted(set(A) & set(B))
    x = np.array([A[k] - B[k] for k in q])
    bs = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": x.mean(), "ci95": [lo, hi], "sig_0.05": bool(lo > 0 or hi < 0)}


CALIBRATED = {("zh", "chengyu_bench")}


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
            c = (calibrated_contrast(lang, task) if (lang, task) in CALIBRATED
                 else rep[lang][task]["contrasts"]["cpt_vs_unfiltered"])
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
    dec = [("ar", "kinayat_meaning", "Kinayat-\nMeaning"),
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
