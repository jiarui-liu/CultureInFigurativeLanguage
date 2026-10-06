"""Figures for Section 4 (cross-cultural idiom analysis).

Reads the committed analysis outputs under docs/paper_stats/analysis_v2/ and writes
PDFs into the Overleaf checkout. No LLM or embedding calls; pure plotting.

    PYTHONPATH=src python src/culture/analysis/v2/make_figures.py [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
STATS = REPO / "docs" / "paper_stats" / "analysis_v2"
DEFAULT_OUT = REPO.parent / "OverleafCultureInFigurativeLanguage" / "latex" / "figures"

# Palette matches main.tex (encolor/zhcolor/selcolor/tagcolor).
EN, ZH, HI, AR = "#1F5FA8", "#B04A2F", "#327DD8", "#EC703E"
LANG_COLOR = {"en": EN, "zh": ZH, "hi": HI, "ar": AR}
LANG_NAME = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}

TYPE_ORDER = [
    "body_mind",
    "artefact_household",
    "nature_cosmos",
    "animal",
    "kinship_social",
    "food_drink",
    "religion_supernatural",
    "occupation_economy",
    "abstract_other",
]
TYPE_LABEL = {
    # One head noun each; the caption gives the scope. "A & B" labels read as
    # codebook internals rather than as categories.
    "body_mind": "Body",            # body parts and the mind
    "artefact_household": "Objects",  # artefacts, tools, household things
    "nature_cosmos": "Nature",      # landscape, weather, sky
    "animal": "Animals",
    "kinship_social": "People",     # kin and social roles
    "food_drink": "Food",           # food and drink
    "religion_supernatural": "Religion",
    "occupation_economy": "Economy",  # money and work
    "abstract_other": "Abstract",
}

PAIRS = [
    ("en-zh", "English–Chinese"),
    ("en-hi", "English–Hindi"),
    ("en-ar", "English–Arabic"),
    ("zh-hi", "Chinese–Hindi"),
    ("zh-ar", "Chinese–Arabic"),
    ("hi-ar", "Hindi–Arabic"),
]


def _rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
        }
    )


def fig_typology(out: Path) -> None:
    """Heatmap of adjusted residuals, 9 semantic types x 4 languages."""
    rows = list(csv.DictReader((STATS / "entity_typology_shares.csv").open()))
    langs = ["en", "zh", "hi", "ar"]
    resid = np.zeros((len(TYPE_ORDER), len(langs)))
    share = np.zeros_like(resid)
    for r in rows:
        if r["type"] not in TYPE_ORDER or r["language"] not in langs:
            continue
        i, j = TYPE_ORDER.index(r["type"]), langs.index(r["language"])
        resid[i, j] = float(r["adjusted_residual"])
        share[i, j] = float(r["share"])

    # Languages as rows, types as columns, matching fig_typology_expanded.
    resid, share = resid.T, share.T

    fig, ax = plt.subplots(figsize=(3.4, 1.72))
    # Clip the scale: the "abstract & other" catch-all carries the two largest
    # residuals and would otherwise flatten every contrast of interest.
    lim = 40.0
    im = ax.imshow(resid, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")

    for i in range(len(langs)):
        for j in range(len(TYPE_ORDER)):
            strong = abs(resid[i, j]) > 0.55 * lim
            ax.text(j, i, f"{share[i, j] * 100:.1f}", ha="center", va="center",
                    fontsize=5.2, color="white" if strong else "#222222")

    ax.set_xticks(range(len(TYPE_ORDER)), [TYPE_LABEL[t] for t in TYPE_ORDER],
                  fontsize=5.8, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(langs)), [LANG_NAME[l] for l in langs], fontsize=6.6)
    ax.set_xticks(np.arange(-0.5, len(TYPE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(langs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015, extend="both")
    cb.set_label("Adj. residual", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4)
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "fig_typology.pdf", bbox_inches="tight")
    plt.close(fig)


DIV_PAIRS = [("en-zh", "multi", "English–Chinese"), ("en-hi", "multi", "English–Hindi"),
             ("en-ar", "multi", "English–Arabic"), ("zh-hi", "cross", "Chinese–Hindi"),
             ("zh-ar", "cross", "Chinese–Arabic"), ("hi-ar", "cross", "Hindi–Arabic")]


def _entity_types():
    """English-anchored entity -> semantic type, same taxonomy as Figure fig:typology."""
    types = {}
    f = STATS / "entity_typology_labels_en.json"
    if f.exists():
        for e, o in json.loads(f.read_text()).items():
            types.setdefault(e, o["type"])
    for l in ("zh", "hi", "ar"):
        tr, la = (STATS / f"entity_translations_{l}_en.json",
                  STATS / f"entity_typology_labels_{l}.json")
        if not (tr.exists() and la.exists()):
            continue
        lab = json.loads(la.read_text())
        for e, t in json.loads(tr.read_text()).items():
            if t and t.get("en") and t["en"] != "none" and e in lab:
                types.setdefault(t["en"], lab[e]["type"])
    return types


N_WAY = 20


def _match_score(percentiles):
    """Expected accuracy of a 20-way match.

    An entity's rank percentile p is the share of wrong candidates at least as close
    as the true counterpart, so the true one beats a share (1-p). Drawing N_WAY-1
    distractors at random, it is the closest of all of them with probability
    (1-p)^(N_WAY-1). Fixing the number of candidates makes the score comparable
    across pairs, which a raw top-1 rate is not (pools range from 76 to 156).
    """
    return np.asarray([(1.0 - q) ** (N_WAY - 1) for q in percentiles], float)


def _boot_ci(x, reps=5000, seed=0):
    x = np.asarray(x, float)
    if x.size < 5:
        return None
    rng = np.random.default_rng(seed)
    s = [float(np.mean(rng.choice(x, x.size, replace=True))) for _ in range(reps)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def fig_divergence(out: Path) -> None:
    """Which kinds of entity diverge, seen from each language in turn.

    The measurement is per language *pair*; a panel for language L pools the three
    pairs L takes part in, so each entity-pair observation appears in two panels.
    The pair axis is flat (every pair lands at 42-63%), so the informative axis is
    the entity type, which varies about threefold.
    """
    types = _entity_types()
    per_lang = defaultdict(lambda: defaultdict(list))
    pool = defaultdict(list)
    for key, fam, _ in DIV_PAIRS:
        f = STATS / f"entity_divergence_{fam}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())["pairs"].get(key, {}).get(
            "gloss", {}).get("per_entity")
        la, lb = key.split("-")
        for r in (rec or []):
            t = types.get(r["entity_en"])
            if not t:
                continue
            per_lang[la][t].append(r["percentile"])
            per_lang[lb][t].append(r["percentile"])
            pool[t].append(r["percentile"])

    MIN_N = 10
    keep = [t for t, v in pool.items() if len(v) >= 15]
    # One shared ordering (pooled) so the panels are comparable at a glance.
    order = sorted(keep, key=lambda t: -float(_match_score(pool[t]).mean()))

    langs = [("en", "English"), ("zh", "Chinese"), ("hi", "Hindi"), ("ar", "Arabic")]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.75), sharey=True)
    ys = np.arange(len(order))

    for ax, (lk, lname) in zip(axes, langs):
        ax.axvline(100.0 / N_WAY, ls=(0, (3, 2.5)), lw=0.9, color="#B04A2F",
                   zorder=2)
        for y, t in zip(ys, order):
            v = per_lang[lk].get(t, [])
            if len(v) < MIN_N:
                ax.text(36, y, f"n={len(v)}", fontsize=5.3, color="#BBBBBB",
                        ha="center", va="center", style="italic")
                continue
            hit = _match_score(v)
            ci = _boot_ci(hit)
            if ci:
                ax.hlines(y, 100 * ci[0], 100 * ci[1], color="#55606E", lw=1.1,
                          zorder=3)
            ax.plot(100.0 * float(np.mean(hit)), y, "o", ms=4.6, color=ZH,
                    mec="white",
                    mew=0.7, zorder=4)
        ax.set_xlim(0, 70)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xticks([0, 25, 50], ["0", "25", "50"], fontsize=6.8)
        ax.set_title(lname, loc="center", fontsize=8.2, pad=4)
        ax.grid(axis="x", lw=0.4, color="#E2E2E2", zorder=0)
        ax.set_axisbelow(True)
        ax.invert_yaxis()

    axes[0].set_yticks(ys, [TYPE_LABEL.get(t, t) for t in order], fontsize=7.0)
    axes[0].text(6.5, len(order) - 0.55, "chance", fontsize=5.8, color="#B04A2F",
                 va="center")
    fig.supxlabel("the true counterpart is the best of 20 candidates "
                  "(% of entities)", fontsize=7.8, y=-0.02)
    fig.tight_layout()
    fig.savefig(out / "fig_divergence.pdf", bbox_inches="tight")
    plt.close(fig)


SUBTYPE_LABEL = {
    "action_conflict": "Action",        # acting, striving, conflict
    "quantity_space": "Quantity",       # amount, size, place, direction
    "morality_truth": "Morality",       # right and wrong, truth and deceit
    "speech_name": "Speech",            # speaking, names, reputation
    "time_change": "Time",              # time and change
    "life_death_fate": "Fate",          # life, death, fortune
    "residual_other": "Unclassified",
}


def fig_typology_expanded(out: Path) -> None:
    """fig_typology with the abstract/other catch-all split into its seven subtypes.

    Same encoding as fig_typology on purpose: the reader learns one picture, not two.
    """
    f = STATS / "entity_subtypology_shares.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    rows = list(csv.DictReader(f.open()))
    langs = ["en", "zh", "hi", "ar"]
    level, share, resid = {}, {}, {}
    for r in rows:
        level[r["type"]] = r["level"]
        share[(r["language"], r["type"])] = float(r["share_of_all_mentions"])
        resid[(r["language"], r["type"])] = float(r["adjusted_residual"])
    concrete = [t for t in TYPE_ORDER if level.get(t) == "concrete"]
    # Subtypes by size, but the residual catch-all always last: it is a leftover bin,
    # not a category, and it is also the least reliable (38% second-annotator
    # agreement, against 62-89% for the others).
    subs = sorted([t for t, l in level.items()
                   if l == "subtype" and t != "residual_other"],
                  key=lambda t: -max(share[(l, t)] for l in langs))
    if "residual_other" in level:
        subs.append("residual_other")
    order = concrete + subs

    # Transposed relative to fig_typology: 15 types would make a tall, thin strip in
    # a column, so types run along x and the four languages are the rows.
    M = np.array([[resid[(l, t)] for t in order] for l in langs])
    S = np.array([[100 * share[(l, t)] for t in order] for l in langs])

    fig, ax = plt.subplots(figsize=(7.1, 1.95))
    lim = 40.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(langs)):
        for j in range(len(order)):
            strong = abs(M[i, j]) > 0.55 * lim
            ax.text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", fontsize=6.0,
                    color="white" if strong else "#222222")

    labels = [TYPE_LABEL.get(t, SUBTYPE_LABEL.get(t, t)) for t in order]
    ax.set_xticks(range(len(order)), labels, fontsize=6.4, rotation=38, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(len(langs)), [LANG_NAME[l] for l in langs], fontsize=7.4)
    ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(langs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.012, extend="both")
    cb.set_label("Adjusted residual", fontsize=6.6)
    cb.ax.tick_params(labelsize=6.0)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_typology_expanded.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_learning(out: Path) -> None:
    """(a) seen/unseen gap vs the untrained base; (b) accuracy against corpus exposure."""
    sel = json.loads((STATS / "selective_learning.json").read_text())
    dose = json.loads((STATS / "exposure_dose_response.json").read_text())

    fig = plt.figure(figsize=(7.1, 4.75))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.28)
    axa = fig.add_subplot(gs[0, :])
    dose_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # ---- (a) raw seen/unseen gap per arm, base as the reference -----------
    arms = [("base", "Base"), ("unfiltered", "Random"), ("untagged", "Idiom$-$tags"),
            ("cpt", "Idiom-CPT")]
    colors = {"base": "#9AA3AE", "unfiltered": HI, "untagged": AR, "cpt": ZH}
    langs = [("ar", "Arabic"), ("hi", "Hindi"), ("zh", "Chinese")]

    width, gap = 0.19, 0.22
    # Every arm now has per-item records from our own evaluation, so the panel uses one
    # consistent source and the difference-in-differences carry bootstrap intervals.
    LOCAL_KEY = {"base": "base", "unfiltered": "random", "untagged": "idiom_untagged",
                 "cpt": "idiom_cpt"}
    for li, (lk, lname) in enumerate(langs):
        L = sel["languages"].get(lk, {})
        local = L.get("arms", {})
        gaps, did, dci = {}, {}, {}
        for pub_key, loc_key in LOCAL_KEY.items():
            e = local.get(loc_key)
            if e is None:
                continue
            gaps[pub_key] = e["gap"]
            if e.get("gap_minus_base_gap") is not None:
                did[pub_key] = e["gap_minus_base_gap"]
                dci[pub_key] = e.get("gap_minus_base_gap_ci95")
        for ai, (ak, alabel) in enumerate(arms):
            if ak not in gaps:
                continue
            x = li + (ai - 1.5) * width
            axa.bar(x, gaps[ak], width * 0.92, color=colors[ak], zorder=3,
                    label=alabel if li == 0 else None)
            if ak == "cpt" and ak in did:
                ci = dci.get(ak)
                txt = f"{did[ak]:+.0f}"
                if ci:
                    txt += f"\n[{ci[0]:.0f}, {ci[1]:.0f}]"
                axa.annotate(txt, (x, gaps[ak]), textcoords="offset points",
                             xytext=(0, 3), ha="center", fontsize=5.8, color=ZH,
                             linespacing=1.15)
        # base reference line across the group
        if "base" in gaps:
            axa.plot([li - 2 * width, li + 2 * width], [gaps["base"]] * 2,
                     ls=":", lw=0.9, color="#444444", zorder=4)

    axa.axhline(0, lw=0.7, color="#333333", zorder=2)
    axa.set_xticks(range(len(langs)), [n for _, n in langs])
    axa.set_ylabel("IdiomAtlas-MC gap:\nseen $-$ unseen (points)")
    axa.set_title("(a) Only Arabic and Hindi gaps are training-made", loc="left", pad=5)
    axa.legend(frameon=False, ncol=2, loc="upper right", handlelength=1.1,
               columnspacing=1.0, fontsize=6.8)
    axa.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
    axa.set_axisbelow(True)
    axa.set_ylim(-6, 52)

    # ---- (b-d) accuracy vs exposure, one panel per language ---------------
    series = [("base", "Base", "#9AA3AE", "o"), ("random", "Random", HI, "s"),
              ("idiom_cpt", "Idiom-CPT", "#8C2F1B", "v"),
              ("idiom_untagged", "Idiom$-$tags", AR, "^"),
              ("culture", "Culture", ZH, "D"),
              ("culture_notes", "Culture+notes", "#E8A33D", "P")]
    ALL_ARMS = {k: lab for k, lab, _, _ in series}
    panel_tag = {"ar": "(b)", "hi": "(c)", "zh": "(d)"}
    for ax, (lk, lname) in zip(dose_axes, [("ar", "Arabic"), ("hi", "Hindi"),
                                           ("zh", "Chinese")]):
        L = dose.get("languages", {}).get(lk)
        if not L:
            ax.set_visible(False)
            continue
        bins = L["by_exposure_bin"]
        labels = [k for k in bins if bins[k]["n"] >= 10]
        xs = np.arange(len(labels))
        for key, lab, col, mk in series:
            if not any(key in bins[k] for k in labels):
                continue
            ys, lo, hi_ = [], [], []
            for k in labels:
                v = bins[k].get(key)
                if v is None:
                    ys.append(np.nan); lo.append(0); hi_.append(0); continue
                ys.append(v["mean"] * 100)
                ci = v.get("ci95") or [v["mean"], v["mean"]]
                lo.append((v["mean"] - ci[0]) * 100)
                hi_.append((ci[1] - v["mean"]) * 100)
            ax.errorbar(xs, ys, yerr=[lo, hi_], marker=mk, ms=3.0, lw=1.1,
                        capsize=1.6, color=col, label=lab, elinewidth=0.6, zorder=3)
        # Say which arms have no IdiomAtlas-MC evaluation for this language, so a
        # missing line is never mistaken for a missing effect.
        absent = [lab for k, lab in ALL_ARMS.items()
                  if not any(k in bins[b] for b in labels)]
        if absent:
            import textwrap
            msg = textwrap.fill("not evaluated here: " + ", ".join(absent), 30)
            ax.text(0.03, 0.03, msg, transform=ax.transAxes, fontsize=5.2,
                    color="#999999", va="bottom", linespacing=1.25)
        ax.set_xticks(xs, labels, rotation=40, ha="right", fontsize=6.2)
        ax.set_title(f"{panel_tag[lk]} {lname}", loc="left", pad=4, fontsize=8)
        ax.grid(axis="y", lw=0.4, color="#DDDDDD", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=6.8)
    dose_axes[0].set_ylabel("IdiomAtlas-MC seen\naccuracy (%)")
    dose_axes[1].set_xlabel("times the idiom occurs in the training corpus")
    h, l = dose_axes[-1].get_legend_handles_labels()
    dose_axes[-1].legend(h, l, frameon=False, fontsize=6.2, loc="lower right",
                         handlelength=1.2, labelspacing=0.2, borderpad=0.15)

    fig.tight_layout()
    fig.savefig(out / "fig_learning.pdf", bbox_inches="tight")
    plt.close(fig)


# Plain-language names for the six taxonomy categories, ordered from the "what things
# mean" end to the "what is factually so" end.
LAYER_CATS = [
    ("symbolic_evaluative", "Symbolism & judgement", "#B04A2F"),
    ("social_norm_relation", "Social norms & roles", "#E08A5F"),
    ("generic_pragmatic", "Culture-neutral sense", "#F0D9C4"),
    ("material_practice", "Customs & daily life", "#BFD3E6"),
    ("factual_knowledge", "Named facts", "#5B8FC9"),
    ("language_form", "Wording & dialect", "#1F3F6B"),
]

# One block per language: the idiom corpus against that language's culture benchmarks,
# pooled. English is excluded (it is analysis-only and has no culture benchmark here).
LAYER_GROUPS = [
    ("Hindi", [("Idioms", ["Idioms (hi)"]),
               ("Culture benchmarks", ["Global-PIQA (hi)"]),
               ("Culture pretraining data", ["Culture corpus (hi)"])]),
    ("Chinese", [("Idioms", ["Idioms (zh)"]),
                 ("Culture benchmarks", ["CCPM (zh)"]),
                 ("Culture pretraining data", ["Culture corpus (zh)"])]),
    ("Arabic", [("Idioms", ["Idioms (ar)"]),
                ("Culture benchmarks", ["ArabCulture", "Alyah", "DziriEval",
                                        "ArabicCulturalQA", "Global-PIQA (ar)"]),
                ("Culture pretraining data", ["Culture corpus (ar)"])]),
]
LOW_AGREE = 0.45


def _pool(by_source, agree, names):
    """Pool several sources by item count; returns (shares, n, min agreement)."""
    tot = Counter()
    n = 0
    ags = []
    for nm in names:
        v = by_source.get(nm)
        if not v:
            continue
        for k, c in v["counts"].items():
            tot[k] += c
        n += v["n"]
        a = agree.get(nm, {}).get("raw_agreement")
        if a is not None:
            ags.append(a)
    if not n:
        return None, 0, None
    return {k: tot.get(k, 0) / n for k, _, _ in LAYER_CATS}, n, (min(ags) if ags else None)


def fig_layer(out: Path) -> None:
    """Idioms vs that language's culture benchmarks, one block per language."""
    src = STATS / "culture_layer_taxonomy.json"
    if not src.exists():
        src = STATS / "culture_layer_taxonomy_preview.json"
    blob = json.loads(src.read_text())
    by_source = blob["by_source"]
    agree = blob.get("agreement_by_source", {})

    # Build a flat row list: a bold header per language, then its bars.
    rows = []  # (kind, text, shares, n, agreement)
    for lang, entries in LAYER_GROUPS:
        block = []
        for label, names in entries:
            shares, n, ag = _pool(by_source, agree, names)
            if shares is not None:
                block.append(("bar", label, shares, n, ag))
        if block:
            rows.append(("header", lang, None, 0, None))
            rows.extend(block)

    fig, ax = plt.subplots(figsize=(7.1, 0.315 * len(rows) + 1.30))
    ys, y = [], 0.0
    for kind, *_ in rows:
        if kind == "header" and ys:
            y += 0.45
        ys.append(y)
        y += 1.0
    ys = np.array(ys)
    ys = ys.max() - ys  # draw top-down

    for yy, (kind, text, shares, n, ag) in zip(ys, rows):
        if kind == "header":
            ax.text(-1.5, yy, text, ha="right", va="center", fontsize=8.2,
                    fontweight="bold")
            continue
        left = 0.0
        for key, _, col in LAYER_CATS:
            w = shares.get(key, 0.0) * 100
            if w <= 0:
                continue
            ax.barh(yy, w, left=left, color=col, height=0.74, zorder=3,
                    edgecolor="white", linewidth=0.5)
            if w >= 8:
                ax.text(left + w / 2, yy, f"{w:.0f}", ha="center", va="center",
                        fontsize=6.4,
                        color="white" if col in ("#B04A2F", "#1F3F6B", "#5B8FC9")
                        else "#333333")
            left += w
        dag = " $^\\dagger$" if (ag is not None and ag < LOW_AGREE) else ""
        ax.text(-1.5, yy, text + dag, ha="right", va="center", fontsize=7.3)
        ax.text(101.5, yy, f"{n:,}", ha="left", va="center", fontsize=6.3,
                color="#666666")

    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 0.7)
    ax.set_xlabel("% of the source's sampled items")
    ax.text(101.5, ys.max() + 0.62, "items", ha="left", va="center",
            fontsize=6.3, color="#666666")
    handles = [Line2D([], [], color=c, lw=5, label=l) for _, l, c in LAYER_CATS]
    ax.legend(handles=handles, frameon=False, ncol=3, fontsize=6.9,
              loc="lower center", bbox_to_anchor=(0.5, 1.005), handlelength=1.3,
              columnspacing=1.2)
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_layer.pdf", bbox_inches="tight")
    plt.close(fig)


# Benchmark groups, renamed in plain language and ordered from "what the idiom means"
# to "general knowledge". The first four probe the symbolic/figurative layer that
# fig_layer shows idioms carry; "Culture benchmarks" is the facts-and-practices layer.
# One benchmark group per column, so a column means the same thing in every row.
MATRIX_COLS = [
    ("idiom_meaning", "Idiom meaning\n(shown)"),
    ("idiom_unseen", "Idiom meaning\n(unseen)"),
    ("figurative", "Figurative\ninference"),
    ("cloze", "Idiom\ncloze"),
    ("symbolism", "Symbolism\nprobe"),
    ("culture", "Culture\nbenchmarks"),
    ("regional", "Regional\nknowledge"),
]
# All four trained arms, grouped by direction.
MATRIX_ROWS = [
    ("idiom_tagged", "Idiom-CPT"),
    ("idiom_untagged", "Idiom$-$tags"),
    ("culture", "Culture"),
    ("culture_notes", "Culture+notes"),
]
MATRIX_LANGS = [("ar", "Arabic"), ("hi", "Hindi"), ("zh", "Chinese")]


def fig_transfer_matrix(out: Path) -> None:
    """2B transfer matrix: every language, every trained arm, every benchmark group."""
    f = REPO / "docs" / "paper_stats" / "v2" / "bidir_2b_i.json"
    rec = json.loads(f.read_text())["per_task"]
    # Which (lang, group) combinations have a benchmark at all? Anything absent here is
    # a benchmark that does not exist for that language, not a result we are hiding.
    exists = {(r["lang"], r["group"]) for r in rec}

    rows = []
    for lang, lname in MATRIX_LANGS:
        for arm, rlabel in MATRIX_ROWS:
            rows.append((lang, lname, arm, rlabel))

    nr, nc = len(rows), len(MATRIX_COLS)
    M = np.full((nr, nc), np.nan)
    info = {}
    for ri, (lang, _, arm, _) in enumerate(rows):
        for ci, (grp, _) in enumerate(MATRIX_COLS):
            hits = [r for r in rec if r["lang"] == lang and r["arm"] == arm
                    and r["group"] == grp]
            if not hits:
                continue
            M[ri, ci] = float(np.mean([h["delta"] for h in hits])) * 100
            info[(ri, ci)] = sum(
                1 for h in hits
                if h.get("p_holm_group") is not None and h["p_holm_group"] < 0.05)

    fig, ax = plt.subplots(figsize=(6.9, 0.36 * nr + 1.7))
    lim = max(float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0, 1.0)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")

    for ri, (lang, _, arm, _) in enumerate(rows):
        for ci, (grp, _) in enumerate(MATRIX_COLS):
            if (ri, ci) in info:
                d, sig = M[ri, ci], info[(ri, ci)]
                ax.text(ci, ri, f"{d:+.1f}{'*' if sig else ''}", ha="center",
                        va="center", fontsize=6.8,
                        color="white" if abs(d) > 0.55 * lim else "#222222")
            else:
                # Distinguish "no benchmark exists" from a value of zero.
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                           facecolor="#EDEDED", edgecolor="white",
                                           linewidth=1.0, hatch="///", zorder=2))
                ax.text(ci, ri, "no\nbenchmark", ha="center", va="center",
                        fontsize=5.2, color="#888888", zorder=3)

    ax.set_xticks(range(nc), [lab for _, lab in MATRIX_COLS], fontsize=6.8)
    ax.set_yticks(range(nr), [rl for _, _, _, rl in rows], fontsize=7.0)
    for i, (lang, lname) in enumerate(MATRIX_LANGS):
        idx = [j for j, r in enumerate(rows) if r[0] == lang]
        ax.text(-1.9, float(np.mean(idx)), lname, ha="right", va="center",
                fontsize=8.2, fontweight="bold")
        if i:
            ax.axhline(min(idx) - 0.5, color="#444444", lw=1.1)
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("accuracy change vs. the token-matched control (points)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_transfer_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _rc()
    made = []
    for name, fn in (("fig_typology.pdf", fig_typology),
                     ("fig_typology_expanded.pdf", fig_typology_expanded),
                     ("fig_learning.pdf", fig_learning),
                     ("fig_layer.pdf", fig_layer),
                     ("fig_transfer_matrix.pdf", fig_transfer_matrix)):
        try:
            fn(args.out)
            made.append(name)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
    print(f"wrote {', '.join(made)} -> {args.out}")


if __name__ == "__main__":
    main()
