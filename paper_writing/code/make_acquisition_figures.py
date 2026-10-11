#!/usr/bin/env python3
"""Generate the main acquisition and symbolism figures for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_writing" / "figures"
BLUE = "#4F81B8"
ORANGE = "#D68152"
GREY = "#B8BEC7"
INK = "#30343B"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 17,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )


def asymmetric(values, intervals):
    values = np.asarray(values, float)
    lo = np.asarray([x[0] for x in intervals], float)
    hi = np.asarray([x[1] for x in intervals], float)
    return np.vstack((values - lo, hi - values))


def annotate(ax, bars, fmt="{:+.1f}"):
    for bar in bars:
        value = bar.get_height()
        offset = 0.8 if value >= 0 else -1.1
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, fmt.format(value), ha="center", va=va, fontsize=14)


def make_acquisition() -> None:
    idiom = json.loads((ROOT / "docs/paper_stats/v2/idiomatlas_9b.json").read_text())
    ci = json.loads((ROOT / "docs/paper_stats/v2/ci_report.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.3), gridspec_kw={"width_ratios": [1.0, 1.15, 1.25]})

    # Panel a: learning meanings in all three languages.
    langs = ["Arabic", "Hindi", "Chinese"]
    keys = ["ar_seen", "hi_seen", "zh_seen"]
    values = [idiom[k]["cpt-unfiltered"][0] for k in keys]
    intervals = [idiom[k]["cpt-unfiltered"][1] for k in keys]
    x = np.arange(3)
    bars = axes[0].bar(x, values, color=ORANGE, edgecolor=INK, linewidth=0.8, hatch="//", width=0.64, yerr=asymmetric(values, intervals), capsize=4)
    axes[0].set_xticks(x, langs)
    axes[0].set_ylabel("Gain over Random-CPT (points)")
    axes[0].set_ylim(0, 33)
    axes[0].axhline(0, color=INK, linewidth=0.8)
    annotate(axes[0], bars)
    axes[0].text(0.5, -0.24, "(a) Meanings of encountered idioms", transform=axes[0].transAxes, ha="center", fontweight="bold")

    # Panel b: transfer beyond the listed meanings.
    tasks = ["Unseen figurative\ninference", "Emirati cultural\nreasoning"]
    context = [3.82, 2.05]
    context_ci = [[0.64, 7.01], [0.43, 3.67]]
    tagged = [5.41, 2.90]
    tagged_ci = [[1.59, 9.24], [1.20, 4.61]]
    x = np.arange(2)
    width = 0.34
    b1 = axes[1].bar(x - width / 2, context, width, label="Idiom text", color=BLUE, edgecolor=INK, hatch="..", yerr=asymmetric(context, context_ci), capsize=4)
    b2 = axes[1].bar(x + width / 2, tagged, width, label="Text + meanings", color=ORANGE, edgecolor=INK, hatch="//", yerr=asymmetric(tagged, tagged_ci), capsize=4)
    axes[1].set_xticks(x, tasks)
    axes[1].set_ylim(0, 11)
    axes[1].axhline(0, color=INK, linewidth=0.8)
    axes[1].legend(frameon=False, loc="upper right")
    annotate(axes[1], b1)
    annotate(axes[1], b2)
    axes[1].text(0.5, -0.24, "(b) Transfer in Arabic", transform=axes[1].transAxes, ha="center", fontweight="bold")

    # Panel c: running context and appended meanings have different effects.
    splits = ["Encountered", "Unseen"]
    context = [idiom["ar_seen"]["untagged-unf"][0], idiom["ar_unseen"]["untagged-unf"][0]]
    context_ci = [idiom["ar_seen"]["untagged-unf"][1], idiom["ar_unseen"]["untagged-unf"][1]]
    tagged = [idiom["ar_seen"]["cpt-unfiltered"][0], idiom["ar_unseen"]["cpt-unfiltered"][0]]
    tagged_ci = [idiom["ar_seen"]["cpt-unfiltered"][1], idiom["ar_unseen"]["cpt-unfiltered"][1]]
    x = np.arange(2)
    b1 = axes[2].bar(x - width / 2, context, width, label="Idiom text", color=BLUE, edgecolor=INK, hatch="..", yerr=asymmetric(context, context_ci), capsize=4)
    b2 = axes[2].bar(x + width / 2, tagged, width, label="Text + meanings", color=ORANGE, edgecolor=INK, hatch="//", yerr=asymmetric(tagged, tagged_ci), capsize=4)
    axes[2].set_xticks(x, splits)
    axes[2].set_ylim(-10, 33)
    axes[2].axhline(0, color=INK, linewidth=0.9)
    axes[2].legend(frameon=False, loc="upper right")
    annotate(axes[2], b1)
    annotate(axes[2], b2)
    axes[2].text(0.5, -0.24, "(c) Arabic idiom meaning", transform=axes[2].transAxes, ha="center", fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9DDE3", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.27, top=0.98, wspace=0.31)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_acquisition.pdf", bbox_inches="tight")
    plt.close(fig)


def make_symbolism() -> None:
    stats = json.loads((ROOT / "docs/paper_stats/v3/cross_language_symbolism.json").read_text())["contrasts"]["cpt_minus_unfiltered"]
    labels = ["Arabic", "Hindi", "Chinese", "Pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))

    lure_values = [100 * stats["languages"][lang]["english_lure"]["delta"] for lang in ("ar", "hi", "zh")] + [100 * stats["pooled"]["english_lure"]["micro_delta"]]
    lure_ci = [[100 * x for x in stats["languages"][lang]["english_lure"]["ci95"]] for lang in ("ar", "hi", "zh")] + [[100 * x for x in stats["pooled"]["english_lure"]["micro_ci95"]]]
    colors = [GREY, BLUE, BLUE, ORANGE]
    y = np.arange(4)
    bars = axes[0].barh(y, lure_values, color=colors, edgecolor=INK, hatch=["", "..", "..", "//"], xerr=asymmetric(lure_values, lure_ci), capsize=4, height=0.62)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Change in English-association choices (points)")
    axes[0].axvline(0, color=INK, linewidth=0.9)
    axes[0].set_xlim(-17, 7)
    for bar, value in zip(bars, lure_values):
        axes[0].text(value + (0.5 if value >= 0 else -0.5), bar.get_y() + bar.get_height() / 2, f"{value:+.1f}", va="center", ha="left" if value >= 0 else "right", fontsize=14)
    axes[0].text(0.5, -0.24, "(a) Fewer English associations is better", transform=axes[0].transAxes, ha="center", fontweight="bold")

    margin_values = [stats["languages"][lang]["gold_lure_margin"]["delta"] for lang in ("ar", "hi", "zh")] + [stats["pooled"]["gold_lure_margin"]["micro_delta"]]
    margin_ci = [stats["languages"][lang]["gold_lure_margin"]["ci95"] for lang in ("ar", "hi", "zh")] + [stats["pooled"]["gold_lure_margin"]["micro_ci95"]]
    bars = axes[1].barh(y, margin_values, color=colors, edgecolor=INK, hatch=["", "..", "..", "//"], xerr=asymmetric(margin_values, margin_ci), capsize=4, height=0.62)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Change in target-vs-English margin")
    axes[1].axvline(0, color=INK, linewidth=0.9)
    axes[1].set_xlim(-0.30, 0.55)
    for bar, value in zip(bars, margin_values):
        axes[1].text(value + (0.018 if value >= 0 else -0.018), bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=14)
    axes[1].text(0.5, -0.24, "(b) A larger target-culture margin is better", transform=axes[1].transAxes, ha="center", fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#D9DDE3", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.28, top=0.98, wspace=0.34)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_symbolism.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    style()
    make_acquisition()
    make_symbolism()
