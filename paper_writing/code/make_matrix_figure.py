#!/usr/bin/env python3
"""Transfer-matrix figure for the bidirectional study.

Reads docs/paper_stats/v2/bidir_2b_i.json (culture.bidirectional.aggregate) and draws, per
training condition (rows) and benchmark group (columns), the mean gain over Random-CPT
in percentage points. Diverging blue (gain) / gray (0) / red (loss) scale, symmetric
around zero; every cell carries its value, and an asterisk when the 95% CI excludes 0.
Output: latex/figures/fig_transfer_matrix.pdf
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
OVERLEAF = "/home/jiaruil5/culture_pretrain/OverleafCultureInFigurativeLanguage"  # paper repo (LaTeX only)

REPO = "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage"
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs/paper_stats/v2/bidir_2b_i.json")
out = os.path.join(OVERLEAF, "latex", "figures", "fig_transfer_matrix.pdf")
M = json.load(open(src))["matrix"]

ROWS = [("idiom_untagged", "Idiom-CPT $-$ tags"), ("idiom_tagged", "Idiom-CPT"),
        ("culture", "Culture-CPT"), ("culture_notes", "Culture-CPT + notes")]
COLS = [("idiom_meaning", "Meaning\n(seen)"), ("idiom_unseen", "Meaning\n(unseen)"),
        ("figurative", "Figurative\ninference"), ("cloze", "Idiom\ncloze"), ("symbolism", "Symbolism"),
        ("culture", "Culture"), ("regional", "Regional\nknowledge")]
COLS = [c for c in COLS if any(f"{r}|{c[0]}" in M for r, _ in ROWS)]

INK, MUTED = "#1f1f1e", "#6b6a66"
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
vals = [[100 * M[f"{r}|{c}"]["delta"] if f"{r}|{c}" in M else float("nan") for c, _ in COLS] for r, _ in ROWS]
vmax = max(3.0, max(abs(v) for row in vals for v in row if v == v))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(6.3, 2.35))
im = ax.imshow(vals, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
for i, (r, _) in enumerate(ROWS):
    for j, (c, _) in enumerate(COLS):
        k = f"{r}|{c}"
        if k not in M:
            ax.text(j, i, "n/a", ha="center", va="center", color=MUTED, fontsize=7)
            continue
        d = M[k]
        sig = d["ci95"][0] > 0 or d["ci95"][1] < 0
        ax.text(j, i, f"{100*d['delta']:+.1f}{'*' if sig else ''}", ha="center", va="center",
                color=INK, fontsize=8, fontweight="bold" if sig else "normal")
ax.set_xticks(range(len(COLS)), [c[1] for c in COLS], fontsize=7, color=INK)
ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS], fontsize=7.5, color=INK)
ax.tick_params(which="both", length=0)
ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
# white gaps between cells
ax.set_xticks([x - 0.5 for x in range(1, len(COLS))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(ROWS))], minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.axhline(1.5, color=INK, linewidth=1.0)  # idiom rows | culture rows
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("gain over Random-CPT (pp)", fontsize=7, color=MUTED)
cb.ax.tick_params(labelsize=6.5, colors=MUTED)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
