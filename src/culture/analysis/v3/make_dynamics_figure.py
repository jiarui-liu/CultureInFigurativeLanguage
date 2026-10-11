"""Figure: Arabic training dynamics (pass 3).

Reads docs/paper_stats/v3/dynamics_ar.json and writes fig_dynamics.pdf into the Overleaf
checkout, in the palette of the other figures.

    PYTHONPATH=src python src/culture/analysis/v3/make_dynamics_figure.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
STATS = REPO / "docs" / "paper_stats" / "v3"
DEFAULT_OUT = REPO.parent / "OverleafCultureInFigurativeLanguage" / "latex" / "figures"

TAG, SEL, EN, ZH = "#EC703E", "#327DD8", "#1F5FA8", "#B04A2F"
STEPS = [200, 400, 800, 1200, "final"]
XPOS = [200, 400, 800, 1200, 1608]

PANELS = [
    ("Idiom meaning", [("kinayat_meaning", "Kinayat-Meaning", TAG),
                       ("idiomatlas_mc_ar_seen", "IdiomAtlas-MC seen", SEL),
                       ("idiomatlas_mc_ar_unseen", "IdiomAtlas-MC unseen", ZH)]),
    ("Figurative and culture", [("ar_figurative", "AR-Figurative", TAG),
                                ("alyah", "Alyah", SEL),
                                ("arabculture", "ArabCulture", EN),
                                ("symbolism_v2_ar_letter", "Symbolism probe", ZH)]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    d = json.load(open(STATS / "dynamics_ar.json"))

    arms = [("cpt", r"Idiom-CPT", "-"), ("culture", "Culture-CPT", "--")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    for ax, (title, series) in zip(axes, PANELS):
        for task, label, color in series:
            for arm, armlabel, ls in arms:
                ys, xs = [], []
                for st, x in zip(STEPS, XPOS):
                    v = d.get(f"{task}/{arm}/{st}")
                    if v:
                        ys.append(100 * v["delta_acc"])
                        xs.append(x)
                if len(ys) < 2:
                    continue
                ax.plot(xs, ys, ls, color=color, marker="o", ms=3.2, lw=1.6,
                        alpha=1.0 if arm == "cpt" else 0.55,
                        label=f"{label} ({armlabel})")
        ax.axhline(0, color="#8A93A0", lw=0.8, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step (of 1,608)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="best")
    axes[0].set_ylabel("accuracy vs \\textsc{Random-CPT} (pts)".replace("\\textsc{", "")
                       .replace("}", ""), fontsize=9)
    fig.tight_layout()
    out = Path(args.out) / "fig_dynamics.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("[write]", out)


if __name__ == "__main__":
    main()
