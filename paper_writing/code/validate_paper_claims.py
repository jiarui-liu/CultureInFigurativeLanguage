#!/usr/bin/env python3
"""Fail if a headline paper claim drifts from its generated statistics."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "OverleafCultureInFigurativeLanguage"


def load(relative: str):
    return json.loads((ROOT / relative).read_text())


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, found {actual}")


idiom = load("docs/paper_stats/v2/idiomatlas_9b.json")
cross = load("docs/paper_stats/v3/cross_language_symbolism.json")["contrasts"]["cpt_minus_unfiltered"]
stats = load("docs/paper_stats/reframe_stats_9b.json")
audit = load("docs/paper_stats/v3/symbolism_probe_audit.json")

for key, expected in (("ar_seen", 24.7), ("hi_seen", 25.5), ("zh_seen", 10.7)):
    close(idiom[key]["cpt-unfiltered"][0], expected)
close(idiom["ar_unseen"]["cpt-unfiltered"][0], -4.0)
close(idiom["ar_unseen"]["cpt-untagged"][0], -7.0)
close(idiom["ar_unseen"]["untagged-unf"][0], 3.0)

holm = {(row["lang"], row["task"]): row for row in stats["holm"]}
close(100 * holm[("ar", "ar_figurative")]["delta"], 5.414, 0.01)
close(100 * holm[("ar", "alyah")]["delta"], 2.904, 0.01)
close(100 * stats["figurative_removed"]["alyah"]["cpt_vs_unfiltered"]["delta"], 2.299, 0.01)

pooled = cross["pooled"]
close(100 * pooled["english_lure"]["micro_delta"], -4.357, 0.01)
close(pooled["gold_lure_margin"]["micro_delta"], 0.18498, 0.001)
assert pooled["english_lure"]["micro_ci95"][1] < 0
assert pooled["gold_lure_margin"]["micro_ci95"][0] > 0

assert sum(x["consensus"] for x in audit["languages"].values()) == 376
assert sum(x["auditable"] for x in audit["languages"].values()) == 426

sources = [PAPER / "main.tex"] + [PAPER / "latex" / f for f in ("01_intro.tex", "04_analysis.tex", "05_exp.tex")]
paper_text = "\n".join(path.read_text() for path in sources)
for token in ("75.6K", "5.5\\%", "20.1K", "10.7 to 25.5", "5.4 points", "2.9 points", "4.4 points", "0.185", "$-4.0$"):
    if token not in paper_text:
        raise AssertionError(f"headline token missing from paper: {token}")

print("Validated 17 numerical invariants and 9 headline-paper tokens.")
