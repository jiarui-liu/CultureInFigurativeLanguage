#!/usr/bin/env python3
"""Compare two run summaries (base vs. CPT) and print the per-task delta.

    python -m culture.evaluation.compare_results \\
        --base results/hi/base/summary.json \\
        --cpt  results/hi/cpt/summary.json
"""

import argparse
import json
from typing import Any, Dict, Optional


def _primary(task_metrics: Dict[str, Any]) -> Optional[float]:
    """The headline number for a task (acc_norm for MC, idiom_score for IdiomCE)."""
    return task_metrics.get("primary")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="summary.json for the base/untrained model.")
    p.add_argument("--cpt", required=True, help="summary.json for the continued-pretrained model.")
    args = p.parse_args()

    with open(args.base, encoding="utf-8") as f:
        base = json.load(f)
    with open(args.cpt, encoding="utf-8") as f:
        cpt = json.load(f)

    tasks = sorted(set(base["tasks"]) | set(cpt["tasks"]))
    print(f"{'task':<14}{'metric':<16}{'base':>10}{'cpt':>10}{'delta':>10}")
    print("-" * 60)
    for t in tasks:
        b = base["tasks"].get(t, {})
        c = cpt["tasks"].get(t, {})
        metric = "idiom_score" if t == "idiomce" else b.get("score_mode", "acc")
        bp, cp = _primary(b), _primary(c)
        delta = (cp - bp) if (bp is not None and cp is not None) else None
        bs = f"{bp:.4f}" if bp is not None else "-"
        cs = f"{cp:.4f}" if cp is not None else "-"
        ds = f"{delta:+.4f}" if delta is not None else "-"
        print(f"{t:<14}{metric:<16}{bs:>10}{cs:>10}{ds:>10}")


if __name__ == "__main__":
    main()
