#!/usr/bin/env python3
"""Cross-language paired analysis of the cultural-symbolism probe.

The script uses the saved per-item predictions from the 9B study. It reports
accuracy, selection of the English-association lure, and the gold-minus-lure
log-probability margin. Pooled intervals use a stratified bootstrap that
resamples items within each language, preserving the evaluation design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


LANGUAGES = ("ar", "hi", "zh")
CONTRASTS = (("cpt", "unfiltered"), ("untagged", "unfiltered"), ("cpt", "untagged"))


def load_records(root: Path, lang: str, arm: str) -> dict[str, dict]:
    path = root / lang / arm / f"symbolism_v2_{lang}_letter.json"
    records = json.loads(path.read_text())["records"]
    return {r["qid"]: r for r in records}


def outcomes(records: dict[str, dict], qids: list[str]) -> dict[str, np.ndarray]:
    accuracy = np.asarray([records[q]["correct"] for q in qids], dtype=float)
    lure = np.asarray([int(records[q]["pred"] == records[q]["lure"]) for q in qids], dtype=float)
    margin = np.asarray(
        [records[q]["logprobs"][records[q]["gold"]] - records[q]["logprobs"][records[q]["lure"]] for q in qids],
        dtype=float,
    )
    return {"accuracy": accuracy, "english_lure": lure, "gold_lure_margin": margin}


def paired_binary_p(a: np.ndarray, b: np.ndarray) -> float:
    wins = int(np.sum((a == 1) & (b == 0)))
    losses = int(np.sum((a == 0) & (b == 1)))
    return float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0


def sign_flip_p(delta: np.ndarray, rng: np.random.Generator, draws: int = 100_000) -> float:
    observed = abs(float(delta.mean()))
    if not np.any(delta):
        return 1.0
    exceed = 0
    for start in range(0, draws, 2_000):
        size = min(2_000, draws - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(delta)))
        exceed += int(np.sum(np.abs((signs * delta).mean(axis=1)) >= observed))
    return float((exceed + 1) / (draws + 1))


def bootstrap(delta: np.ndarray, rng: np.random.Generator, draws: int = 20_000) -> list[float]:
    means = np.empty(draws)
    for start in range(0, draws, 2_000):
        size = min(2_000, draws - start)
        idx = rng.integers(0, len(delta), size=(size, len(delta)))
        means[start : start + size] = delta[idx].mean(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def stratified_bootstrap(deltas: dict[str, np.ndarray], rng: np.random.Generator, draws: int = 20_000) -> dict[str, list[float]]:
    micro = np.empty(draws)
    macro = np.empty(draws)
    total_n = sum(len(x) for x in deltas.values())
    for i in range(draws):
        language_means = []
        weighted_sum = 0.0
        for lang in LANGUAGES:
            delta = deltas[lang]
            sample_mean = float(delta[rng.integers(0, len(delta), size=len(delta))].mean())
            language_means.append(sample_mean)
            weighted_sum += len(delta) * sample_mean
        micro[i] = weighted_sum / total_n
        macro[i] = np.mean(language_means)
    return {
        "micro_ci95": [float(x) for x in np.quantile(micro, (0.025, 0.975))],
        "macro_ci95": [float(x) for x in np.quantile(macro, (0.025, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=Path("/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval"))
    parser.add_argument("--out", type=Path, default=Path("docs/paper_stats/v3/cross_language_symbolism.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(20261011)
    loaded = {lang: {arm: load_records(args.eval_root, lang, arm) for arm in {x for c in CONTRASTS for x in c}} for lang in LANGUAGES}
    result: dict[str, dict] = {"definition": {"scale": "raw units; multiply binary outcomes by 100 for percentage points"}, "contrasts": {}}

    for arm_a, arm_b in CONTRASTS:
        name = f"{arm_a}_minus_{arm_b}"
        result["contrasts"][name] = {"languages": {}, "pooled": {}}
        deltas_by_metric: dict[str, dict[str, np.ndarray]] = {m: {} for m in ("accuracy", "english_lure", "gold_lure_margin")}
        for lang in LANGUAGES:
            qids = sorted(set(loaded[lang][arm_a]) & set(loaded[lang][arm_b]))
            a = outcomes(loaded[lang][arm_a], qids)
            b = outcomes(loaded[lang][arm_b], qids)
            language_result = {"n": len(qids)}
            for metric in a:
                delta = a[metric] - b[metric]
                deltas_by_metric[metric][lang] = delta
                p = paired_binary_p(a[metric], b[metric]) if metric != "gold_lure_margin" else sign_flip_p(delta, rng)
                language_result[metric] = {"delta": float(delta.mean()), "ci95": bootstrap(delta, rng), "p": p}
            result["contrasts"][name]["languages"][lang] = language_result

        for metric, deltas in deltas_by_metric.items():
            concatenated = np.concatenate([deltas[lang] for lang in LANGUAGES])
            pooled = {
                "n": int(len(concatenated)),
                "micro_delta": float(concatenated.mean()),
                "macro_delta": float(np.mean([deltas[lang].mean() for lang in LANGUAGES])),
                **stratified_bootstrap(deltas, rng),
            }
            pooled["p"] = sign_flip_p(concatenated, rng)
            result["contrasts"][name]["pooled"][metric] = pooled

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
