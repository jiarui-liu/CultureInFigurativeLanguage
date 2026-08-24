#!/usr/bin/env python3
"""Point 1: bootstrap CIs + paired significance for the CPT eval tables.

Each task file eval/<lang>/<run>/<task>.json has {metrics, records}, where every
record has a qid and per-item correctness (`correct_norm` for continuation tasks,
`correct` for letter tasks; we follow metrics.score_mode / metrics.primary).

For each language and task we report, per run, acc with an item-level bootstrap
95% CI; and paired contrasts CPT-vs-base, CPT-vs-unfiltered, CPT-vs-untagged
(where available) with a paired-bootstrap 95% CI on the delta and a McNemar exact
two-sided p-value on the per-item wins/losses.

Deterministic bootstrap (fixed integer RNG seed via numpy default_rng) so results
are reproducible without depending on wall-clock.

Usage:
  python compute_cis.py --eval_root /lustre-storage/.../eval --langs ar hi zh \
      --out_dir <repo>/docs/paper_stats
"""
import argparse, glob, json, math, os
import numpy as np

RUNS = ["base", "cpt", "unfiltered", "untagged"]
B = 10000
SEED = 20260824


def load_run(run_dir):
    """qid -> correctness(bool) per task, using the primary metric."""
    tasks = {}
    for jf in glob.glob(os.path.join(run_dir, "*.json")):
        name = os.path.basename(jf)[:-5]
        if name == "summary":
            continue
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        recs = d.get("records")
        if not recs:
            continue
        mode = d.get("metrics", {}).get("score_mode", "continuation")
        key = "correct_norm" if mode == "continuation" else "correct"
        item = {}
        for r in recs:
            qid = r.get("qid")
            v = r.get(key, r.get("correct"))
            if qid is not None and v is not None:
                item[qid] = int(v)
        if item:
            tasks[name] = item
    return tasks


def boot_ci(arr, rng, b=B):
    arr = np.asarray(arr, float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(b, n))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired(a, b_, rng, bN=B):
    """a,b aligned per-item correctness (cpt=a vs other=b). delta = mean(a)-mean(b)."""
    a = np.asarray(a, float); b_ = np.asarray(b_, float)
    n = len(a)
    d = a.mean() - b_.mean()
    idx = rng.integers(0, n, size=(bN, n))
    deltas = a[idx].mean(axis=1) - b_[idx].mean(axis=1)
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    # McNemar exact (two-sided) on discordant pairs
    b01 = int(np.sum((a == 0) & (b_ == 1)))  # cpt wrong, other right
    c10 = int(np.sum((a == 1) & (b_ == 0)))  # cpt right, other wrong
    nd = b01 + c10
    if nd == 0:
        p = 1.0
    else:
        k = min(b01, c10)
        # two-sided exact binomial p, param 0.5
        from math import comb
        p = min(1.0, 2 * sum(comb(nd, i) for i in range(0, k + 1)) / (2 ** nd))
    return d, lo, hi, c10, b01, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--langs", nargs="+", default=["ar", "hi", "zh"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    report = {}
    for lang in args.langs:
        lroot = os.path.join(args.eval_root, lang)
        runs = {r: load_run(os.path.join(lroot, r)) for r in RUNS
                if os.path.isdir(os.path.join(lroot, r))}
        if "cpt" not in runs or "base" not in runs:
            print(f"[skip {lang}] missing base/cpt"); continue
        tasks = sorted(runs["cpt"].keys())
        lang_rep = {}
        for t in tasks:
            entry = {"per_run": {}, "contrasts": {}}
            # per-run acc + CI on the union-of-present items for that run
            for r, td in runs.items():
                if t in td:
                    vals = list(td[t].values())
                    lo, hi = boot_ci(vals, rng)
                    entry["per_run"][r] = {"acc": round(float(np.mean(vals)), 4),
                                            "n": len(vals), "ci95": [round(lo, 4), round(hi, 4)]}
            # paired contrasts vs cpt
            for other in ["base", "unfiltered", "untagged"]:
                if other in runs and t in runs[other]:
                    common = [q for q in runs["cpt"][t] if q in runs[other][t]]
                    if not common:
                        continue
                    a = [runs["cpt"][t][q] for q in common]
                    b_ = [runs[other][t][q] for q in common]
                    d, lo, hi, cpt_win, other_win, p = paired(a, b_, rng)
                    entry["contrasts"][f"cpt_vs_{other}"] = {
                        "n": len(common), "delta": round(d, 4),
                        "ci95": [round(lo, 4), round(hi, 4)],
                        "cpt_win": cpt_win, "other_win": other_win,
                        "mcnemar_p": round(p, 5),
                        "sig_0.05": bool(p < 0.05 and not (lo <= 0 <= hi)),
                    }
            lang_rep[t] = entry
        report[lang] = lang_rep

    out_json = os.path.join(args.out_dir, "ci_report.json")
    json.dump(report, open(out_json, "w"), ensure_ascii=False, indent=2)
    print("wrote", out_json)

    # human-readable summary
    lines = []
    for lang, lr in report.items():
        lines.append(f"\n===== {lang.upper()} =====")
        for t, e in lr.items():
            acc = {r: v["acc"] for r, v in e["per_run"].items()}
            lines.append(f"[{t}]  " + "  ".join(f"{r}={acc[r]:.3f}" for r in RUNS if r in acc))
            for c, cv in e["contrasts"].items():
                star = "*" if cv["sig_0.05"] else " "
                lines.append(f"    {star}{c}: Δ={cv['delta']:+.3f} CI[{cv['ci95'][0]:+.3f},{cv['ci95'][1]:+.3f}] "
                             f"p={cv['mcnemar_p']:.4f} (win {cv['cpt_win']}/{cv['other_win']})")
    txt = "\n".join(lines)
    open(os.path.join(args.out_dir, "ci_summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
