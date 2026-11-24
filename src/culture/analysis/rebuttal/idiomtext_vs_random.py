#!/usr/bin/env python3
"""Rebuttal R-4Spy: direct paired tests of Idiom-Text (run `untagged`) vs Random-CPT (run `unfiltered`),
plus Idiom+Meaning (`cpt`) vs both, on every 9B benchmark whose per-item records exist on babel.

For each (lang, task): delta, paired-bootstrap 95% CI (10k), exact McNemar p, Holm-adjusted p within
language x family. Also a pooled unseen-idiom test across languages (language-stratified bootstrap,
item-weighted, and an equal-language-weight variant), and a fixed-effect sign test.

Usage: python -m culture.analysis.rebuttal.idiomtext_vs_random --out docs/paper_stats/rebuttal/idiomtext_vs_random.json
"""
import argparse, json, math, os
from pathlib import Path
import numpy as np

ROOT = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
DIRS = ["eval9b", "eval9b_ext"]
RUN = {"base": "Base", "unfiltered": "Random-CPT", "untagged": "Idiom-Text", "cpt": "Idiom+Meaning"}
FAMILY = {  # unseen = idioms absent from training docs / KB
    "idiomatlas_mc_{L}_seen": "seen", "jawaher_ar": "seen", "absher_meaning": "seen",
    "absher_cultural_interpretation": "seen",
    "idiomatlas_mc_{L}_unseen": "unseen", "midi_ar": "unseen", "midi_zh": "unseen",
    "hindi_proverbs": "unseen", "xiehouyu_zh": "unseen",
    "symbolism_v2_{L}_letter": "symbolism",
    "midi_ar_Egypt": "unseen_by_variety", "midi_ar_Syrian": "unseen_by_variety",
    "midi_ar_UAE": "unseen_by_variety", "midi_ar_Morocco": "unseen_by_variety",
    "include_ar": "culture", "include_hi": "culture", "include_zh": "culture",
    "culturalbench_hi": "culture", "culturalbench_zh": "culture",
}
B = 10000


VIRTUAL = {  # virtual task -> (file stem, record filter); MIDI is scored on figurative uses only, as in the paper
    "midi_ar": ("midi_ar", lambda r: r.get("use") == "figurative"),
    "midi_zh": ("midi_zh", lambda r: r.get("use") == "figurative"),
    **{f"midi_ar_{v}": ("midi_ar", lambda r, v=v: r.get("use") == "figurative" and r.get("variety") == f"Arabic_{v}")
       for v in ["Egypt", "Syrian", "UAE", "Morocco"]},
}


def load(L, run, task):
    stem, keep = VIRTUAL.get(task, (task, lambda r: True))
    for d in DIRS:
        p = ROOT / d / L / run / f"{stem}.json"
        if p.exists():
            j = json.load(open(p))
            mode = j.get("metrics", {}).get("score_mode", "continuation")
            key = "correct_norm" if mode == "continuation" else "correct"
            out, seen = {}, {}
            for r in j["records"]:
                if not keep(r):
                    continue
                q = r["qid"]; k = seen.get(q, 0); seen[q] = k + 1
                out[q if k == 0 else f"{q}#{k}"] = int(r.get(key, r.get("correct")))
            return out
    return None


def mcnemar(a, b):
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), n01, n10


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0] * m; run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rng = np.random.default_rng(20261124)
    rows, pooled = [], {}
    for L in ["ar", "hi", "zh"]:
        for tmpl, fam in FAMILY.items():
            task = tmpl.format(L=L)
            recs = {r: load(L, r, task) for r in RUN}
            recs = {r: v for r, v in recs.items() if v}
            if "unfiltered" not in recs or len(recs) < 2:
                continue
            qids = sorted(set.intersection(*[set(v) for v in recs.values()]))
            acc = {r: np.array([v[q] for q in qids]) for r, v in recs.items()}
            row = {"lang": L, "task": task, "family": fam, "n": len(qids),
                   "acc": {RUN[r]: round(100 * x.mean(), 1) for r, x in acc.items()}}
            idx = rng.integers(0, len(qids), size=(B, len(qids)))
            for t, c in [("untagged", "unfiltered"), ("cpt", "unfiltered"), ("cpt", "untagged")]:
                if t not in acc or c not in acc:
                    continue
                d = acc[t] - acc[c]; bs = d[idx].mean(1)
                p, w, l = mcnemar(acc[t], acc[c])
                row[f"{RUN[t]} - {RUN[c]}"] = {"delta": round(100 * d.mean(), 1),
                    "ci95": [round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)],
                    "p_boot_two_sided": round(float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))), 4),
                    "mcnemar_p": round(p, 5), "wins": w, "losses": l}
                if fam == "unseen":
                    pooled.setdefault(f"{RUN[t]} - {RUN[c]}", []).append((L, task, d))
            rows.append(row)
    # Holm within lang x family x contrast
    groups = {}
    for i, r in enumerate(rows):
        for k, v in r.items():
            if isinstance(v, dict) and "mcnemar_p" in v:
                groups.setdefault((r["lang"], r["family"], k), []).append((i, k))
    for g, mem in groups.items():
        adj = holm([rows[i][k]["mcnemar_p"] for i, k in mem])
        for (i, k), p in zip(mem, adj):
            rows[i][k]["holm_p"] = round(p, 5)
    # pooled unseen: stratified bootstrap over items within each benchmark
    pool_out = {}
    for k, lst in pooled.items():
        for scope, sel in [("idiomatlas_mc_unseen_only", [x for x in lst if "idiomatlas" in x[1]]),
                           ("all_unseen_benchmarks", lst)]:
            if not sel:
                continue
            ds = [x[2] for x in sel]; ns = np.array([len(x) for x in ds])
            point_iw = float(np.concatenate(ds).mean())
            point_eq = float(np.mean([x.mean() for x in ds]))
            bi, be = [], []
            for _ in range(B):
                ms = [x[rng.integers(0, len(x), len(x))].mean() for x in ds]
                bi.append(np.dot(ms, ns) / ns.sum()); be.append(np.mean(ms))
            bi, be = np.array(bi), np.array(be)
            pool_out[f"{k} | {scope}"] = {
                "benchmarks": [f"{x[0]}:{x[1]}" for x in sel], "n_items": int(ns.sum()),
                "item_weighted": {"delta": round(100 * point_iw, 2),
                                  "ci95": [round(100 * np.percentile(bi, 2.5), 2), round(100 * np.percentile(bi, 97.5), 2)],
                                  "p_one_sided_le0": round(float((bi <= 0).mean()), 4)},
                "benchmark_weighted": {"delta": round(100 * point_eq, 2),
                                       "ci95": [round(100 * np.percentile(be, 2.5), 2), round(100 * np.percentile(be, 97.5), 2)],
                                       "p_one_sided_le0": round(float((be <= 0).mean()), 4)},
                "n_positive": int(sum(x.mean() > 0 for x in ds)), "n_benchmarks": len(ds)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"rows": rows, "pooled_unseen": pool_out}, open(a.out, "w"), indent=1, ensure_ascii=False)
    for r in rows:
        s = f"{r['lang']} {r['task']:34s} n={r['n']:5d} "
        for k in ["Idiom-Text - Random-CPT", "Idiom+Meaning - Random-CPT", "Idiom+Meaning - Idiom-Text"]:
            if k in r:
                v = r[k]; s += f"| {k.split(' - ')[0][:6]}-{k.split(' - ')[1][:6]} {v['delta']:+5.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}] p={v['mcnemar_p']:.3f} holm={v['holm_p']:.3f} "
        print(s)
    for k, v in pool_out.items():
        print(k, v["n_items"], v["item_weighted"], v["benchmark_weighted"], f"{v['n_positive']}/{v['n_benchmarks']} positive")


if __name__ == "__main__":
    main()
