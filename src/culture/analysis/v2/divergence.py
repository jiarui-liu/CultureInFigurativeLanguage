"""Prompt-independent divergence between the meaning distributions of an entity's idioms
in two languages, with size-matched null baselines.

Each entity i has two sets of unit vectors: A_i (idioms of language 1 containing the entity)
and B_i (idioms of language 2 containing its translation). Divergence measures between sets:
  centroid : 1 - cos(mean(A), mean(B))
  energy   : energy distance with cosine distance d = 1 - cos (U-statistic, diagonal excluded)
  chamfer  : 1 - average cross-set nearest-neighbour cosine similarity (symmetrised)

Size-matched design (all comparisons are k-vs-k subsamples, averaged over R draws):
  cross_same   A_i vs B_i                          same entity, different language
  cross_diff   A_i vs B_j (j != i, random)         different entity, different language (null)
  within_same  A_i' vs A_i'' (disjoint halves)     same entity, same language (split-half)
  within_diff  A_i vs A_j                          different entity, same language
Full-set calibration: percentile of centroid(A_i, B_i) among centroid(A_i, B_j) over all j != i
(0 = the translation's idioms are the closest of all entities; 0.5 = no closer than a random entity).
"""
import numpy as np

METRICS = ("centroid", "energy", "chamfer")


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def centroid(A, B):
    return float(1 - _unit(A.mean(0)) @ _unit(B.mean(0)))


def energy(A, B):
    dab = 1 - A @ B.T
    daa = 1 - A @ A.T
    dbb = 1 - B @ B.T
    na, nb = len(A), len(B)
    ia = (daa.sum() - np.trace(daa)) / (na * (na - 1)) if na > 1 else 0.0
    ib = (dbb.sum() - np.trace(dbb)) / (nb * (nb - 1)) if nb > 1 else 0.0
    return float(2 * dab.mean() - ia - ib)


def chamfer(A, B):
    s = A @ B.T
    return float(1 - 0.5 * (s.max(1).mean() + s.max(0).mean()))


FN = {"centroid": centroid, "energy": energy, "chamfer": chamfer}


def all_metrics(A, B):
    return {m: FN[m](A, B) for m in METRICS}


def size_matched(A_sets, B_sets, k=5, reps=30, seed=0):
    """A_sets, B_sets: lists of arrays (same entity order). Only entities with >=k on both sides
    are used. Returns per-entity dict of condition -> metric -> mean over reps."""
    rng = np.random.default_rng(seed)
    idx = [i for i in range(len(A_sets)) if len(A_sets[i]) >= k and len(B_sets[i]) >= k]
    out = {}
    for i in idx:
        acc = {c: {m: [] for m in METRICS} for c in
               ("cross_same", "cross_diff", "within_same_a", "within_same_b", "within_diff_a", "within_diff_b")}
        others = [j for j in idx if j != i]
        for _ in range(reps):
            a = A_sets[i][rng.choice(len(A_sets[i]), k, replace=False)]
            b = B_sets[i][rng.choice(len(B_sets[i]), k, replace=False)]
            j = others[rng.integers(len(others))]
            bj = B_sets[j][rng.choice(len(B_sets[j]), k, replace=False)]
            j2 = others[rng.integers(len(others))]
            aj = A_sets[j2][rng.choice(len(A_sets[j2]), k, replace=False)]
            j3 = others[rng.integers(len(others))]
            bj3 = B_sets[j3][rng.choice(len(B_sets[j3]), k, replace=False)]
            for m in METRICS:
                acc["cross_same"][m].append(FN[m](a, b))
                acc["cross_diff"][m].append(FN[m](a, bj))
                acc["within_diff_a"][m].append(FN[m](a, aj))
                acc["within_diff_b"][m].append(FN[m](b, bj3))
            for side, S in (("a", A_sets[i]), ("b", B_sets[i])):
                if len(S) >= 2 * k:
                    p = rng.permutation(len(S))
                    x, y = S[p[:k]], S[p[k:2 * k]]
                    for m in METRICS:
                        acc[f"within_same_{side}"][m].append(FN[m](x, y))
        out[i] = {c: {m: (float(np.mean(v)) if v else None) for m, v in d.items()} for c, d in acc.items()}
    return out


def summarize_size_matched(per, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    conds = ("cross_same", "cross_diff", "within_same_a", "within_same_b", "within_diff_a", "within_diff_b")
    res = {"n_entities": len(per)}
    for m in METRICS:
        r = {}
        for c in conds:
            v = np.array([per[i][c][m] for i in per if per[i][c][m] is not None])
            if len(v) == 0:
                continue
            bs = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
            r[c] = {"mean": round(float(v.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                              round(float(np.percentile(bs, 97.5)), 4)], "n": int(len(v))}
        # paired comparisons over entities
        ids = [i for i in per]
        cs = np.array([per[i]["cross_same"][m] for i in ids])
        cd = np.array([per[i]["cross_diff"][m] for i in ids])
        r["frac_entities_cross_same_lt_cross_diff"] = round(float((cs < cd).mean()), 4)
        # divergence index: 0 = as close as split halves of the same entity in one language,
        # 1 = as far as a random different entity across languages. Entities with both split-halves.
        di_ids = [i for i in ids if per[i]["within_same_a"][m] is not None and per[i]["within_same_b"][m] is not None]
        if di_ids:
            W = np.array([(per[i]["within_same_a"][m] + per[i]["within_same_b"][m]) / 2 for i in di_ids])
            X = np.array([per[i]["cross_same"][m] for i in di_ids])
            D = np.array([per[i]["cross_diff"][m] for i in di_ids])
            r["divergence_index_aggregate"] = {
                "value": round(float((X.mean() - W.mean()) / (D.mean() - W.mean())), 3), "n_entities": len(di_ids),
                "definition": "(mean cross_same - mean within_same) / (mean cross_diff - mean within_same)"}
        res[m] = r
    return res


def full_set_calibration(A_sets, B_sets, idx):
    """For entities idx (all with non-empty sets), centroid divergence of the true pair and its
    percentile among all mismatched pairs, in both directions."""
    ca = np.stack([_unit(A_sets[i].mean(0)) for i in idx])
    cb = np.stack([_unit(B_sets[i].mean(0)) for i in idx])
    S = ca @ cb.T  # similarity
    n = len(idx)
    out = {}
    for r, i in enumerate(idx):
        true = S[r, r]
        row = np.delete(S[r], r)
        col = np.delete(S[:, r], r)
        out[i] = {"centroid_div": float(1 - true),
                  "pct_a2b": float((row >= true).mean()),  # share of wrong B-entities at least as close
                  "pct_b2a": float((col >= true).mean()),
                  "rank_a2b": int((row > true).sum()) + 1,
                  "mean_div_random": float(1 - row.mean()),
                  "z": float((row.mean() - true) / (row.std() + 1e-12)) * -1}
        out[i]["percentile"] = (out[i]["pct_a2b"] + out[i]["pct_b2a"]) / 2
    return out
