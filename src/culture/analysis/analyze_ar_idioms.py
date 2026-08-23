#!/usr/bin/env python3
"""Linguistic analyses of the Arabic idiom KB — the Arabic counterpart of the
Chinese/English analyses in this directory.

Replicates, for Arabic, what the existing scripts do for zh/en:

======  ==========================================  ==========================
sub-cmd  analysis                                    zh/en counterpart
======  ==========================================  ==========================
stats    descriptive statistics                      ``idiom_statistics.py``
entities entity frequency distribution + plot        ``entity_clustering.py``
cluster  entity embedding k-means + 2-D map          ``cluster_entities_with_embeddings.py``
semantic intra-lingual clusters of shared meanings   ``intra_lingual_idiom_clusters.py``
variety  per-variety contrastive analysis            (no zh counterpart)
all      run every sub-command                       —
======  ==========================================  ==========================

``variety`` has no Chinese counterpart on purpose: Arabic entries carry an ISO
639-3 variety and a register (classical / MSA / colloquial), which Chinese does
not, so we can ask which entities and themes are dialect-specific.

Embeddings come from the MetaGen API (``text-embedding-3-small`` @512d) and are
cached to disk, so re-runs are free. Arabic plot labels are shaped + bidi-ordered
(matplotlib renders raw Arabic as disconnected, left-to-right letters otherwise).

Usage::

    export EMBEDDING_API_KEY="$APE_API_KEY"
    python analyze_ar_idioms.py all \\
        --input  data/idioms/ar/idioms_merged_llm_formatted.jsonl \\
        --outdir data/idioms/ar/analysis

    python analyze_ar_idioms.py --self-test
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# parents[2] == <repo>/src  (this file is src/culture/analysis/…)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_AUTORESEARCH = Path.home() / "local/git-repos/meta-autoresearch/code"
if _AUTORESEARCH.is_dir():
    sys.path.insert(0, str(_AUTORESEARCH))

from culture.data_processing.ar_idioms.normalize import normalize_ar  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("culture.analysis.ar")

NAN = "NAN"


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_rows(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def fld(row: Dict[str, Any], name: str) -> List[str]:
    # NB: `.get("output", {})` is NOT enough — some rows carry an explicit
    # "output": null (9 such rows in the English KB), which returns None.
    out = row.get("output") or {}
    v = out.get(name, NAN)
    if v == NAN or v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s", path)


def write_jsonl(rows: Sequence[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %s (%d rows)", path, len(rows))


_AR_FONT_CANDIDATES = [
    "/usr/share/fonts/google-droid-sans-fonts/DroidKufi-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def setup_arabic_font() -> Optional[str]:
    """Register an Arabic-capable font with matplotlib.

    DejaVu Sans (matplotlib's default) has NO Arabic glyphs, so every label
    renders as tofu boxes. We look for a real Arabic face and make it the default.
    """
    import glob
    import matplotlib
    from matplotlib import font_manager
    cands = [p for p in _AR_FONT_CANDIDATES if os.path.exists(p)]
    cands += sorted(glob.glob("/usr/share/fonts/**/*Kufi*.ttf", recursive=True))
    cands += sorted(glob.glob("/usr/share/fonts/**/*Naskh*.ttf", recursive=True))
    cands += sorted(glob.glob("/usr/share/fonts/**/*Arabic*.ttf", recursive=True))
    for path in cands:
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.family"] = name
            logger.info("plot font: %s (%s)", name, path)
            return name
        except Exception:  # noqa: BLE001
            continue
    logger.warning("no Arabic font found; plot labels will not render correctly")
    return None


def ar_label(s: str) -> str:
    """Shape + bidi-order Arabic so matplotlib renders it correctly."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(s))
    except Exception:  # noqa: BLE001 — degrade to raw text rather than crash
        return s


# --------------------------------------------------------------------------- #
# Embeddings (MetaGen API, disk-cached)
# --------------------------------------------------------------------------- #
def embed_cached(texts: List[str], cache_path: Path, batch: int = 128) -> np.ndarray:
    """Embed `texts`, reusing a JSONL cache keyed by the exact string."""
    cache: Dict[str, List[float]] = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                cache[d["t"]] = d["v"]
        logger.info("embedding cache: %d vectors", len(cache))

    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        from autoresearch.utils.llm import embed_texts
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            for i in range(0, len(missing), batch):
                chunk = missing[i:i + batch]
                vecs = _embed_with_retry(embed_texts, chunk)
                for t, v in zip(chunk, vecs):
                    cache[t] = v
                    f.write(json.dumps({"t": t, "v": v}, ensure_ascii=False) + "\n")
                f.flush()
                logger.info("  embedded %d/%d", min(i + batch, len(missing)), len(missing))
    return np.array([cache[t] for t in texts], dtype=np.float32)


def _embed_with_retry(embed_texts, chunk: List[str], attempts: int = 6):
    """Retry one embedding batch with exponential backoff.

    ``autoresearch.utils.llm.embed_texts`` raises straight through on 5xx, so a
    single transient gateway timeout used to kill a 40-minute run: the ar<->en
    job died at 1,152/4,084 on ``HTTP Error 504``. Everything already embedded is
    on disk, so a retry here (and a rerun at worst) costs nothing but time.
    """
    import time
    for attempt in range(attempts):
        try:
            return embed_texts(chunk)
        except Exception as e:  # noqa: BLE001 — any transport/5xx error is worth retrying
            # 401/403 is a missing or wrong APE_API_KEY, not a blip. Backing off
            # six times just delays the same failure by five minutes.
            if getattr(e, "code", None) in (401, 403):
                raise RuntimeError(
                    f"Embedding API rejected the credentials (HTTP {e.code}). Embeddings go "
                    "to the APE endpoint, and EmbeddingSettings.from_env() reads "
                    "EMBEDDING_API_KEY first and METAGEN_API_KEY only as a fallback -- so "
                    "run `export EMBEDDING_API_KEY=$APE_API_KEY`. Exporting APE_API_KEY "
                    "alone silently falls back to the METAGEN key, which that endpoint "
                    "rejects. Note also that ~/.bashrc returns early in non-interactive "
                    "shells, so `source ~/.bashrc` inside a tmux command sets nothing."
                ) from e
            if attempt == attempts - 1:
                raise
            delay = 5 * (2 ** attempt)
            logger.warning("  embed batch failed (%s: %s); retry %d/%d in %ds",
                           type(e).__name__, str(e)[:120], attempt + 1, attempts - 1, delay)
            time.sleep(delay)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


# --------------------------------------------------------------------------- #
# A1 — statistics
# --------------------------------------------------------------------------- #
def cmd_stats(rows: List[Dict[str, Any]], outdir: Path, args) -> Dict[str, Any]:
    n = len(rows)
    def cov(f):
        c = sum(1 for r in rows if fld(r, f))
        return {"have": c, "nan": n - c, "pct": round(100 * c / n, 2)}

    tok = [len(r["output"]["idiom"].split()) for r in rows]
    ch = [len(r["output"]["idiom"]) for r in rows]
    figlen = [len(m) for r in rows for m in fld(r, "figurative_meanings")]

    stats = {
        "total_idioms": n,
        "field_coverage": {f: cov(f) for f in
                           ("entities", "literal_meanings", "figurative_meanings",
                            "figurative_meanings_en", "examples")},
        "idiom_tokens": {"mean": round(float(np.mean(tok)), 2),
                         "median": int(np.median(tok)),
                         "min": int(np.min(tok)), "max": int(np.max(tok)),
                         "p90": int(np.percentile(tok, 90))},
        "idiom_chars": {"mean": round(float(np.mean(ch)), 2),
                        "median": int(np.median(ch)), "max": int(np.max(ch))},
        "figurative_meaning_chars": {
            "mean": round(float(np.mean(figlen)), 2) if figlen else 0,
            "median": int(np.median(figlen)) if figlen else 0},
        "by_register": dict(Counter(r["output"].get("register", NAN) for r in rows)),
        "by_variety": dict(Counter(
            v for r in rows
            for v in (r["output"].get("variety") if r["output"].get("variety") != NAN
                      else [NAN])).most_common()),
        "by_source": dict(Counter(s for r in rows
                                  for s in r.get("meta", {}).get("sources", [])).most_common()),
        "entities_per_idiom": {
            "mean": round(float(np.mean([len(fld(r, "entities")) for r in rows])), 2)},
        "unique_entities": len({e for r in rows for e in fld(r, "entities")}),
    }
    write_json(stats, outdir / "statistics.json")
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:1500])
    return stats


# --------------------------------------------------------------------------- #
# A2 — entity frequency
# --------------------------------------------------------------------------- #
def entity_counter(rows: List[Dict[str, Any]]) -> Counter:
    """Count entities, normalizing so ال/diacritic variants merge."""
    c = Counter()
    surface: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for e in fld(r, "entities"):
            k = normalize_ar(e)
            k = re.sub(r"^ال", "", k)          # merge definite/indefinite
            if len(k) < 2:
                continue
            c[k] += 1
            surface[k][e] += 1
    # relabel each key with its most common surface form for readability
    out = Counter()
    for k, n in c.items():
        out[surface[k].most_common(1)[0][0]] = n
    return out


def cmd_entities(rows: List[Dict[str, Any]], outdir: Path, args) -> Dict[str, Any]:
    c = entity_counter(rows)
    top = c.most_common(args.top_k)
    write_json([e for e, _ in c.most_common(200)], outdir / "top_entities_ar.json")
    write_json({"total_unique": len(c), "total_mentions": sum(c.values()),
                "top": [{"entity": e, "count": n} for e, n in top]},
               outdir / "entity_frequency.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        setup_arabic_font()
        fig, ax = plt.subplots(figsize=(11, 7))
        labels = [ar_label(e) for e, _ in top][::-1]
        vals = [n for _, n in top][::-1]
        ax.barh(range(len(vals)), vals, color="#2b7bba")
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("occurrences in idioms")
        ax.set_title(f"Top {len(top)} entities in Arabic idioms (n={len(rows)})")
        fig.tight_layout()
        p = outdir / "plots" / "ar_entity_frequency.pdf"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p); plt.close(fig)
        logger.info("wrote %s", p)
    except Exception as e:  # noqa: BLE001
        logger.warning("plot skipped: %s", e)

    print("\nTop 25 entities:")
    for e, n in top[:25]:
        print(f"  {n:>5}  {e}")
    return {"unique": len(c), "top": top[:25]}


# --------------------------------------------------------------------------- #
# A3 — entity embedding clustering
# --------------------------------------------------------------------------- #
def cmd_cluster(rows: List[Dict[str, Any]], outdir: Path, args) -> Dict[str, Any]:
    c = entity_counter(rows)
    ents = [e for e, n in c.most_common(args.max_entities) if n >= args.min_count]
    if len(ents) < args.n_clusters:
        logger.warning("only %d entities; reducing k", len(ents))
        args.n_clusters = max(2, len(ents) // 2)
    logger.info("clustering %d entities into %d clusters", len(ents), args.n_clusters)

    X = l2norm(embed_cached(ents, outdir / "cache" / "entity_embeddings.jsonl"))

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=42).fit(X)
    labels, cent = km.labels_, km.cluster_centers_

    clusters = []
    for k in range(args.n_clusters):
        idx = np.where(labels == k)[0]
        if not len(idx):
            continue
        # rank members by closeness to the centroid -> "central entities"
        d = X[idx] @ cent[k]
        order = idx[np.argsort(-d)]
        clusters.append({
            "cluster": int(k), "size": int(len(idx)),
            "central_entities": [ents[i] for i in order[:10]],
            "members": [{"entity": ents[i], "count": c[ents[i]],
                         "similarity": round(float(X[i] @ cent[k]), 4)}
                        for i in order],
        })
    clusters.sort(key=lambda x: -x["size"])
    write_json(clusters, outdir / "entity_clusters.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        setup_arabic_font()
        from sklearn.decomposition import PCA
        P = PCA(n_components=2, random_state=42).fit_transform(X)
        fig, ax = plt.subplots(figsize=(13, 10))
        ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab20", s=18, alpha=0.75)
        for cl in clusters:                       # annotate cluster centres only
            for e in cl["central_entities"][:3]:
                i = ents.index(e)
                ax.annotate(ar_label(e), (P[i, 0], P[i, 1]), fontsize=8)
        ax.set_title(f"Arabic idiom entities — {args.n_clusters} k-means clusters")
        fig.tight_layout()
        p = outdir / "plots" / "ar_entity_clusters.pdf"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p); plt.close(fig)
        logger.info("wrote %s", p)
    except Exception as e:  # noqa: BLE001
        logger.warning("cluster plot skipped: %s", e)

    print(f"\n{len(clusters)} entity clusters; central entities per cluster:")
    for cl in clusters[:12]:
        print(f"  [{cl['size']:>4}] {'، '.join(cl['central_entities'][:6])}")
    return {"n_clusters": len(clusters), "n_entities": len(ents)}


# --------------------------------------------------------------------------- #
# A4 — intra-lingual semantic clusters
# --------------------------------------------------------------------------- #
def cmd_semantic(rows: List[Dict[str, Any]], outdir: Path, args) -> Dict[str, Any]:
    """Group idioms whose FIGURATIVE MEANINGS are near-duplicates.

    Mirrors ``intra_lingual_idiom_clusters.py``: one embedding per meaning, then
    greedy agglomeration at a cosine threshold, so a cluster == "idioms that mean
    roughly the same thing".
    """
    items: List[Tuple[int, str]] = []
    for i, r in enumerate(rows):
        for m in fld(r, "figurative_meanings"):
            if len(m) >= 10:
                items.append((i, m[:400]))
    logger.info("embedding %d figurative meanings", len(items))
    X = l2norm(embed_cached([m for _, m in items],
                            outdir / "cache" / "meaning_embeddings.jsonl"))

    # Greedy single-pass agglomeration on cosine similarity.
    n = len(items)
    unassigned = np.ones(n, dtype=bool)
    clusters: List[List[int]] = []
    order = np.argsort(-np.asarray([len(m) for _, m in items]))   # long, informative first
    for seed in order:
        if not unassigned[seed]:
            continue
        sims = X @ X[seed]
        members = np.where(unassigned & (sims >= args.threshold))[0]
        if len(members) < 2:
            unassigned[seed] = False
            continue
        unassigned[members] = False
        clusters.append(members.tolist())

    out = []
    for cl in sorted(clusters, key=len, reverse=True):
        rid = {items[j][0] for j in cl}
        if len(rid) < 2:                                  # same idiom, two glosses
            continue
        out.append({
            "shared_meaning": items[cl[0]][1],
            "ar_idiom_count": len(rid),
            "ar_idioms": [{
                "idiom": rows[i]["output"]["idiom"],
                "entities": fld(rows[i], "entities"),
                "figurative_meanings": fld(rows[i], "figurative_meanings"),
                "literal_meanings": fld(rows[i], "literal_meanings"),
                "variety": rows[i]["output"].get("variety", NAN),
            } for i in sorted(rid)],
        })
    write_jsonl(out, outdir / "ar_only_clusters.jsonl")

    # pairwise view, as the zh pipeline also emits
    pairs = []
    for cl in out:
        ids = cl["ar_idioms"]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pairs.append({"idiom_a": ids[a]["idiom"], "idiom_b": ids[b]["idiom"],
                              "shared_meaning": cl["shared_meaning"],
                              "entities_a": ids[a]["entities"],
                              "entities_b": ids[b]["entities"],
                              "variety_a": ids[a]["variety"],
                              "variety_b": ids[b]["variety"]})
    write_jsonl(pairs, outdir / "ar_intra_lingual_pairs.jsonl")

    print(f"\n{len(out)} semantic clusters, {len(pairs)} idiom pairs")
    for cl in out[:8]:
        print(f"  [{cl['ar_idiom_count']}] {cl['shared_meaning'][:70]}")
        for it in cl["ar_idioms"][:3]:
            print(f"        - {it['idiom'][:60]}")
    return {"clusters": len(out), "pairs": len(pairs)}


# --------------------------------------------------------------------------- #
# A7 — per-variety contrastive (Arabic-specific)
# --------------------------------------------------------------------------- #
def cmd_variety(rows: List[Dict[str, Any]], outdir: Path, args) -> Dict[str, Any]:
    """Which entities are distinctive of each Arabic variety / register?

    Distinctiveness = ratio of the entity's within-variety rate to its overall
    rate, with a frequency floor so single occurrences cannot top the list.
    """
    by_var: Dict[str, Counter] = defaultdict(Counter)
    var_tot: Counter = Counter()
    overall = Counter()
    for r in rows:
        vs = r["output"].get("variety", NAN)
        vs = vs if isinstance(vs, list) else [NAN]
        ents = {re.sub(r"^ال", "", normalize_ar(e)) for e in fld(r, "entities")}
        ents = {e for e in ents if len(e) >= 2}
        for v in vs:
            var_tot[v] += 1
            for e in ents:
                by_var[v][e] += 1
        for e in ents:
            overall[e] += 1

    total = sum(var_tot.values()) or 1
    result = {}
    for v, cnt in by_var.items():
        if var_tot[v] < args.min_variety_size:
            continue
        scored = []
        for e, n in cnt.items():
            if n < args.min_count:
                continue
            rate_v = n / var_tot[v]
            rate_o = overall[e] / total
            scored.append((e, n, round(rate_v / rate_o, 2) if rate_o else 0.0))
        scored.sort(key=lambda x: -x[2])
        result[v] = {"n_idioms": var_tot[v],
                     "distinctive_entities": [
                         {"entity": e, "count": n, "lift": lift}
                         for e, n, lift in scored[:20]]}
    write_json(result, outdir / "variety_contrast.json")

    reg = defaultdict(Counter)
    for r in rows:
        reg[r["output"].get("register", NAN)][len(r["output"]["idiom"].split())] += 1
    write_json({k: dict(sorted(v.items())) for k, v in reg.items()},
               outdir / "register_length_distribution.json")

    print("\nDistinctive entities per variety:")
    for v, d in sorted(result.items(), key=lambda kv: -kv[1]["n_idioms"]):
        tops = "، ".join(x["entity"] for x in d["distinctive_entities"][:6])
        print(f"  {v:<6} (n={d['n_idioms']:>5})  {tops}")
    return {"varieties": len(result)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
CMDS = {"stats": cmd_stats, "entities": cmd_entities, "cluster": cmd_cluster,
        "semantic": cmd_semantic, "variety": cmd_variety}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="all",
                   choices=list(CMDS) + ["all"])
    p.add_argument("--input", default="data/idioms/ar/idioms_merged_llm_formatted.jsonl")
    p.add_argument("--outdir", default="data/idioms/ar/analysis")
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--n_clusters", type=int, default=12)
    p.add_argument("--max_entities", type=int, default=1500)
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--min_variety_size", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.86,
                   help="Cosine threshold for semantic clustering.")
    p.add_argument("--self-test", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    rows = load_rows(args.input)
    outdir = Path(args.outdir)
    logger.info("loaded %d Arabic idioms", len(rows))
    todo = list(CMDS) if args.command == "all" else [args.command]
    summary = {}
    for c in todo:
        logger.info("=== %s ===", c)
        summary[c] = CMDS[c](rows, outdir, args)
    write_json(summary, outdir / "analysis_summary.json")
    return 0


def self_test() -> int:
    import tempfile
    rows = [
        {"output": {"idiom": "أهل مكة أدرى بشعابها", "entities": ["مكة", "شعابها"],
                    "literal_meanings": ["سكان مكة أعرف بطرقها"],
                    "figurative_meanings": ["يضرب لمن هو أعلم بشأن نفسه"],
                    "figurative_meanings_en": NAN, "examples": NAN,
                    "variety": ["arb"], "variety_region": ["Classical Arabic"],
                    "register": "classical"},
         "meta": {"sources": ["tahaalselwii_classical"]}},
        {"output": {"idiom": "الحمي مكة والطريق", "entities": ["الحمي", "مكة"],
                    "literal_meanings": NAN,
                    "figurative_meanings": ["يضرب لمن هو أعلم بشأن نفسه تماما"],
                    "figurative_meanings_en": NAN, "examples": NAN,
                    "variety": ["arz"], "variety_region": ["Egyptian"],
                    "register": "colloquial"},
         "meta": {"sources": ["tahaalselwii_colloquial"]}},
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        args = build_parser().parse_args(["stats"])
        st = cmd_stats(rows, out, args)
        assert st["total_idioms"] == 2
        assert st["field_coverage"]["entities"]["pct"] == 100.0
        assert st["field_coverage"]["examples"]["pct"] == 0.0
        assert st["by_register"]["classical"] == 1
        # {مكة, شعابها, الحمي} — مكة occurs in both rows
        assert st["unique_entities"] == 3, st["unique_entities"]
        assert (out / "statistics.json").exists()

        # entity counting merges ال- and diacritic variants
        c = entity_counter([{"output": {"entities": ["الحمي", "حمي", "مَكَّة", "مكة"]}}])
        assert c.most_common(1)[0][1] == 2, c

        # variety contrast runs and separates the two varieties
        a2 = build_parser().parse_args(["variety"])
        a2.min_variety_size, a2.min_count = 1, 1
        v = cmd_variety(rows, out, a2)
        assert v["varieties"] == 2
        got = json.loads((out / "variety_contrast.json").read_text())
        assert set(got) == {"arb", "arz"}

        # label shaping must not crash and must return a string
        assert isinstance(ar_label("مكة"), str) and ar_label("مكة")
        # an Arabic-capable font must be discoverable (else plots are tofu boxes)
        import matplotlib
        matplotlib.use("Agg")
        assert setup_arabic_font() is not None, "no Arabic font on this machine"

        # fld() honours the NAN convention
        assert fld(rows[1], "literal_meanings") == []
        assert fld(rows[0], "entities") == ["مكة", "شعابها"]

    print("all analyze_ar_idioms.py self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
