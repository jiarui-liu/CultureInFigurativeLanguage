#!/usr/bin/env python3
"""Are cross-language differences in IdiomAtlas driven by languages or by sources / registers?

(a) Entity-category distributions (15 categories: 8 concrete types + 7 abstract subtypes, the labels
    behind Figure fig:entities) by language versus by source within a language. Pairwise
    bias-corrected Cramer's V for every pair of groups (between-language pairs and within-language
    source pairs), Poisson-bootstrap CIs over idioms, permutation tests for the within-language source
    effect, and a mutual-information decomposition I(C; L,S) = I(C; L) + I(C; S | L).
(b) Robustness of the language contrast when each language is restricted to one source or register
    (all combinations), Arabic classical vs colloquial as separate groups, and the same-entity valence
    and meaning-map divergence (imagery_items.jsonl) with Arabic split by register.
(c) 9B IdiomAtlas-MC Arabic contrasts by register / source of the test idiom; MIDI-ar by dialect
    (copied from idiomtext_vs_random.json); register of the Arabic symbolism-probe evidence idioms.
(d) Hindi source: dialect parallels listed by the dictionary.

Usage: PYTHONPATH=src python -m culture.analysis.rebuttal.source_heterogeneity \
         --out docs/paper_stats/rebuttal/source_heterogeneity.json
"""
import argparse
import difflib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src/culture/analysis/v2"))
from common import KB, norm_entity  # noqa: E402
from culture.data_processing.ar_idioms.normalize import normalize_ar  # noqa: E402

D = REPO / "culture/data/idioms"
LAB = REPO / "docs/paper_stats/analysis_v2"
BID = Path("/data/group_data/r3lit_culture_pretrain/culture/bidir")
LANGS = ["en", "zh", "hi", "ar"]
TYPES = ["animal", "body_mind", "nature_cosmos", "food_drink", "kinship_social", "religion_supernatural",
         "occupation_economy", "artefact_household"]
SUBTYPES = ["time_change", "speech_name", "life_death_fate", "morality_truth", "action_conflict",
            "quantity_space", "residual_other"]
CATS = TYPES + SUBTYPES
EN_SRC = {"phrases_reformatted.jsonl": "en_dictionary", "english_idioms_reformatted.jsonl": "en_wiktionary",
          "reformatted_en_idiom_meaning.jsonl": "en_idiomkb"}
REPS = 1000
rng = np.random.default_rng(20261124)


# ----------------------------------------------------------------------------- data
def labels15(lang):
    t = json.load(open(LAB / f"entity_typology_labels_{lang}.json"))
    s = json.load(open(LAB / f"entity_subtypology_labels_{lang}.json"))
    out = {}
    for e, v in t.items():
        ty = v.get("type")
        if ty in TYPES:
            out[e] = ty
        elif ty == "abstract_other":
            st = (s.get(e) or {}).get("subtype")
            out[e] = st if st in SUBTYPES else "residual_other"
    return out


def load(lang):
    """Rows of the KB file the paper uses, with normalized entities and source metadata."""
    merged = None
    if lang in ("en", "zh"):
        merged = [json.loads(l) for l in open(D / lang / "idioms_merged.jsonl", encoding="utf-8")]
    rows = []
    for line in open(KB[lang], encoding="utf-8"):
        o = json.loads(line)
        out = o.get("output") or {}
        raw = out.get("entities")
        raw = raw if isinstance(raw, list) else []
        ents = list(dict.fromkeys(n for n in (norm_entity(e, lang) for e in raw) if n))
        idiom = out.get("idiom") or o.get("idiom")
        r = {"idiom": idiom, "ents": ents}
        if lang == "en":
            m = merged[o["index"]]
            srcs = [EN_SRC[m[k]["source"]] for k in sorted(m) if k.startswith("source") and isinstance(m[k], dict)]
            r["sources"], r["source"] = sorted(set(srcs)), srcs[0]
        elif lang == "zh":
            m = merged[o["index"]]
            srcs = []
            for k in ["source1", "source2", "source3", "source4"]:
                v = m.get(k)
                for x in (v if isinstance(v, list) else [v] if v else []):
                    if isinstance(x, dict) and x.get("source_input"):
                        srcs.append("zh_" + x["source_input"])
            r["sources"], r["source"] = sorted(set(srcs)), srcs[0]
        elif lang == "hi":
            r["sources"], r["source"] = ["hi_brhat_lokokti_kosh"], "hi_brhat_lokokti_kosh"
        else:
            srcs = ["ar_" + s for s in o["meta"]["sources"]]
            r["sources"], r["source"] = sorted(set(srcs)), srcs[0]
            r["register"] = out.get("register")
            r["variety"] = (o["meta"].get("varieties_raw") or [None])[0]
        rows.append(r)
    return rows


def matrix(rows, lab):
    """Idiom x category counts (each idiom counts each distinct labelled entity once)."""
    X = np.zeros((len(rows), len(CATS)), dtype=np.float32)
    for i, r in enumerate(rows):
        for e in r["ents"]:
            c = lab.get(e)
            if c:
                X[i, CATS.index(c)] += 1
    return X


# ----------------------------------------------------------------------------- statistics
def cramers_v(tab, corrected=True):
    tab = np.asarray(tab, float)
    tab = tab[:, tab.sum(0) > 0]
    tab = tab[tab.sum(1) > 0]
    n = tab.sum()
    r, k = tab.shape
    if r < 2 or k < 2:
        return float("nan")
    chi2 = chi2_contingency(tab, correction=False)[0]
    phi2 = chi2 / n
    if not corrected:
        return float(np.sqrt(phi2 / min(r - 1, k - 1)))
    # Bergsma (2013) bias correction
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / min(rc - 1, kc - 1)))


def jsd(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    f = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0))
    return float((f(p, m) + f(q, m)) / 2)


def boot_tables(Xs, reps=REPS):
    """Poisson-bootstrap replicate tables: list over groups of (reps x K)."""
    return [rng.poisson(1.0, size=(reps, X.shape[0])).astype(np.float32) @ X for X in Xs]


def pair_stats(Xa, Xb, Ba, Bb):
    ta, tb = Xa.sum(0), Xb.sum(0)
    v = cramers_v(np.vstack([ta, tb]))
    bv = np.array([cramers_v(np.vstack([Ba[i], Bb[i]])) for i in range(len(Ba))])
    return {"V": round(v, 4), "V_ci95": [round(float(np.nanpercentile(bv, 2.5)), 4), round(float(np.nanpercentile(bv, 97.5)), 4)],
            "JSD_bits": round(jsd(ta, tb), 4), "n_idioms": [int(Xa.shape[0]), int(Xb.shape[0])],
            "n_mentions": [int(ta.sum()), int(tb.sum())], "_boot": bv}


def mi(tab):
    tab = np.asarray(tab, float)
    p = tab / tab.sum()
    pr, pc = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / (pr @ pc)[nz])))


def adjusted_residuals(tab):
    tab = np.asarray(tab, float)
    n = tab.sum()
    r, c = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
    exp = r @ c / n
    return (tab - exp) / np.sqrt(exp * (1 - r / n) * (1 - c / n))


# ----------------------------------------------------------------------------- (a) + (b)
def entity_analysis(rows, X):
    out = {}
    # overall language effect
    tabL = np.vstack([X[L].sum(0) for L in LANGS])
    out["language_V_uncorrected"] = round(cramers_v(tabL, corrected=False), 4)
    out["language_V"] = round(cramers_v(tabL), 4)
    out["category_shares"] = {L: {c: round(float(x), 4) for c, x in zip(CATS, tabL[i] / tabL[i].sum())}
                              for i, L in enumerate(LANGS)}

    # groups: languages, and sources within language (primary source; also exclusive-source entries)
    groups = {}
    for L in LANGS:
        groups[L] = np.arange(len(rows[L]))
    src_groups = {}
    for L in LANGS:
        by = defaultdict(list)
        for i, r in enumerate(rows[L]):
            if len(r["sources"]) == 1:
                by[r["sources"][0]].append(i)
        src_groups[L] = {s: np.array(ix) for s, ix in by.items() if len(ix) >= 150}
    # Chinese sources are nested (Xinhua lists almost every chengyu), so split Xinhua-only entries
    # from entries that other collections also list (more common chengyu)
    src_groups["zh"] = {"zh_xinhua_only": np.array([i for i, r in enumerate(rows["zh"]) if r["sources"] == ["zh_xinhua"]]),
                        "zh_multi_source": np.array([i for i, r in enumerate(rows["zh"]) if len(r["sources"]) > 1])}
    reg_groups = defaultdict(list)
    for i, r in enumerate(rows["ar"]):
        if r["register"] in ("classical", "colloquial"):
            reg_groups["ar_" + r["register"]].append(i)
    reg_groups = {k: np.array(v) for k, v in reg_groups.items()}
    out["exclusive_source_sizes"] = {L: {s: int(len(ix)) for s, ix in g.items()} for L, g in src_groups.items()}
    out["arabic_register_sizes"] = {k: int(len(v)) for k, v in reg_groups.items()}

    # within-language source effect: V with permutation p
    within = {}
    for L in LANGS:
        g = src_groups[L]
        if len(g) < 2:
            continue
        keys = sorted(g)
        idx = np.concatenate([g[k] for k in keys])
        lab = np.concatenate([[j] * len(g[k]) for j, k in enumerate(keys)])
        XX = X[L][idx]
        tab = np.vstack([XX[lab == j].sum(0) for j in range(len(keys))])
        v = cramers_v(tab)
        null = []
        for _ in range(500):
            pl = rng.permutation(lab)
            null.append(cramers_v(np.vstack([XX[pl == j].sum(0) for j in range(len(keys))])))
        within[L] = {"sources": keys, "V": round(v, 4), "perm_null_mean": round(float(np.mean(null)), 4),
                     "perm_p": float((np.sum(np.array(null) >= v) + 1) / (len(null) + 1))}
    # Arabic register effect (classical vs colloquial)
    a, b = X["ar"][reg_groups["ar_classical"]], X["ar"][reg_groups["ar_colloquial"]]
    rt = np.vstack([a.sum(0), b.sum(0)])
    rr = adjusted_residuals(rt)
    within["ar_register"] = {"V": round(cramers_v(rt), 4),
                             "shares_classical": {c: round(float(x), 3) for c, x in zip(CATS, rt[0] / rt[0].sum())},
                             "shares_colloquial": {c: round(float(x), 3) for c, x in zip(CATS, rt[1] / rt[1].sum())},
                             "classical_over_represented": [c for c, _ in sorted(zip(CATS, rr[0]), key=lambda t: -t[1])[:4]],
                             "colloquial_over_represented": [c for c, _ in sorted(zip(CATS, rr[1]), key=lambda t: -t[1])[:4]]}
    out["within_language_source_V"] = within

    # pairwise V: every pair of groups among {languages} U {exclusive sources} U {Arabic registers}
    G = {L: X[L] for L in LANGS}
    for L in LANGS:
        for s, ix in src_groups[L].items():
            G[s] = X[L][ix]
    for k, ix in reg_groups.items():
        G[k] = X["ar"][ix]
    lang_of = {k: (k if k in LANGS else k.split("_")[0]) for k in G}
    names = list(G)
    B = dict(zip(names, boot_tables([G[k] for k in names])))
    pairs = {}
    for x, y in itertools.combinations(names, 2):
        if lang_of[x] == lang_of[y] and (x in LANGS or y in LANGS):
            continue  # a language vs its own subset is not an informative pair
        if lang_of[x] == lang_of[y] and (x in reg_groups) != (y in reg_groups):
            continue  # register groups and source groups of the same language overlap
        if (x in LANGS) != (y in LANGS):
            continue  # compare full KBs with full KBs and subsets with subsets
        pairs[f"{x}|{y}"] = pair_stats(G[x], G[y], B[x], B[y])
    kinds = defaultdict(list)
    for k, v in pairs.items():
        x, y = k.split("|")
        if x in LANGS and y in LANGS:
            kinds["between_language (full KBs)"].append(k)
        elif lang_of[x] == lang_of[y]:
            kinds["within_language (source or register pairs)"].append(k)
        elif x not in LANGS and y not in LANGS:
            kinds["between_language (single-source subsets)"].append(k)
    summ = {}
    for kd, ks in kinds.items():
        vs = [pairs[k]["V"] for k in ks]
        summ[kd] = {"n_pairs": len(ks), "median_V": round(float(np.median(vs)), 4), "min_V": round(min(vs), 4),
                    "max_V": round(max(vs), 4), "pairs": {k: pairs[k]["V"] for k in sorted(ks, key=lambda k: -pairs[k]["V"])}}
    # probability that a random within-language pair exceeds a random between-language (single-source) pair
    wl = kinds["within_language (source or register pairs)"]
    bl = kinds["between_language (single-source subsets)"]
    bw = np.array([pairs[k]["_boot"] for k in wl])
    bb = np.array([pairs[k]["_boot"] for k in bl])
    diff = np.nanmedian(bb, 0) - np.nanmedian(bw, 0)
    summ["median_between_minus_median_within"] = {"delta_V": round(float(np.median([pairs[k]["V"] for k in bl]) - np.median([pairs[k]["V"] for k in wl])), 4),
                                                  "ci95": [round(float(np.nanpercentile(diff, 2.5)), 4), round(float(np.nanpercentile(diff, 97.5)), 4)]}
    out["pairwise_summary"] = summ
    out["pairwise"] = {k: {kk: vv for kk, vv in v.items() if kk != "_boot"} for k, v in pairs.items()}

    # Arabic registers vs other languages, side by side
    ar_side = {}
    for reg in ["ar_classical", "ar_colloquial"]:
        ar_side[reg] = {}
        for L in ["en", "zh", "hi"]:  # Arabic register subset vs the other language's full KB
            ps = pair_stats(G[reg], G[L], B[reg], B[L])
            ar_side[reg][L] = {kk: ps[kk] for kk in ("V", "V_ci95")}
    k = "ar_classical|ar_colloquial" if "ar_classical|ar_colloquial" in pairs else "ar_colloquial|ar_classical"
    ar_side["classical_vs_colloquial"] = {kk: pairs[k][kk] for kk in ("V", "V_ci95")}
    out["arabic_registers_vs_languages"] = ar_side

    # MI decomposition over exclusive-source entries: I(C;L,S) = I(C;L) + I(C;S|L)
    tabs = {L: np.vstack([G[s].sum(0) for s in sorted(src_groups[L])]) for L in LANGS}
    def decomp(tabs):
        full = np.vstack([tabs[L] for L in LANGS])
        lang = np.vstack([tabs[L].sum(0) for L in LANGS])
        i_ls, i_l = mi(full), mi(lang)
        return i_ls, i_l, i_ls - i_l
    i_ls, i_l, i_s = decomp(tabs)
    bt = []
    for rep in range(REPS):
        bt.append(decomp({L: np.vstack([B[s][rep] for s in sorted(src_groups[L])]) for L in LANGS}))
    bt = np.array(bt)
    out["mutual_information_bits"] = {
        "I(C;L,S)": round(i_ls, 5), "I(C;L)": round(i_l, 5), "I(C;S|L)": round(i_s, 5),
        "share_language": round(i_l / i_ls, 3),
        "share_language_ci95": [round(float(np.percentile(bt[:, 1] / bt[:, 0], 2.5)), 3), round(float(np.percentile(bt[:, 1] / bt[:, 0], 97.5)), 3)],
        "note": "exclusive-source entries only; Hindi has one source so contributes nothing to I(C;S|L)"}

    # (b) language contrast with every language restricted to a single source / register
    choices = {"en": sorted(src_groups["en"]), "zh": sorted(src_groups["zh"]), "hi": ["hi"],
               "ar": ["ar_classical", "ar_colloquial"] + sorted(src_groups["ar"])}
    full_res = adjusted_residuals(tabL)
    headline = {"en": ["body_mind", "artefact_household"], "zh": ["nature_cosmos"],
                "hi": ["kinship_social", "food_drink", "artefact_household"],
                "ar": ["religion_supernatural", "kinship_social", "animal"]}
    combos = []
    for ce in choices["en"]:
        for cz in choices["zh"]:
            for ca in choices["ar"]:
                ks = [ce, cz, "hi", ca]
                tab = np.vstack([G[k].sum(0) for k in ks])
                res = adjusted_residuals(tab)
                head = {L: {c: round(float(res[i, CATS.index(c)]), 1) for c in headline[L]} for i, L in enumerate(LANGS)}
                ok = sum(int(np.sign(res[i, CATS.index(c)]) == np.sign(full_res[i, CATS.index(c)]))
                         for i, L in enumerate(LANGS) for c in headline[L])
                combos.append({"groups": ks, "V": round(cramers_v(tab), 4), "headline_residuals": head,
                               "headline_signs_kept": f"{ok}/{sum(len(v) for v in headline.values())}",
                               "n_idioms": [int(G[k].shape[0]) for k in ks]})
    vs = [c["V"] for c in combos]
    flips = Counter()
    for c in combos:
        for i, L in enumerate(LANGS):
            for cat, v in c["headline_residuals"][L].items():
                if np.sign(v) != np.sign(full_res[i, CATS.index(cat)]):
                    flips[f"{L}:{cat} ({'/'.join(g for g in c['groups'] if g.startswith(L))})"] += 1
    out["single_source_restriction"] = {"full_KB_V": out["language_V"], "n_combinations": len(combos),
                                        "V_min": min(vs), "V_median": round(float(np.median(vs)), 4), "V_max": max(vs),
                                        "full_headline_residuals": {L: {c: round(float(full_res[i, CATS.index(c)]), 1) for c in headline[L]} for i, L in enumerate(LANGS)},
                                        "headline_sign_flips": dict(flips.most_common()),
                                        "combinations": combos}
    return out


# ----------------------------------------------------------------------------- (b) valence / meaning maps
def imagery_analysis(ar_reg):
    items = [json.loads(l) for l in open(REPO / "docs/paper_stats/v6/imagery_items.jsonl", encoding="utf-8")]
    E = np.load(REPO / "docs/paper_stats/v6/imagery_gloss_emb.npy")
    E = E / np.linalg.norm(E, axis=1, keepdims=True)

    def val(x):
        v = [x[k]["valence"] for k in ("primary", "second") if x.get(k) and isinstance(x[k].get("valence"), (int, float))]
        return float(np.mean(v)) if v else None

    grp = []
    miss = 0
    for x in items:
        g = x["lang"]
        if g == "ar":
            reg = ar_reg.get(x["idiom"]) or ar_reg.get(normalize_ar(x["idiom"]))
            if reg is None:
                miss += 1
            g = f"ar_{reg}" if reg in ("classical", "colloquial") else "ar_other"
        grp.append(g)
    by = defaultdict(list)
    for i, x in enumerate(items):
        by[x["anchor"]].append(i)
    MIN = 10
    res = {"arabic_items_unmatched": miss, "min_items_per_group": MIN, "anchors": {}}
    p_reg, p_lang, d_reg, d_lang, c_reg, c_lang = [], [], [], [], [], []
    for a, ix in by.items():
        G = defaultdict(list)
        for i in ix:
            G[grp[i]].append(i)
        if len(G.get("ar_classical", [])) < MIN or len(G.get("ar_colloquial", [])) < MIN:
            continue
        others = [L for L in ["en", "zh", "hi"] if len(G.get(L, [])) >= MIN]
        if not others:
            continue
        vals = {g: np.array([v for v in (val(items[i]) for i in G[g]) if v is not None]) for g in G}
        cent = {g: E[G[g]].mean(0) for g in G}
        cos = lambda u, w: float(u @ w / np.linalg.norm(u) / np.linalg.norm(w))
        nmin = min(len(vals["ar_classical"]), len(vals["ar_colloquial"]))
        row = {"n": {g: len(G[g]) for g in G}, "mean_valence": {g: round(float(v.mean()), 3) for g, v in vals.items() if len(v)}}
        # register contrast
        pr = mannwhitneyu(vals["ar_classical"], vals["ar_colloquial"]).pvalue
        dr = abs(vals["ar_classical"].mean() - vals["ar_colloquial"].mean())
        cr = 1 - cos(cent["ar_classical"], cent["ar_colloquial"])
        # Arabic (each register, size-matched) vs each other language, size-matched subsamples
        pl, dl, cl = [], [], []
        for reg in ["ar_classical", "ar_colloquial"]:
            for L in others:
                m = min(nmin, len(vals[L]))
                a_ = rng.choice(vals[reg], m, replace=False)
                b_ = rng.choice(vals[L], m, replace=False)
                pl.append(mannwhitneyu(a_, b_).pvalue)
                dl.append(abs(vals[reg].mean() - vals[L].mean()))
                cl.append(1 - cos(cent[reg], cent[L]))
        row.update({"register_p": pr, "register_abs_diff": round(dr, 3), "register_centroid_dist": round(cr, 4),
                    "ar_vs_lang_median_p": float(np.median(pl)), "ar_vs_lang_median_abs_diff": round(float(np.median(dl)), 3),
                    "ar_vs_lang_median_centroid_dist": round(float(np.median(cl)), 4)})
        res["anchors"][a] = row
        p_reg.append(pr); d_reg.append(dr); c_reg.append(cr)
        p_lang.extend(pl); d_lang.append(np.median(dl)); c_lang.append(np.median(cl))

    def bh(p):
        p = np.asarray(p)
        o = np.argsort(p)
        q = p[o] * len(p) / (np.arange(len(p)) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty_like(q)
        out[o] = np.minimum(q, 1)
        return out
    n = len(p_reg)
    res["summary"] = {
        "n_anchors": n,
        "valence_sig_q05_classical_vs_colloquial": int((bh(p_reg) < 0.05).sum()),
        "valence_sig_q05_arabic_register_vs_other_language_sizematched": f"{int((bh(p_lang) < 0.05).sum())}/{len(p_lang)}",
        "median_abs_valence_diff_register": round(float(np.median(d_reg)), 3),
        "median_abs_valence_diff_ar_vs_language": round(float(np.median(d_lang)), 3),
        "median_centroid_dist_register": round(float(np.median(c_reg)), 4),
        "median_centroid_dist_ar_vs_language": round(float(np.median(c_lang)), 4),
        "anchors_where_register_gap_smaller_than_language_gap_valence": int(np.sum(np.array(d_reg) < np.array(d_lang))),
        "anchors_where_register_gap_smaller_than_language_gap_meaning": int(np.sum(np.array(c_reg) < np.array(c_lang))),
    }
    # the paper's 4-language Kruskal test with Arabic restricted to one register
    for reg in ["ar_classical", "ar_colloquial"]:
        ps = []
        for a, ix in by.items():
            G = defaultdict(list)
            for i in ix:
                g = grp[i]
                if g.startswith("ar") and g != reg:
                    continue
                v = val(items[i])
                if v is not None:
                    G[g].append(v)
            G = {g: v for g, v in G.items() if len(v) >= 5}
            if len(G) >= 2:
                ps.append(kruskal(*G.values()).pvalue)
        res["summary"][f"kruskal_language_sig_q05_with_{reg}_only"] = f"{int((bh(ps) < 0.05).sum())}/{len(ps)}"
    return res


# ----------------------------------------------------------------------------- (c) model side
def mcnemar_free_boot(a, b, reps=10000):
    d = a - b
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    bs = d[idx].mean(1)
    return [round(100 * d.mean(), 1), round(100 * float(np.percentile(bs, 2.5)), 1), round(100 * float(np.percentile(bs, 97.5)), 1)]


def model_side(ar_rows):
    meta = {}
    for r in ar_rows:
        meta.setdefault(r["idiom"].strip(), r)
        meta.setdefault(normalize_ar(r["idiom"].strip()), r)
    out = {}
    runs = {"unfiltered": "Random-CPT", "untagged": "Idiom-Text", "cpt": "Idiom+Meaning", "base": "Base"}
    for split in ["seen", "unseen"]:
        recs = {}
        for r in runs:
            j = json.load(open(BID / "eval9b/ar" / r / f"idiomatlas_mc_ar_{split}.json"))
            recs[r] = {x["qid"]: x for x in j["records"]}
        qids = sorted(set.intersection(*[set(v) for v in recs.values()]))
        grp = {}
        unmatched = 0
        for q in qids:
            idiom = recs["base"][q]["idiom"].strip()
            m = meta.get(idiom) or meta.get(normalize_ar(idiom))
            if m is None:
                unmatched += 1
                continue
            grp[q] = (m["register"], m["source"])
        res = {"unmatched": unmatched}
        for key, pos in [("register", 0), ("source", 1)]:
            cnt = Counter(v[pos] for v in grp.values())
            res[key] = {}
            for g, n in cnt.most_common():
                if n < 40:
                    res[key][g] = {"n": n, "note": "too few items"}
                    continue
                qs = [q for q in grp if grp[q][pos] == g]
                A = {r: np.array([int(recs[r][q]["correct_norm"]) for q in qs]) for r in runs}
                res[key][g] = {"n": n, "acc": {runs[r]: round(100 * A[r].mean(), 1) for r in runs},
                               "Idiom-Text - Random-CPT": mcnemar_free_boot(A["untagged"], A["unfiltered"]),
                               "Idiom+Meaning - Random-CPT": mcnemar_free_boot(A["cpt"], A["unfiltered"]),
                               "Idiom+Meaning - Idiom-Text": mcnemar_free_boot(A["cpt"], A["untagged"])}
        out[f"idiomatlas_mc_ar_{split}"] = res
    # MIDI-ar by dialect (computed by idiomtext_vs_random.py)
    p = REPO / "docs/paper_stats/rebuttal/idiomtext_vs_random.json"
    if p.exists():
        rows = json.load(open(p))["rows"]
        out["midi_ar_by_dialect"] = {r["task"]: {k: v for k, v in r.items() if k in ("n", "acc", "Idiom-Text - Random-CPT", "Idiom+Meaning - Random-CPT", "Idiom+Meaning - Idiom-Text")}
                                     for r in rows if r["task"].startswith("midi_ar")}
    # symbolism probe evidence registers
    items = json.load(open(BID / "probe_v2/ar/items.json"))
    probe = [json.loads(l) for l in open(BID / "eval_data/mc/symbolism_v2_ar_letter.jsonl", encoding="utf-8")]
    kept = []
    for pq in probe:
        ent = pq["meta"]["entity"]
        cand = [it for it in items if it["p"]["tgt"] == ent and it.get("opts") and all(o in pq["context"] for o in it["opts"])]
        if cand:
            kept.append((pq["qid"], cand[0]))
    def reg_of(idiom):
        m = meta.get(idiom.strip()) or meta.get(normalize_ar(idiom.strip()))
        return m["register"] if m else "unmatched"
    ev_all, ev_gold, item_major = Counter(), Counter(), {}
    for qid, it in kept:
        regs = [reg_of(x["idiom"]) for x in it["p"]["tgt_idioms"]]
        ev_all.update(regs)
        # gold_evidence strings are copied by the generator LLM and often lose diacritics or words:
        # map each to the closest idiom actually shown to it (difflib ratio >= 0.6 on normalized text)
        shown = [(normalize_ar(x["idiom"]), x["idiom"]) for x in it["p"]["tgt_idioms"]]
        g = []
        for ev in it["o"].get("gold_evidence", []):
            ne = normalize_ar(ev)
            best = max(shown, key=lambda t: difflib.SequenceMatcher(None, ne, t[0]).ratio()) if shown else None
            ok = best and difflib.SequenceMatcher(None, ne, best[0]).ratio() >= 0.6
            g.append(reg_of(best[1]) if ok else "unmatched")
        ev_gold.update(g)
        c = Counter(x for x in (g or regs) if x in ("classical", "colloquial"))
        item_major[qid] = c.most_common(1)[0][0] if c else "unmatched"
    tot = lambda c: {k: [v, round(v / sum(c.values()), 3)] for k, v in c.most_common()}
    sym = {"n_probe_items_matched": len(kept), "evidence_idioms_shown": tot(ev_all), "gold_evidence_idioms": tot(ev_gold),
           "items_by_majority_gold_evidence_register": dict(Counter(item_major.values()))}
    # probe contrast by majority register (small n)
    rec = {}
    for r in ["unfiltered", "untagged", "cpt"]:
        j = json.load(open(BID / "eval9b/ar" / r / "symbolism_v2_ar_letter.json"))
        rec[r] = {x["qid"]: x for x in j["records"]}
    for g in ["classical", "colloquial"]:
        qs = [q for q, v in item_major.items() if v == g and all(q in rec[r] for r in rec)]
        if len(qs) < 10:
            continue
        A = {r: np.array([int(rec[r][q]["correct"]) for q in qs]) for r in rec}
        lure_of = {pp["qid"]: pp["meta"]["lure"] for pp in probe}
        lure = {r: np.array([int(rec[r][q]["pred"] == lure_of[q]) for q in qs]) for r in rec}
        sym[f"probe_{g}_majority"] = {"n": len(qs), "acc": {runs[r]: round(100 * A[r].mean(), 1) for r in rec},
                                      "lure_rate": {runs[r]: round(100 * lure[r].mean(), 1) for r in rec},
                                      "Idiom+Meaning - Random-CPT (acc)": mcnemar_free_boot(A["cpt"], A["unfiltered"]),
                                      "Idiom+Meaning - Random-CPT (lure)": mcnemar_free_boot(lure["cpt"], lure["unfiltered"])}
    out["symbolism_probe_ar"] = sym
    cnt = json.load(open(BID / "ar_idiom_counts.json"))
    sc = {normalize_ar(k): v for k, v in cnt.get("seen_counts", {}).items()}
    ever = {normalize_ar(k) for k in cnt.get("ever", [])} | set(sc)
    cov = defaultdict(lambda: [0, 0, 0])
    for r in ar_rows:
        k = normalize_ar(r["idiom"].strip())
        for g in (r["source"], "register:" + str(r["register"])):
            cov[g][0] += 1
            cov[g][1] += int(k in ever)
            cov[g][2] += int(sc.get(k, 0))
    out["arabic_web_coverage"] = {g: {"idioms": a, "share_found_on_web": round(b / a, 3), "training_docs": c}
                                  for g, (a, b, c) in sorted(cov.items(), key=lambda t: -t[1][0])}
    return out


# ----------------------------------------------------------------------------- (d) Hindi source
def hindi_note():
    H = REPO / "culture/data/hi_idioms"
    par = [json.loads(l) for l in open(H / "cross_language_parallels_hi.jsonl", encoding="utf-8")]
    raw = sum(1 for _ in open(H / "entries_raw.jsonl", encoding="utf-8"))
    c = Counter(p["language"] for x in par for p in x["parallels"])
    return {"raw_entries": raw, "headwords_with_parallels": len(par), "parallels_total": sum(len(x["parallels"]) for x in par),
            "parallels_by_variety": dict(c.most_common()),
            "note": "Headwords are (standard) Hindi proverbs; the dictionary lists 'tulaniya' (compare) parallels in "
                    "regional varieties and other languages. IdiomAtlas keeps only the Hindi headword and its "
                    "explanation; parallels are stored separately and not pooled into the Hindi inventory."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {L: load(L) for L in LANGS}
    lab = {L: labels15(L) for L in LANGS}
    X = {L: matrix(rows[L], lab[L]) for L in LANGS}
    out = {"categories": CATS, "kb_sizes": {L: len(rows[L]) for L in LANGS}}
    out["entities"] = entity_analysis(rows, X)
    ar_reg = {}
    for r in rows["ar"]:
        ar_reg.setdefault(r["idiom"], r["register"])
        ar_reg.setdefault(normalize_ar(r["idiom"]), r["register"])
    out["imagery"] = imagery_analysis(ar_reg)
    out["model"] = model_side(rows["ar"])
    out["hindi_source"] = hindi_note()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False, default=float)
    e = out["entities"]
    print("language V", e["language_V"], "(uncorrected", e["language_V_uncorrected"], ")")
    print("within-language source V", json.dumps(e["within_language_source_V"]))
    for k, v in e["pairwise_summary"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "pairs"} if isinstance(v, dict) else v)
    print("MI", e["mutual_information_bits"])
    print("arabic registers vs languages", json.dumps(e["arabic_registers_vs_languages"]))
    s = e["single_source_restriction"]
    print("single-source V", s["V_min"], s["V_median"], s["V_max"], "of", s["n_combinations"], "| headline signs kept:",
          Counter(c["headline_signs_kept"] for c in s["combinations"]))
    print("imagery", json.dumps(out["imagery"]["summary"]))
    print("flips", json.dumps(s["headline_sign_flips"], ensure_ascii=False))
    print("ar register", json.dumps(e["within_language_source_V"]["ar_register"]))
    print("probe", json.dumps(out["model"]["symbolism_probe_ar"], ensure_ascii=False))
    print("coverage", json.dumps(out["model"]["arabic_web_coverage"], ensure_ascii=False))
    print("hindi", json.dumps(out["hindi_source"], ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
