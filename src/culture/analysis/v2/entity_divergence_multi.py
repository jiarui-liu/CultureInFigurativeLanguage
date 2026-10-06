#!/usr/bin/env python3
"""Task 4: the prompt-independent divergence score for en-zh, en-hi and en-ar under ONE protocol.

1. Take the top-300 entities (by mentions) of zh / hi / ar.
2. The LLM translates each (shown with its surface form and one example idiom) into English
   singular nouns: a primary translation and up to two close synonyms, or "none".
3. Source entities with the same primary translation are merged (e.g. Arabic singular/plural
   forms, Hindi oblique forms); the English side is every English idiom whose entity list
   contains the primary translation (synonyms are not used: they are often loose).
4. Up to 20 idioms per side are sampled (seed 0), as in the original en-zh analysis; each idiom's
   figurative meaning is paraphrased into English by the same local LLM (Qwen3.5-27B-FP8) for all
   languages (gloss space) and also embedded natively.
5. Divergence and size-matched baselines as in divergence.py; headline statistics use entities
   with >=5 idioms per side.

    PYTHONPATH=src python src/culture/analysis/v2/entity_divergence_multi.py
"""
import json
import os
import random
from collections import defaultdict

import numpy as np

from common import OUT, dump, embed, entity_counter, entity_index, load_kb, surface_forms
from divergence import full_set_calibration, size_matched, summarize_size_matched
from entity_divergence_en_zh import idiom_vectors_native, stack
from gloss import gloss
from culture.bidirectional.llm_api import parse_json
from local_llm import PRIMARY, generate

LANG_NAME = {"zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
TOP = 300
CAP = 20
MIN_N = 5

TR_PROMPT = """Translate each {lang} noun below into English, as it is used in the example idiom.
Give the primary English translation as a lower-case singular noun (one or two words), plus up to two close English synonyms that could replace it. If the word is not a noun or has no noun equivalent, give "none".

{items}

Return a JSON object mapping each id (as a string) to {{"en": "<primary>", "syn": ["<synonym>", ...]}}. Return only the JSON object."""


def translate_entities(lang, ents, sf, examples):
    lines = []
    for i, e in enumerate(ents, 1):
        shown = sf[e].most_common(1)[0][0] if sf.get(e) else e
        lines.append(f"{i}. {shown}  |  example: {examples.get(e, '')}")
    batches = [list(range(i, min(i + 50, len(ents)))) for i in range(0, len(ents), 50)]
    prompts = []
    for b in batches:
        items = []
        for k, j in enumerate(b, 1):
            items.append(f"{k}." + lines[j].split(".", 1)[1])
        prompts.append(TR_PROMPT.format(lang=LANG_NAME[lang], items="\n".join(items)))
    outs = generate(prompts, tag="v2_entity_translate_primary", model=PRIMARY, max_tokens=8000)
    tr = {}
    for b, o in zip(batches, outs):
        d = parse_json(o) or {}
        for k, j in enumerate(b, 1):
            v = d.get(str(k)) if isinstance(d, dict) else None
            if isinstance(v, dict) and isinstance(v.get("en"), str):
                prim = v["en"].strip().lower()
                syn = [s.strip().lower() for s in (v.get("syn") or []) if isinstance(s, str)]
                tr[ents[j]] = {"en": prim, "syn": syn}
    return tr


def main():
    rng = random.Random(0)
    en_kb = load_kb("en")
    en_idx = entity_index(en_kb)
    en_fig = {r["idiom"]: r["fig"] for r in en_kb}
    results = {"method": __doc__, "pairs": {}}
    all_sets = {}
    for lang in ("zh", "hi", "ar"):
        kb = load_kb(lang)
        c = entity_counter(kb)
        idx = entity_index(kb)
        sf = surface_forms(lang, kb)
        examples = {}
        for r in kb:
            for e in r["entities"]:
                examples.setdefault(e, r["idiom"])
        top = [e for e, _ in c.most_common(TOP)]
        tr = translate_entities(lang, top, sf, examples)
        groups = defaultdict(list)
        for e in top:
            t = tr.get(e)
            if t and t["en"] and t["en"] != "none":
                groups[t["en"]].append(e)
        recs = []
        for en_ent, src_ents in groups.items():
            # English side: the primary translation only. The LLM's "synonyms" are often loose
            # (tiger -> cat, fish -> prey) and would mix in unrelated imagery; they are kept in
            # entity_translations_*.json for reference but not used for retrieval.
            syn = {en_ent}
            en_ids = sorted({j for s in syn for j in en_idx.get(s, []) if en_kb[j]["fig"]})
            src_ids = sorted({j for e in src_ents for j in idx.get(e, []) if kb[j]["fig"]})
            if not en_ids:
                continue
            recs.append({"entity_en": en_ent, "entities_src": src_ents, "en_synonyms_used": sorted(s for s in syn if s in en_idx),
                         "n_en_full": len(en_ids), "n_src_full": len(src_ids),
                         "en_sample": [en_kb[j]["idiom"] for j in (rng.sample(en_ids, CAP) if len(en_ids) > CAP else en_ids)],
                         "src_sample": [kb[j]["idiom"] for j in (rng.sample(src_ids, CAP) if len(src_ids) > CAP else src_ids)]})
        src_fig = {r["idiom"]: r["fig"] for r in kb}
        all_sets[lang] = (recs, src_fig)
        results["pairs"][f"en-{lang}"] = {
            "top_entities": TOP, "translated": len(tr),
            "translated_not_none": sum(1 for t in tr.values() if t["en"] != "none"),
            "distinct_english_translations": len(groups), "matched_to_english_entities": len(recs),
            "mention_share_of_top300": round(sum(c[e] for e in top) / sum(c.values()), 4)}
        dump({e: tr.get(e) for e in top}, f"entity_translations_{lang}_en.json")

    # ---------------- gloss all sampled idioms (one vLLM session), then embed
    en_items = {i: en_fig[i] for lang in all_sets for r in all_sets[lang][0] for i in r["en_sample"]}
    g_en = gloss("en", list(en_items.items()))
    g_src = {}
    for lang, (recs, src_fig) in all_sets.items():
        items = {i: src_fig[i] for r in recs for i in r["src_sample"]}
        g_src[lang] = gloss(lang, list(items.items()))
    gk_en = list(g_en)
    gv_en = dict(zip(gk_en, embed([g_en[k] for k in gk_en])))
    nv_en = idiom_vectors_native(list(en_items.items()))
    for lang, (recs, src_fig) in all_sets.items():
        gk = list(g_src[lang])
        gv = dict(zip(gk, embed([g_src[lang][k] for k in gk])))
        nv = idiom_vectors_native([(i, src_fig[i]) for r in recs for i in r["src_sample"]])
        out = results["pairs"][f"en-{lang}"]
        for name, (VE, VS) in {"gloss": (gv_en, gv), "native": (nv_en, nv)}.items():
            A = [stack(VE, r["en_sample"]) for r in recs]
            B = [stack(VS, r["src_sample"]) for r in recs]
            q = [i for i in range(len(recs)) if len(A[i]) >= MIN_N and len(B[i]) >= MIN_N]
            per = size_matched([A[i] for i in q], [B[i] for i in q], k=5, reps=30)
            summ = summarize_size_matched(per)
            cal = full_set_calibration(A, B, q)
            pct = np.array([cal[i]["percentile"] for i in q])
            summ["full_set_calibration"] = {
                "n_entities": len(q), "mean_percentile": round(float(pct.mean()), 4),
                "median_percentile": round(float(np.median(pct)), 4),
                "frac_true_pair_top1_en2src": round(float(np.mean([cal[i]["rank_a2b"] == 1 for i in q])), 4),
                "frac_percentile_le_0.05": round(float((pct <= 0.05).mean()), 4),
                "frac_percentile_ge_0.50": round(float((pct >= 0.50).mean()), 4),
                "mean_centroid_div_true": round(float(np.mean([cal[i]["centroid_div"] for i in q])), 4),
                "mean_centroid_div_random": round(float(np.mean([cal[i]["mean_div_random"] for i in q])), 4)}
            # Per-entity percentiles, for the ECDF in fig_divergence.
            summ["per_entity"] = [
                {"entity_en": recs[i]["entity_en"],
                 "percentile": round(float(cal[i]["percentile"]), 4),
                 "rank_a2b": int(cal[i]["rank_a2b"])}
                for i in q
            ]
            out[name] = summ
            out["n_qualifying_ge5"] = len(q)
            if name == "gloss":
                order = sorted(q, key=lambda i: -cal[i]["centroid_div"])
                row = lambda i: {"entity_en": recs[i]["entity_en"], "entities_src": recs[i]["entities_src"][:4],
                                 "n_en": len(A[i]), "n_src": len(B[i]), "n_en_full": recs[i]["n_en_full"],
                                 "n_src_full": recs[i]["n_src_full"],
                                 "centroid_div": round(cal[i]["centroid_div"], 4), "percentile": round(cal[i]["percentile"], 3)}
                out["top10_most_divergent"] = [row(i) for i in order[:10]]
                out["top10_least_divergent"] = [row(i) for i in order[::-1][:10]]
                out["per_entity"] = [row(i) for i in order]
        print(lang, json.dumps({k: v for k, v in out.items() if k not in ("per_entity",)}, ensure_ascii=False)[:1500])
    dump(results, "entity_divergence_multi.json")


if __name__ == "__main__":
    main()
