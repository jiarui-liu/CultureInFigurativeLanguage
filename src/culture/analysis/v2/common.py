"""Shared helpers for the v2 paper analyses (entity typology, divergence, meaning clusters).

Conventions mirror the paper's figure code (paper_writing/code/make_figures.py):
  * an entity *mention* = an entity listed for an idiom, de-duplicated within the idiom;
  * English entities are lower-cased and the six dictionary slot fillers are dropped;
  * Arabic entities are normalized with normalize_ar and a leading article (ال / لل) is stripped.
    Exception: the article strip is not applied when it would destroy the word الله (-> "له");
    we keep الله as "الله" (the figure's version maps it to "له", which is a bug).
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.environ.get("CULTURE_REPO", "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage")
DATA = os.path.join(REPO, "culture/data/idioms")
OUT = os.path.join(REPO, "docs/paper_stats/analysis_v2")
EMB_MODEL = os.environ.get("CULTURE_EMB_MODEL", "/data/group_data/r3lit_culture_pretrain/models/Qwen/Qwen3-Embedding-0.6B")
# gemini-3.8-flash was requested, but its free-tier quota (20 requests/day/project) was exhausted
# (a concurrent pipeline uses it); gemini-3.7-flash is the closest available model.
LLM = "gemini-3.7-flash"

# Arabic KB with entities: the enriched release on HF (Jerry9999/CultureInFigurativeLanguage,
# data/idioms/ar/), identical record count (10,386) to the local un-enriched file.
AR_KB = os.environ.get("CULTURE_AR_KB", os.path.join(OUT, "ar_hf/data/idioms/ar/idioms_merged_llm_formatted.jsonl"))
if not os.path.exists(AR_KB):
    AR_KB = os.path.join(DATA, "ar/idioms_merged_llm_formatted.jsonl")
KB = {
    "en": os.path.join(DATA, "en/idioms_merged_llm_formatted_figurative_only.jsonl"),
    "zh": os.path.join(DATA, "zh/idioms_merged_llm_formatted_figurative_only.jsonl"),
    "hi": os.path.join(DATA, "hi/idioms_merged_llm_formatted_figurative_only.jsonl"),
    "ar": AR_KB,
}
EN_SLOTS = {"something", "someone", "thing", "person", "place", "way"}

sys.path.insert(0, os.path.join(REPO, "src"))
from culture.data_processing.ar_idioms.normalize import normalize_ar  # noqa: E402


def norm_entity(e, lang):
    if not isinstance(e, str) or not e.strip() or e == "NAN":
        return None
    e = e.strip()
    if lang == "en":
        e = e.lower()
        return None if e in EN_SLOTS else e
    if lang == "ar":
        n = normalize_ar(e)
        if n in ("الله", "لله", "والله", "بالله"):
            return "الله"
        n = re.sub(r"^(ال|لل)", "", n)
        return n or None
    return e


def flatten(x):
    if x is None or x == "NAN":
        return []
    if isinstance(x, str):
        return [x] if x.strip() else []
    out = []
    for i in x:
        out.extend(flatten(i))
    return out


def load_kb(lang):
    """Return list of dicts: idiom, entities (normalized, deduped), fig (list[str]), raw_entities."""
    rows = []
    for line in open(KB[lang], encoding="utf-8"):
        o = json.loads(line)
        out = o.get("output") or {}
        raw = out.get("entities")
        raw = raw if isinstance(raw, list) else []
        ents, seen = [], set()
        for e in raw:
            n = norm_entity(e, lang)
            if n and n not in seen:
                seen.add(n)
                ents.append(n)
        rows.append({"idiom": out.get("idiom") or o.get("idiom"), "entities": ents,
                     "raw_entities": raw, "fig": flatten(out.get("figurative_meanings"))})
    return rows


def entity_counter(rows):
    c = Counter()
    for r in rows:
        c.update(r["entities"])
    return c


def entity_index(rows):
    idx = defaultdict(list)
    for i, r in enumerate(rows):
        for e in r["entities"]:
            idx[e].append(i)
    return idx


def surface_forms(lang, rows):
    """normalized entity -> Counter of raw surface forms (for showing the LLM a readable form)."""
    sf = defaultdict(Counter)
    for r in rows:
        for e in r["raw_entities"]:
            n = norm_entity(e, lang)
            if n:
                sf[n][e.strip()] += 1
    return sf


def dump(obj, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p


_model = None


def embed(texts, batch_size=32):
    import numpy as np
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMB_MODEL, device="cuda")
        _model.half()
        _model.max_seq_length = 512  # long Arabic commentaries otherwise OOM next to vLLM
    if not texts:
        return np.zeros((0, 1024), dtype="float32")
    return _model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=len(texts) > 5000, convert_to_numpy=True).astype("float32")
