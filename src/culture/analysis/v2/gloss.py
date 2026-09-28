"""Put figurative meanings of idioms from any language into one comparable space:
a concise English paraphrase written by an LLM from the dictionary definitions only.

All languages (including English) go through the same prompt and model, so that differences
between languages are not artefacts of dictionary style or of the cross-lingual embedding gap.
Model: the PRIMARY local model (Qwen3.5-27B-FP8, vLLM). One idiom per prompt; cached under tag v2_gloss_primary.
"""
from local_llm import PRIMARY, generate

LANG_NAME = {"en": "English", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
PROMPT = """Here is a {lang} idiom or proverb with its dictionary definition(s).

Idiom: {idiom}
Definition(s): {defs}

Write a concise English paraphrase (one sentence, at most 25 words) of its FIGURATIVE meaning, using only the given definitions. Do not describe the literal image, etymology, source text or story, and do not add meanings that are not in the definitions. If several distinct figurative senses are given, keep each briefly, separated by "; ".
Answer with the paraphrase only."""


def _clip(s, n):
    return s if len(s) <= n else s[:n] + " ..."


def _clean(r):
    r = (r or "").strip().strip('"').strip()
    for pre in ("Paraphrase:", "Figurative meaning:"):
        if r.lower().startswith(pre.lower()):
            r = r[len(pre):].strip()
    return r.split("\n")[0].strip()


def gloss(lang, items):
    """items: list of (idiom, [figurative meaning strings]). Returns {idiom: english gloss}."""
    seen, uniq = set(), []
    for idiom, fig in items:
        if idiom not in seen and fig:
            seen.add(idiom)
            uniq.append((idiom, fig))
    prompts = [PROMPT.format(lang=LANG_NAME[lang], idiom=i, defs=_clip(" | ".join(f), 1500)) for i, f in uniq]
    outs = generate(prompts, tag="v2_gloss_primary", model=PRIMARY, max_tokens=150)
    return {i: _clean(o) for (i, _), o in zip(uniq, outs) if _clean(o)}
