#!/usr/bin/env python3
"""Arabic text normalization for idiom matching.

WHY THIS EXISTS
---------------
The corpus filter (``culture/training/mC4/download_and_filter_mc4.py``) matches an
idiom inventory against web text with Aho-Corasick *exact substring* matching, and
normalizes nothing except lowercasing for English. That is fine for Chinese
(chengyu are frozen, unvocalized 4-character units) but **fails almost completely
for Arabic**, because:

- Idiom dictionaries are **vocalized** (with tashkīl/harakat): ``إنَّ دَوَاءَ الشَّقِّ أنْ تَحُوصَهُ``
- Web text is **unvocalized**:                                ``ان دواء الشق ان تحوصه``
- Hamza carriers vary freely in real text: ``أ إ آ ٱ`` are all written ``ا``
- ``ى``/``ي`` (alif maqsura / ya) and ``ة``/``ه`` (taa marbuta / ha) are interchanged

Measured on real dictionary entries vs. typical web spellings, raw matching
recovered **0/4**; normalizing both sides recovered **4/4** (see ``self_test``).

WHAT THIS DOES (and does not) SOLVE
-----------------------------------
Solved — deterministic, cheap, no false-positive risk in practice:
  diacritics, tatweel, hamza-carrier variation, alif maqsura, taa marbuta,
  Arabic-Indic digits, invisible joiners, whitespace.

NOT solved here (genuine morphology — needs a different strategy):
  verb conjugation inside an idiom (ضرب → ضربت/يضرب), clitics attached to
  *internal* words, insertion of material, word-order change, dialect respelling.
  Note that proclitics (و ف ب ك ل ال س) on the **first** word and enclitic pronouns
  on the **last** word are harmless for substring matching — they only add
  characters outside the matched span. Only *internal* affixation breaks it.

Pure standard library on purpose: this runs over ~10^8 documents, so it must be
fast and dependency-free. The transformations mirror CAMeL Tools'
``normalize_alef_ar`` / ``normalize_alef_maksura_ar`` / ``normalize_teh_marbuta_ar``
/ ``dediac_ar`` so results stay comparable to that toolchain.

Usage::

    from culture.data_processing.ar_idioms.normalize import normalize_ar
    normalize_ar("إنَّ دَوَاءَ الشَّقِّ")   # -> 'ان دواء الشق'

Run ``python normalize.py`` to execute the self-test.
"""

import re
import unicodedata
from typing import Iterable, List, Optional

__all__ = [
    "normalize_ar",
    "dediacritize",
    "normalize_orthography",
    "normalize_idiom_for_matching",
    "strip_quote_furniture",
    "arabic_ratio",
    "looks_arabic",
    "MIN_PATTERN_CHARS",
]

# Patterns shorter than this (after normalization) are where essentially all
# false positives come from — 47.7% of Kinayat entries are <=2 tokens. Callers
# building a matcher should skip them.
MIN_PATTERN_CHARS = 10


# --------------------------------------------------------------------------- #
# Character classes
# --------------------------------------------------------------------------- #
# Harakat, Quranic annotation marks, superscript alef, and tatweel (kashida).
# NOTE: U+0621 (ء, standalone hamza) is a LETTER and is deliberately NOT removed.
_DIACRITICS = (
    "ؐ-ؚ"    # Arabic signs (sallallahou alayhe wassallam, etc.)
    "ً-ٟ"    # harakat + combining hamza/maddah marks
    "ٰ"           # superscript alef
    "ۖ-ۭ"    # Quranic annotation marks
    "࣓-ࣿ"    # Arabic Extended-A combining marks
)
_TATWEEL = "ـ"
_INVISIBLES = "​-‏‪-‮⁦-⁩﻿"  # ZWSP/ZWNJ/ZWJ/bidi

_RE_DIAC = re.compile(f"[{_DIACRITICS}{_TATWEEL}]")
_RE_INVISIBLE = re.compile(f"[{_INVISIBLES}]")
_RE_WS = re.compile(r"\s+")

# Hamza carriers -> bare alef (CAMeL: normalize_alef_ar)
_ALEF_FORMS = str.maketrans({
    "آ": "ا",  # آ alef with madda
    "أ": "ا",  # أ alef with hamza above
    "إ": "ا",  # إ alef with hamza below
    "ٱ": "ا",  # ٱ alef wasla
    "ٲ": "ا",  # ٲ
    "ٳ": "ا",  # ٳ
    "ٵ": "ا",  # ٵ
})
# alif maqsura -> ya (CAMeL: normalize_alef_maksura_ar)
_MAQSURA = str.maketrans({"ى": "ي"})          # ى -> ي
# taa marbuta -> ha (CAMeL: normalize_teh_marbuta_ar)
_TEH_MARBUTA = str.maketrans({"ة": "ه"})      # ة -> ه
# Aggressive (opt-in): other hamza seats -> their base letter
_HAMZA_SEATS = str.maketrans({"ؤ": "و", "ئ": "ي"})  # ؤ->و  ئ->ي
# Arabic-Indic + Extended Arabic-Indic digits -> ASCII
_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩"
    "۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
# Perso-Urdu letters that leak into Arabic web text and must fold to their Arabic
# counterparts (measured in FineWeb-2 arb_Arab: ی in 307 docs, ک in 154 docs).
# Mirrors CAMeL Tools' `arclean` mapper.
_PERSO_URDU = str.maketrans({
    "ی": "ي",  # U+06CC farsi yeh
    "ک": "ك",  # U+06A9 keheh
    "ڪ": "ك",  # U+06AA swash kaf
    "گ": "ك",  # U+06AF gaf
    "پ": "ب",  # U+067E peh
    "چ": "ج",  # U+0686 tcheh
    "ژ": "ز",  # U+0698 jeh
    "ۀ": "ه",  # U+06C0 heh with yeh above
    "ە": "ه",  # U+06D5 ae
})

_ARABIC_BLOCK = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
# Arabic Presentation Forms A/B (ligatures like ﷲ ﷺ, and isolated/initial/medial
# glyph variants). Rare, but NFKC is 26x slower than everything else combined, so
# only pay for it when such a character is actually present.
_RE_PRESENTATION = re.compile(r"[ﭐ-﷿ﹰ-ﻼ]")
# Quotation furniture wrapping dictionary entries. 100% of the 3,187 Taymur
# colloquial proverbs in `tahaalselwii` are wrapped in guillemets — leaving them
# in gives those entries ZERO recall.
_RE_QUOTE_FURNITURE = re.compile(r'^[\s«»"“”\'‘’\[\](){}]+|[\s«»"“”\'‘’\[\](){}]+$')
_RE_TRAILING_PUNCT = re.compile(r"[\s.،؛؟!:\-–—…]+$")


# --------------------------------------------------------------------------- #
# Core transforms
# --------------------------------------------------------------------------- #
def dediacritize(text: str) -> str:
    """Strip harakat, Quranic marks, superscript alef and tatweel.

    This is the single highest-impact step: dictionaries are vocalized, web text
    is not.
    """
    return _RE_DIAC.sub("", text)


def normalize_orthography(text: str, aggressive_hamza: bool = False) -> str:
    """Fold the orthographic variation that carries no meaning for matching.

    ``aggressive_hamza`` additionally folds ``ؤ→و`` and ``ئ→ي``. It raises recall
    slightly but can merge genuinely distinct words, so it is off by default.
    """
    text = text.translate(_PERSO_URDU)
    text = text.translate(_ALEF_FORMS)
    text = text.translate(_MAQSURA)
    text = text.translate(_TEH_MARBUTA)
    text = text.translate(_DIGITS)
    if aggressive_hamza:
        text = text.translate(_HAMZA_SEATS)
    return text


def strip_quote_furniture(text: str) -> str:
    """Remove wrapping quotes/brackets and trailing punctuation from an entry.

    Dictionary inventories wrap their headwords: every one of the 3,187 Taymur
    colloquial proverbs is ``«...»``. Those characters never appear around the
    proverb in running text, so leaving them in yields zero matches.
    """
    prev = None
    while prev != text:                       # nested furniture, e.g. «"...„»
        prev = text
        text = _RE_QUOTE_FURNITURE.sub("", text)
    return _RE_TRAILING_PUNCT.sub("", text).strip()


def normalize_ar(
    text: str,
    *,
    dediac: bool = True,
    orthography: bool = True,
    aggressive_hamza: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    """Full normalization pipeline. Apply to **both** the idiom list and the corpus.

    Order matters: NFC-compose first so decomposed hamza sequences (ا + U+0654)
    become the precomposed أ, which the alef table then folds to ا. Diacritics are
    stripped before orthography so combining marks cannot block the translations.
    """
    if not text:
        return ""
    # NFKC only when a presentation form is actually present (it is ~26x slower
    # than every other step combined); otherwise a guarded NFC, which is a ~50x
    # speedup over unconditional normalization on already-normal text.
    if _RE_PRESENTATION.search(text):
        text = unicodedata.normalize("NFKC", text)
    elif not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
    text = _RE_INVISIBLE.sub("", text)
    if dediac:
        text = dediacritize(text)
    if orthography:
        text = normalize_orthography(text, aggressive_hamza=aggressive_hamza)
    if collapse_whitespace:
        text = _RE_WS.sub(" ", text).strip()
    return text


def normalize_idiom_for_matching(idiom: str, **kwargs) -> str:
    """Idiom-side normalization: strip furniture + parentheticals, then normalize.

    Mirrors the parenthetical stripping the existing matcher already does for
    English, and additionally removes the quote furniture that dictionary
    inventories wrap around headwords (see :func:`strip_quote_furniture`).
    """
    base = re.sub(r"\([^)]*\)", "", idiom).strip() or idiom
    base = strip_quote_furniture(base) or base
    return normalize_ar(base, **kwargs)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def arabic_ratio(text: str) -> float:
    """Fraction of non-space characters that are in an Arabic Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(bool(_ARABIC_BLOCK.match(c)) for c in chars) / len(chars)


def looks_arabic(text: str, threshold: float = 0.5) -> bool:
    """Cheap guard against non-Arabic rows leaking into the idiom inventory."""
    return arabic_ratio(text) >= threshold


# --------------------------------------------------------------------------- #
# Self-test — real dictionary entries vs. typical unvocalized web spellings
# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Verify the normalizer on verbatim entries from the surveyed datasets."""
    # (dictionary citation form, realistic web-text sentence containing it)
    pairs = [
        ("إنَّ دَوَاءَ الشَّقِّ أنْ تَحُوصَهُ",
         "قالوا ان دواء الشق ان تحوصه وهذا مثل قديم"),          # tahaalselwii classical
        ("خَبَرَ أبْيَضْ", "يا خبر ابيض! ايه اللي حصل ده"),        # Kinayat
        ("أَجْرِ مْنَاوِلْ", "انا في الموضوع ده ماليش غير اجر مناول"),  # Kinayat
        ("إنَّ مِنَ الْبَيَانِ لَسِحْراً", "وقد ورد ان من البيان لسحرا في الحديث"),
        ("أهل مكة أدرى بشعابها", "يقولون أهل مكة أدرى بشعابها دائما"),   # already bare
    ]
    raw_hits = norm_hits = 0
    for dict_form, web in pairs:
        raw_hits += dict_form in web
        norm_hits += normalize_idiom_for_matching(dict_form) in normalize_ar(web)
    print(f"substring recall — raw: {raw_hits}/{len(pairs)}   normalized: {norm_hits}/{len(pairs)}")
    assert norm_hits == len(pairs), "normalization failed to recover a known match"

    # Unit checks on each transform.
    assert dediacritize("مُحَمَّدٌ") == "محمد"
    assert dediacritize("الرَّحْمَٰن") == "الرحمن"                  # superscript alef
    assert normalize_orthography("أإآٱ") == "اااا"
    assert normalize_orthography("مصطفى") == "مصطفي"               # ى -> ي
    assert normalize_orthography("مدرسة") == "مدرسه"               # ة -> ه
    assert normalize_orthography("٢٠٢٦") == "2026"
    assert normalize_ar("مـــحـــمـ__د".replace("_", "")) == "محمد"  # tatweel
    assert normalize_ar("  ا  ب  ") == "ا ب"                        # whitespace
    assert normalize_ar("") == ""

    # NFC: decomposed alef+hamza-above must fold like the precomposed form.
    assert normalize_ar("أحمد") == normalize_ar("أحمد")

    # ء (standalone hamza) is a letter and must survive.
    assert "ء" in normalize_ar("سماء")

    # aggressive_hamza is opt-in.
    assert normalize_ar("مسؤول") == "مسؤول"
    assert normalize_ar("مسؤول", aggressive_hamza=True) == "مسوول"

    # Parenthetical stripping on the idiom side.
    assert normalize_idiom_for_matching("خَبَرَ أبْيَضْ (كناية)") == "خبر ابيض"

    # --- REGRESSION: guillemets around Taymur colloquial entries -------------
    # 100% of the 3,187 colloquial proverbs are wrapped in «...». Verbatim entry:
    quoted = "«آخِرِ الحَيَاة الْمُوتْ»"
    web = "زي ما بيقولوا اخر الحياه الموت وخلاص"
    assert strip_quote_furniture(quoted) == "آخِرِ الحَيَاة الْمُوتْ"
    key = normalize_idiom_for_matching(quoted)
    assert not key.startswith("«") and "»" not in key, key
    assert key in normalize_ar(web), "guillemets must not block the match"
    # Nested / mixed furniture and trailing punctuation.
    assert strip_quote_furniture('«"مثل"»') == "مثل"
    assert strip_quote_furniture("مثل عربي.") == "مثل عربي"

    # --- Perso-Urdu letters leaking into Arabic web text ---------------------
    assert normalize_ar("کتاب") == normalize_ar("كتاب")   # ک U+06A9 -> ك
    assert normalize_ar("عربی") == normalize_ar("عربي")   # ی U+06CC -> ي

    # --- Presentation forms only pay for NFKC when present -------------------
    assert "الله" in normalize_ar("ﷲ")                     # U+FDF2 ligature
    assert normalize_ar("عربي") == "عربي"                   # untouched fast path

    # --- Short-pattern guard is exported and sane ----------------------------
    assert MIN_PATTERN_CHARS >= 10
    assert len(normalize_idiom_for_matching("خَبَرَ أبْيَضْ")) < MIN_PATTERN_CHARS

    # Idempotence — normalizing twice must not change anything.
    for _, web in pairs:
        once = normalize_ar(web)
        assert normalize_ar(once) == once

    # Language guard.
    assert looks_arabic("أهل مكة أدرى بشعابها")
    assert not looks_arabic("hello world")

    print("all normalize.py self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
