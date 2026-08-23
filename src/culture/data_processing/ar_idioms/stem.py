#!/usr/bin/env python3
"""Tier-2 light stemming + stem-sequence matching for Arabic idioms.

WHY
---
:mod:`normalize` fixes *orthographic* variation (diacritics, hamza carriers,
ة/ى) and buys a measured **47.7x** recall gain. It cannot fix *morphology*:
a verbal idiom's verb agrees with its subject and its object carries a pronoun,
so the interior of the idiom changes::

    citation   عينه فيه          surface   عينها فيها
    citation   طول باله          surface   تطول بالك
    citation   اكل عليه الدهر وشرب   surface   اكل عليها الدهر وشرب

Measured on 60k FineWeb-2 `arb_Arab` docs (see
``docs/literature_reviews/arabic_idiom_resources.md`` §7):

===============================  =====  ==============  =========
mode                              docs  distinct idioms  precision
===============================  =====  ==============  =========
S1 surface-exact (normalize only)   307             229  --
S2 stem-exact, CONTIGUOUS (here)    409             284  ~95%
S3 stem + gaps <= 2                 442             310  ~50%  <-- rejected
===============================  =====  ==============  =========

So: contiguous stem matching only. Gapped matching is deliberately NOT
implemented — it roughly halves precision, and the literature agrees (SAMER's
best global setting is max-gap 2 with reordering off; an unbounded
bag-of-lemmas scores P=0.41).

HOW
---
Both sides are reduced to a space-joined *stem sequence* and matched with the
same Aho-Corasick substring machinery already used elsewhere. Wrapping each
side in sentinel spaces makes a substring hit equivalent to a **contiguous
token subsequence**, so no partial-token false positives are possible.

CRITICAL TUNING
---------------
``MIN_STEM_LEN = 2``, not 3. A 3-character floor refuses to strip ``ها`` from
``فيها`` (which would leave the 2-char ``في``) and that single guard accounted
for most residual misses in the gold set.

Run ``python stem.py`` for the self-test.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from culture.data_processing.ar_idioms.normalize import (  # noqa: E402
    normalize_ar,
    normalize_idiom_for_matching,
)

__all__ = [
    "light_stem",
    "stem_tokens",
    "stem_key",
    "StemMatcher",
    "MIN_STEM_LEN",
]

# Never strip an affix if fewer than this many characters would remain.
# MUST be 2 — see module docstring.
MIN_STEM_LEN = 2

# Only match idioms with at least this many tokens; 1-2 token "idioms" produce
# almost all false positives under stem matching.
MIN_STEM_TOKENS = 3

_RE_ARABIC_TOKEN = re.compile(r"[ء-ي]+")

# Proclitic cluster: [conjunction][preposition/future][definite article].
# Longest-first alternatives inside each group. `لل` is the ل+ال contraction.
_RE_PROCLITIC = re.compile(
    r"^(?:[وف])?"        # و and/ ف so
    r"(?:لل|[بكلس])?"    # لل (li+al), or ب/ك/ل/س
    r"(?:ال)?"           # definite article
)

# Enclitic pronouns and common verbal suffixes, longest first so that e.g. `هما`
# is tried before `ه`.
_ENCLITICS: Tuple[str, ...] = (
    "هما", "كما", "هن", "هم", "ها", "كن", "كم", "نا", "ني", "وا",
    "ون", "ين", "ات", "ان", "تم", "تن", "ه", "ك", "ي",
)


def light_stem(token: str) -> str:
    """Strip at most one enclitic and at most one proclitic cluster.

    Deliberately conservative: this is a *matching aid*, not a linguistic
    lemmatizer. An affix is only removed when at least :data:`MIN_STEM_LEN`
    characters remain.

    ORDER MATTERS — enclitic first. Doing proclitics first mangles words that
    merely *begin* with a clitic letter: ``فيها`` would lose its ``ف`` (leaving
    ``يها``) while ``فيه`` became ``يه``, so the two would no longer match. Taking
    the suffix off first yields ``في`` for both.

    NOT stripped: imperfect-verb prefixes (ي/ت/ن/أ). They collide with far too
    many ordinary nouns (تاريخ، نور، يوم…) to be safe, so a conjugated verb like
    ``تطول`` will not match the citation ``طول``. This is a known, accepted
    limitation — the measured residual misses are dominated by *pronoun
    suffixes*, which are handled.
    """
    if len(token) <= MIN_STEM_LEN:
        return token

    for suf in _ENCLITICS:
        if token.endswith(suf) and len(token) - len(suf) >= MIN_STEM_LEN:
            token = token[: -len(suf)]
            break

    m = _RE_PROCLITIC.match(token)
    if m and m.end() and len(token) - m.end() >= MIN_STEM_LEN:
        token = token[m.end():]
    return token


def stem_tokens(text: str, normalize: bool = True) -> List[str]:
    """Normalize -> tokenize on Arabic letters -> light-stem each token."""
    if normalize:
        text = normalize_ar(text)
    return [light_stem(t) for t in _RE_ARABIC_TOKEN.findall(text)]


def stem_key(text: str, normalize: bool = True) -> str:
    """Space-joined stem sequence wrapped in sentinels.

    The sentinels are what make an Aho-Corasick substring hit equivalent to a
    contiguous *token* subsequence rather than a raw character substring.
    """
    toks = stem_tokens(text, normalize=normalize)
    return f" {' '.join(toks)} " if toks else ""


class StemMatcher:
    """Contiguous stem-sequence matcher over an idiom inventory.

    Falls back to a plain scan when ``pyahocorasick`` is unavailable so the
    module stays importable (and testable) without the dependency.
    """

    def __init__(self, idioms: Iterable[str], min_tokens: int = MIN_STEM_TOKENS):
        self.min_tokens = min_tokens
        self._patterns: Dict[str, str] = {}   # stem key -> original idiom
        self.skipped_short = 0
        for idiom in idioms:
            key = self._pattern_key(idiom)
            if key is None:
                self.skipped_short += 1
                continue
            # First writer wins; keeps behaviour deterministic for duplicates.
            self._patterns.setdefault(key, idiom)

        self._automaton = None
        try:
            import ahocorasick
        except ImportError:
            return
        if not self._patterns:
            return
        A = ahocorasick.Automaton()
        for key, idiom in self._patterns.items():
            A.add_word(key, idiom)
        A.make_automaton()
        self._automaton = A

    def _pattern_key(self, idiom: str) -> Optional[str]:
        """Stem key for an inventory entry, or None if too short to be safe."""
        norm = normalize_idiom_for_matching(idiom)
        toks = stem_tokens(norm, normalize=False)
        if len(toks) < self.min_tokens:
            return None
        return f" {' '.join(toks)} "

    def __len__(self) -> int:
        return len(self._patterns)

    def find(self, text: str) -> Set[str]:
        """Return the set of inventory entries occurring in ``text``."""
        hay = stem_key(text)
        if not hay:
            return set()
        if self._automaton is not None:
            return {idiom for _, idiom in self._automaton.iter(hay)}
        return {idiom for key, idiom in self._patterns.items() if key in hay}

    def contains(self, text: str) -> bool:
        hay = stem_key(text)
        if not hay:
            return False
        if self._automaton is not None:
            for _ in self._automaton.iter(hay):
                return True
            return False
        return any(key in hay for key in self._patterns)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def self_test() -> int:
    # --- light_stem ------------------------------------------------------- #
    assert light_stem("الدهر") == "دهر"        # ال
    assert light_stem("وشرب") == "شرب"          # و
    assert light_stem("بالبيت") == "بيت"        # ب + ال
    assert light_stem("للرجل") == "رجل"         # لل
    assert light_stem("عينها") == "عين"         # enclitic ها
    assert light_stem("عينه") == "عين"          # enclitic ه
    assert light_stem("بالك") == "بال"          # enclitic ك

    # THE critical case: min stem length must be 2, not 3.
    assert light_stem("فيها") == "في", "MIN_STEM_LEN must allow a 2-char stem"
    assert light_stem("فيه") == "في"

    # Conservative: never strip below the floor, never strip twice.
    assert light_stem("له") == "له"             # too short to touch
    assert light_stem("هم") == "هم"
    assert light_stem("في") == "في"

    # Order regression: proclitic-first would give يها / يه and break the match.
    assert light_stem("فيها") == light_stem("فيه") == "في"

    # --- stem sequences match across inflection --------------------------- #
    # Real pairs from the measured gold set (citation vs attested surface).
    # These are the enclitic-pronoun cases that dominate the residual misses.
    pairs = [
        ("عينه فيه",                 "وهي عينها فيها من زمان"),
        ("اكل عليه الدهر وشرب",      "المقتضيات القانونيه التي اكل عليها الدهر وشرب"),
        ("خرج من ايده",              "الموضوع خرج من ايدهم خلاص"),
    ]
    m = StemMatcher([p[0] for p in pairs], min_tokens=2)
    for citation, surface in pairs:
        hits = m.find(surface)
        assert citation in hits, f"stem matching missed: {citation!r} in {surface!r}"

    # KNOWN LIMITATION (documented in light_stem): imperfect-verb prefixes are
    # not stripped, so a conjugated verb does not match its citation form.
    lim = StemMatcher(["طول باله"], min_tokens=2)
    assert not lim.find("لازم تطول بالك شوية"), (
        "if this now passes, verbal-prefix stripping was added — update the docs"
    )

    # --- contiguity is enforced (no gapped matches) ------------------------ #
    g = StemMatcher(["اكل عليه الدهر وشرب"], min_tokens=3)
    assert g.find("المقتضيات التي اكل عليها الدهر وشرب اليوم")    # contiguous -> hit
    assert not g.find("اكل عليها الدهر ثم وشرب")                  # 'ثم' inserted -> no hit
    assert not g.find("شرب الدهر عليها اكل")                      # reordered -> no hit

    # --- token-boundary safety (sentinels) --------------------------------- #
    b = StemMatcher(["عين في"], min_tokens=2)
    assert b.find("عينها فيها")                                   # real token match
    assert not b.find("معينفي")                                   # glued -> must NOT match

    # --- short entries are skipped ----------------------------------------- #
    s = StemMatcher(["خبر ابيض", "اكل عليه الدهر وشرب"])          # default min_tokens=3
    assert len(s) == 1 and s.skipped_short == 1

    # --- normalization is applied on both sides ---------------------------- #
    # Vocalized citation form vs bare web text.
    v = StemMatcher(["أكَلَ عَلَيْه الدَّهْرُ وَشَرِبَ"], min_tokens=3)
    assert v.find("التي اكل عليها الدهر وشرب")

    # --- degenerate input --------------------------------------------------- #
    assert StemMatcher([]).find("نص عربي") == set()
    assert not StemMatcher(["اكل عليه الدهر وشرب"]).contains("")
    assert stem_key("") == ""

    import importlib.util
    backend = "pyahocorasick" if importlib.util.find_spec("ahocorasick") else "fallback scan"
    print(f"all stem.py self-tests passed (backend: {backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
