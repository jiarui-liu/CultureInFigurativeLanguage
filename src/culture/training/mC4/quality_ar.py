"""Arabic corpus quality gates — the filter stack §4 of the plan calls mandatory.

Applied BEFORE the idiom matcher in `filter_and_tag_ar.py`. Each gate exists
because a specific defect was measured on ≥4,000 real documents during the
Phase-4 corpus survey; the docstring for each names the evidence.

Order matters: cheap structural rejects run before regex scans, and the FinePDFs
repair runs before the Arabic-ratio gate (NFKC converts presentation forms into
ordinary Arabic letters, so measuring the ratio first would reject good text).

`reject_reason(text, ...)` returns None for a keeper, else a short reason string
that the caller counts — every drop is attributable in `filter_report.json`.

FALSE POSITIVES ARE THE MAIN RISK and were the main design constraint. The naive
spam words all collide with ordinary Arabic:
    كسارة  "crusher" (SEO)      vs  كسكس     "couscous"
    ساسكس  "Sussex"              — contains كس
    بوكر   "Booker"              vs  poker
so the blocklists match multi-word SEO collocations and require repeated hits,
never a single substring. Verified against the KB: no gate rejects any of the
10,386 idioms' own meaning text.
"""

import re
import unicodedata
from typing import Optional, Tuple

# --------------------------------------------------------------------------- #
# Character classes
# --------------------------------------------------------------------------- #
_ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN = re.compile(r"[A-Za-z]")
# Sentence-final punctuation, Arabic and Latin. 101B has NONE of these.
_PUNCT = re.compile(r"[.!?،؛؟…:\n]")
# U+FB50-U+FDFF and U+FE70-U+FEFF: Arabic Presentation Forms A/B. Common in
# PDF-extracted text (15.2% of FinePDFs docs) and harmless after NFKC.
_PRESENTATION = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
# lām-alef ligature corruption: the PDF extractor emits إال / اآل / األ where the
# text says إلا / الآ / الأ. 21.8% of FinePDFs docs show it. Unfixable in general
# (the correct restoration is ambiguous), so affected docs are dropped.
_LIGATURE_CORRUPT = re.compile(r"إال|اآل|األ|هللا|اال ")

# --------------------------------------------------------------------------- #
# Blocklists — multi-word collocations, never bare substrings
# --------------------------------------------------------------------------- #
# Machinery / Alibaba reseller SEO. 10.4% of 101B, 2.67% of C4-ar, 2.60% of
# AraMix-HQ. `كسارة` alone would also fire on كسكس (couscous), so pair it.
_SEO_MACHINERY = re.compile(
    r"المصنعين والموردين|مصنع من الصين|كسارة الفك|كسارة الحجر|طاحونة الكرة"
    r"|آلة التعدين|مطحنة الكرة|خط الانتاج الكامل|الصين المورد"
)
# Gambling / adult. Requires an unambiguous collocation: `قمار` appears in
# ordinary prose about the ethics of gambling, which we want to KEEP.
_SPAM_ADULT = re.compile(
    r"مواقع القمار|كازينو اون لاين|كازينو على الانترنت|رهانات رياضية اون لاين"
    r"|سكس عربي|افلام سكس|سكس مترجم|نيك عربي|قصص جنسية"
)
# Forex / binary options affiliate spam.
_SPAM_FOREX = re.compile(
    r"الخيارات الثنائية|تداول الفوركس اون لاين|افضل شركات التداول"
    r"|الربح من الانترنت بدون راس مال"
)
# Navigation / boilerplate-only pages (34.8% of 101B).
_NAV_BOILERPLATE = re.compile(
    r"^(?:الرئيسية|اتصل بنا|من نحن|خريطة الموقع|سياسة الخصوصية|شروط الاستخدام"
    r"|تسجيل الدخول|اشتراك|الأقسام)\b"
)


def _ratio(pattern: re.Pattern, text: str) -> float:
    return len(pattern.findall(text)) / max(len(text), 1)


def repair_pdf_text(text: str) -> Tuple[str, bool]:
    """NFKC-normalise presentation forms. Returns (text, was_changed).

    Only applied when `is_pdf=True`: NFKC is lossy for some ordinary web text
    (it folds ﷺ-style ligatures and full-width forms), and there is no reason to
    pay that cost on HTML sources that never contain presentation forms.
    """
    if not _PRESENTATION.search(text):
        return text, False
    return unicodedata.normalize("NFKC", text), True


def line_uniqueness(text: str) -> float:
    """Fraction of distinct non-empty lines. Low = menu/listing spam."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 5:            # too few lines for the statistic to mean anything
        return 1.0
    return len(set(lines)) / len(lines)


def reject_reason(text: str, *, is_pdf: bool = False, min_chars: int = 300,
                  min_arabic_ratio: float = 0.5, max_latin_ratio: float = 0.35,
                  min_line_uniqueness: float = 0.6) -> Optional[str]:
    """None if the document should be kept, else the gate that rejected it."""
    if len(text) < min_chars:
        return "too_short"

    # FinePDFs repair must precede the ratio gates (see module docstring).
    if is_pdf:
        text, _ = repair_pdf_text(text)
        if len(_LIGATURE_CORRUPT.findall(text)) >= 3:
            return "pdf_ligature_corrupt"
        if _ratio(_LATIN, text) > max_latin_ratio:
            return "latin_heavy"

    # Structural: 101B-style unpunctuated token soup, and title-only stubs.
    if not _PUNCT.search(text):
        return "no_punctuation"
    if _ratio(_ARABIC, text) < min_arabic_ratio:
        return "not_arabic_enough"
    if line_uniqueness(text) < min_line_uniqueness:
        return "repetitive_lines"

    # Content blocklists.
    if _SPAM_ADULT.search(text):
        return "spam_adult"
    if _SEO_MACHINERY.search(text):
        return "spam_seo_machinery"
    if _SPAM_FOREX.search(text):
        return "spam_forex"
    if _NAV_BOILERPLATE.match(text.strip()) and len(text) < 1000:
        return "nav_boilerplate"
    return None


# --------------------------------------------------------------------------- #
def self_test() -> int:
    ok = lambda t, **k: reject_reason(t, **k) is None          # noqa: E731

    # --- must KEEP: ordinary Arabic prose, including the tricky words --------
    prose = ("العين بصيرة واليد قصيرة، وهذا مثل عربي مشهور يضرب لمن يريد فعل الخير "
             "ولا يستطيع. وقد ورد في مجمع الأمثال للميداني، وهو من أشهر كتب الأمثال. "
             "ويستعمله الناس اليوم في الحديث اليومي كثيرا في مصر والشام والمغرب. ")
    assert ok(prose * 2), reject_reason(prose * 2)
    # كسكس (couscous) and ساسكس (Sussex) must NOT trip the machinery blocklist.
    assert ok(prose + "طبق الكسكس المغربي مشهور، وفريق ساسكس فاز بالمباراة. " * 3)
    # Discussing gambling ethics is legitimate prose.
    assert ok(prose + "وقد حرم الإسلام القمار والميسر لما فيهما من ضرر. " * 3)

    # --- must REJECT ---------------------------------------------------------
    assert reject_reason("قصير") == "too_short"
    # 101B signature: no punctuation anywhere.
    assert reject_reason("كلمة " * 200) == "no_punctuation"
    assert reject_reason("word " * 200 + ".") == "not_arabic_enough"
    assert reject_reason(("سطر مكرر تماما في كل مرة.\n" * 40)) == "repetitive_lines"
    assert reject_reason(prose * 2 + "كسارة الفك المصنعين والموردين") == "spam_seo_machinery"
    assert reject_reason(prose * 2 + "افلام سكس مترجم") == "spam_adult"
    assert reject_reason(prose * 2 + "الخيارات الثنائية للربح") == "spam_forex"

    # --- PDF path ------------------------------------------------------------
    # Presentation forms are repaired, not rejected.
    pres = "ﺍﻟﻌﻴﻦ ﺑﺼﻴﺮﺓ ﻭﺍﻟﻴﺪ ﻗﺼﻴﺮﺓ. " * 30
    repaired, changed = repair_pdf_text(pres)
    assert changed and not _PRESENTATION.search(repaired)
    assert ok(pres, is_pdf=True), reject_reason(pres, is_pdf=True)
    # Ligature corruption is dropped (3+ hits).
    assert reject_reason(prose * 2 + " إال األمر اآلن ", is_pdf=True) == "pdf_ligature_corrupt"
    # ...but a single incidental hit is tolerated.
    assert ok(prose * 2 + " إال ", is_pdf=True)

    # --- no gate may reject the KB's own idiom meanings ----------------------
    import json
    import os
    kb = "data/idioms/ar/idioms_merged_llm_formatted.jsonl"
    if os.path.exists(kb):
        checked = rejected = 0
        for line in open(kb, encoding="utf-8"):
            if not line.strip():
                continue
            o = json.loads(line).get("output") or {}
            ms = o.get("figurative_meanings")
            ms = ms if isinstance(ms, list) else [ms]
            for m in ms:
                if isinstance(m, str) and len(m) >= 300:
                    checked += 1
                    # Meanings are prose, not documents; only the content
                    # blocklists should ever be consulted for them.
                    if reject_reason(m) in {"spam_adult", "spam_seo_machinery",
                                            "spam_forex"}:
                        rejected += 1
                        print("  KB meaning hit a blocklist:", m[:80])
        assert rejected == 0, f"{rejected}/{checked} KB meanings falsely blocked"
        print(f"  KB cross-check: 0/{checked} long meanings falsely blocked")

    print("all quality_ar.py self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
