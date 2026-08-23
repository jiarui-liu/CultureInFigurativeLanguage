"""Dataset loaders + prompt templating for the Arabic benchmarks.

Companion to :mod:`culture.evaluation.tasks` (Hindi) and
:mod:`culture.evaluation.tasks_zh` (Chinese): the shared dataclasses
(:class:`MCExample`, :class:`MCTask`) and helpers (``_first``,
``_resolve_gold_index``, ``_mc_fewshot_prefix``) are imported from that module
rather than re-implemented here.

Every task below is base-model MULTIPLE-CHOICE, scored by log-likelihood via
:meth:`culture.evaluation.scorer.HFModel.loglikelihood` and evaluated by the
existing :func:`culture.evaluation.run_eval.eval_mc` (no new scoring code).

Dimension 3 — Arabic idiom / figurative understanding
  ``ar_figurative``            Alyah figurative + DziriEval figurative (314 items, clean)

Dimension 4 — Arabic general cultural competence
  ``arabculture``              MBZUAI/ArabCulture, 13 countries, 3-way completion
  ``arabic_cultural_qa``       QCRI/ArabicCulturalQA, 4-way MCQ x 6 dialect variants
  ``arabicmmlu``               MBZUAI/ArabicMMLU, native Arab-region exams (2-5 options)
  ``global_piqa_ar``           Global-PIQA non-parallel, 13 Arabic variants, binary
  ``alyah``                    tiiuae/alyah-emirati-benchmark, 4-way (full set)
  ``dzirieval``                touati-kamel/DziriEval, Algerian Darija, 4-way
  ``global_piqa_ar_parallel``  Global-PIQA parallel arb/ary/arz — CONTROL, culture-agnostic

CONTAMINATION (see plans/arabic_pipeline_plan.md §5). The obvious Arabic idiom
benchmarks are unusable because our own training KB ingested them:
``menaattia/Kinayat`` 314/325 items in the KB, ``UBC-NLP/Jawaher-benchmark``
199/200 test items, ``Renad10/Absher`` 81/83 + 408/478, ``amthal-hassaniya``
319/319. They are NOT loadable here on purpose. The two figurative sources that
ARE used were measured contamination-free (Alyah 6/724 = 0.8 %, DziriEval
1/501 = 0.2 %, and those hits are incidental pan-Arab proverb overlap, not item
leakage). ``--drop_kb_overlap`` removes even those for a zero-overlap claim.

SCORING NOTES that the loader encodes deliberately:
- Alyah has a measured 57.5 % longest-option-is-gold bias (chance 25 %), so it is
  ``score_mode="continuation"`` -> the primary metric is length-normalised
  ``acc_norm``. Raw summed log-prob would be inflated.
- ArabicMMLU option count varies 2/3/4/5 (unlike CMMLU's fixed 4). The letter
  block is built from however many options the row actually has.
- ArabCulture rows carry ``should_discard``; rows flagged ``Yes`` are dropped.

Data access is local-file-first: pass ``--ar_data_dir`` pointing at the tree
built by ``download_ar.sh`` (default ``data/eval/ar/raw``). Parquet/TSV/CSV/JSONL
are all read locally; there is no HF fallback because several of these repos are
Xet-backed and stall behind proxies (use the curl-based downloader instead).
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from culture.evaluation.tasks import (
    LETTERS,
    MCExample,
    MCTask,
    _first,
    _mc_fewshot_prefix,
    _read_rows,
    _resolve_gold_index,
)

logger = logging.getLogger(__name__)

DEFAULT_AR_DATA_DIR = "data/eval/ar/raw"


# --------------------------------------------------------------------------- #
# Templates (edit here to tune prompting; kept explicit for transparency)
# --------------------------------------------------------------------------- #
# ArabCulture: the stem is a scene-setting sentence; options continue it. No
# interrogative framing — the natural continuation IS the task.
ARABCULTURE_TEMPLATE = "{statement}\n"
# MMLU-style letter scoring. "السؤال/الخيارات/الإجابة" = question/options/answer.
AR_MCQ_TEMPLATE = "السؤال: {question}\n{options_block}\nالإجابة:"
# ArabicMMLU rows may carry a reading passage in `Context`.
ARABICMMLU_CONTEXT_TEMPLATE = "{context}\n\nالسؤال: {question}\n{options_block}\nالإجابة:"
# Alyah / DziriEval: question -> answer text, scored as a continuation.
AR_QA_TEMPLATE = "{question}\nالإجابة:"
# Global-PIQA: the prompt is a sentence fragment or question; solutions continue it.
GLOBAL_PIQA_AR_TEMPLATE = "{prompt}\n"

# ArabicMMLU subjects that probe Arab-region knowledge (the CMMLU
# "China-specific subjects" analogue). Excludes Biology/Physics/Math/CS/etc.,
# which are culture-neutral and belong to the Dimension-2 forgetting check.
# These 13 subjects = 10,529 of the 14,455 test rows.
ARABICMMLU_ARAB_SUBJECTS = [
    "Islamic Studies", "Geography", "Driving Test", "General Knowledge",
    "History", "Social Science", "Arabic Language", "Arabic Language (General)",
    "Arabic Language (Grammar)", "Civics", "Law", "Political Science", "Philosophy",
]

# The 13 Arabic varieties in Global-PIQA (ISO 639-3 + script, matching the KB's
# variety labels). bcc/bsk/ckb/pes are Arabic-SCRIPT but not Arabic-language.
GLOBAL_PIQA_AR_VARIETIES = [
    "acm_arab", "acq_arab", "aeb_arab", "afb_arab",
    "apc_arab_jord", "apc_arab_leba", "apc_arab_pale", "apc_arab_syri",
    "arb_arab", "arq_arab", "ars_arab", "ary_arab", "arz_arab",
]
GLOBAL_PIQA_AR_PARALLEL_VARIETIES = ["arb_arab", "ary_arab", "arz_arab"]

# Figurative-language slices used to assemble `ar_figurative`.
ALYAH_FIGURATIVE_CATEGORIES = [
    "Imagery & Figurative Meaning",      # 121
    "Poetry & Creative Expression",      # 32
    "Greetings & Daily Expressions",     # 61
]
DZIRIEVAL_FIGURATIVE_DOMAINS = [
    "Métaphores et Idiomes",             # 50
    "Proverbes et Sagesses",             # 50
]

ARABCULTURE_COUNTRIES = [
    "Algeria", "Egypt", "Jordan", "KSA", "Lebanon", "Libya", "Morocco",
    "Palestine", "Sudan", "Syria", "Tunisia", "UAE", "Yemen",
]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _read_any(path: str) -> List[Dict[str, Any]]:
    """:func:`tasks._read_rows` plus ``.parquet`` (ArabCulture / Alyah ship parquet)."""
    if path.lower().endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    return _read_rows(path)


def _resolve(ar_data_dir: Optional[str], *parts: str) -> str:
    root = ar_data_dir or DEFAULT_AR_DATA_DIR
    return os.path.join(root, *parts)


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found at {path}. Run src/culture/evaluation/download_ar.sh "
            f"(or pass --ar_data_dir to the directory it wrote)."
        )
    return path


def _letter_block(question: str, opts: List[str], context: Optional[str] = None) -> str:
    lines = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
    if context:
        return ARABICMMLU_CONTEXT_TEMPLATE.format(
            context=context, question=question, options_block=lines)
    return AR_MCQ_TEMPLATE.format(question=question, options_block=lines)


def _clean(x: Any) -> str:
    """Cell -> stripped str; pandas NaN/None -> ''."""
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


# --------------------------------------------------------------------------- #
# Optional KB-overlap filter (belt-and-braces decontamination)
# --------------------------------------------------------------------------- #
def _kb_surfaces(kb_path: str) -> set:
    """Normalised idiom surfaces from the Arabic KB, for overlap filtering.

    Uses the SAME normaliser as the corpus filter
    (``culture.data_processing.ar_idioms.normalize.normalize_ar``) so "overlap"
    here means exactly what it means everywhere else in the pipeline.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from culture.data_processing.ar_idioms.normalize import normalize_ar
    surfaces = set()
    with open(kb_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            idiom = (json.loads(line).get("idiom") or "").strip()
            if idiom:
                surfaces.add(normalize_ar(idiom))
    return surfaces


def _drop_kb_overlap(examples: List[MCExample], kb_path: str,
                     fields: str = "context") -> List[MCExample]:
    """Drop examples whose stem (or any option) normalises to a KB idiom surface."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from culture.data_processing.ar_idioms.normalize import normalize_ar
    surfaces = _kb_surfaces(kb_path)
    kept, dropped = [], 0
    for ex in examples:
        blobs = [ex.context] if fields == "context" else [ex.context, *ex.options]
        # An idiom quoted inside a stem is short; test every whitespace n-gram
        # up to 12 tokens rather than the whole stem (which never matches).
        hit = False
        for blob in blobs:
            toks = normalize_ar(blob).split()
            for n in range(2, 13):
                if hit:
                    break
                for i in range(len(toks) - n + 1):
                    if " ".join(toks[i:i + n]) in surfaces:
                        hit = True
                        break
        if hit:
            dropped += 1
        else:
            kept.append(ex)
    logger.info("KB-overlap filter: dropped %d / %d examples", dropped, len(examples))
    return kept


# --------------------------------------------------------------------------- #
# 1. ArabCulture (MBZUAI) — 3-way cultural sentence completion, 13 countries
# --------------------------------------------------------------------------- #
def load_arabculture(ar_data_dir: Optional[str] = None,
                     countries: Optional[List[str]] = None,
                     num_fewshot: int = 0, limit: Optional[int] = None,
                     seed: int = 42, kb_path: Optional[str] = None) -> MCTask:
    """MBZUAI/ArabCulture: natively authored cultural-commonsense completion.

    One parquet per country under ``<ar_data_dir>/arabculture/<Country>.parquet``.
    Schema: ``first_statement`` (the scene), ``options`` = struct with a ``text``
    array (3 options) and ``english_keys``, ``answer_key`` = struct with
    ``english_answer_key`` (A/B/C), plus ``sub_topic``, ``country``, ``region``
    and a ``should_discard`` quality flag set by the annotators.

    Options are free text of unequal length -> ``score_mode="continuation"``
    (primary metric ``acc_norm``). Report per-country / per-region from
    ``meta``.
    """
    rng = random.Random(seed)
    countries = countries or list(ARABCULTURE_COUNTRIES)
    parsed = []
    n_discarded = 0
    for country in countries:
        path = _require(_resolve(ar_data_dir, "arabculture", f"{country}.parquet"),
                        f"ArabCulture/{country}")
        for i, row in enumerate(_read_any(path)):
            if _clean(row.get("should_discard")).lower() == "yes":
                n_discarded += 1
                continue
            opts_field = row.get("options") or {}
            texts = [str(t).strip() for t in list(opts_field.get("text", []))]
            keys = [str(k).strip() for k in list(opts_field.get("english_keys", []))]
            ans = (row.get("answer_key") or {}).get("english_answer_key")
            gold = _resolve_gold_index(ans, keys or texts)
            if gold is None or not texts:
                continue
            parsed.append((
                _clean(row.get("first_statement")), texts, gold,
                f"{country}/{_clean(row.get('sample_id')) or i}",
                {"country": country, "region": _clean(row.get("region")),
                 "sub_topic": _clean(row.get("sub_topic"))},
            ))
    if n_discarded:
        logger.info("ArabCulture: dropped %d rows flagged should_discard=Yes", n_discarded)

    prefix = _mc_fewshot_prefix(
        [(ARABCULTURE_TEMPLATE.format(statement=s), t, g) for s, t, g, _, _ in parsed],
        num_fewshot, rng,
    )
    examples = [
        MCExample(qid=qid, context=prefix + ARABCULTURE_TEMPLATE.format(statement=stmt),
                  options=texts, gold=gold, meta={**meta, "options": texts})
        for stmt, texts, gold, qid, meta in parsed
    ]
    if kb_path:
        examples = _drop_kb_overlap(examples, kb_path)
    if limit:
        examples = examples[:limit]
    return MCTask(name="arabculture", examples=examples, score_mode="continuation")


# --------------------------------------------------------------------------- #
# 2. ArabicCulturalQA (QCRI) — 4-way MCQ, MSA + 5 dialects + English
# --------------------------------------------------------------------------- #
def load_arabic_cultural_qa(ar_data_dir: Optional[str] = None,
                            dialects: Optional[List[str]] = None,
                            num_fewshot: int = 0, limit: Optional[int] = None,
                            seed: int = 42, kb_path: Optional[str] = None) -> MCTask:
    """QCRI/ArabicCulturalQA ``mcq/test.jsonl``: 2,000 questions x 6 variants.

    Schema: ``{id, dialect, question, A, B, C, D, answer}`` where ``answer`` is a
    letter. Dialects: ``msa`` (headline number), ``egyptian``, ``gulf``,
    ``levantine``, ``maghrebi``, ``english``. Default = ``msa`` only; pass
    ``--acqa_dialects all`` for the full cross-dialect consistency axis.

    Balanced options -> MMLU-style letter scoring (``score_mode="letter"``).
    """
    rng = random.Random(seed)
    path = _require(_resolve(ar_data_dir, "acqa_test.jsonl"), "ArabicCulturalQA mcq/test.jsonl")
    rows = _read_any(path)
    want = None if (dialects and [d.lower() for d in dialects] == ["all"]) else \
        {d.lower() for d in (dialects or ["msa"])}

    parsed = []
    for i, row in enumerate(rows):
        dialect = _clean(row.get("dialect")).lower()
        if want is not None and dialect not in want:
            continue
        opts = [_clean(row.get(c)) for c in LETTERS[:4]]
        if not all(opts):
            continue
        gold = _resolve_gold_index(row.get("answer"), LETTERS[:4])
        if gold is None:
            continue
        parsed.append((_clean(row.get("question")), opts, gold,
                       _clean(row.get("id")) or str(i), dialect))

    prefix = ""
    if num_fewshot > 0 and parsed:
        shots = rng.sample(parsed, min(num_fewshot, len(parsed)))
        prefix = "".join(f"{_letter_block(q, o)} {LETTERS[g]}\n\n" for q, o, g, _, _ in shots)

    examples = [
        MCExample(qid=qid, context=prefix + _letter_block(q, opts),
                  options=[" " + LETTERS[j] for j in range(len(opts))], gold=gold,
                  meta={"dialect": dialect, "option_text": opts})
        for q, opts, gold, qid, dialect in parsed
    ]
    if kb_path:
        examples = _drop_kb_overlap(examples, kb_path)
    if limit:
        examples = examples[:limit]
    return MCTask(name="arabic_cultural_qa", examples=examples, score_mode="letter")


# --------------------------------------------------------------------------- #
# 3. ArabicMMLU (MBZUAI) — native Arab-region exams, 2-5 options
# --------------------------------------------------------------------------- #
def load_arabicmmlu(ar_data_dir: Optional[str] = None,
                    subjects: Optional[List[str]] = None,
                    num_fewshot: int = 0, limit: Optional[int] = None,
                    seed: int = 42, kb_path: Optional[str] = None) -> MCTask:
    """MBZUAI/ArabicMMLU ``All/test.csv`` — the CMMLU-China-specific analogue.

    Schema: ``Question``, optional ``Context`` (reading passage, 707 rows),
    ``Option 1..5`` (2-5 present), ``Answer Key`` (A-E), plus ``Subject``,
    ``Country``, ``Level``, ``Group``.

    Unlike CMMLU the option count VARIES (4: 10,120 / 3: 2,121 / 2: 1,874 /
    5: 340), so the letter block is sized per row. Default subject set is the
    13 Arab-region subjects (10,529 rows); pass ``--arabicmmlu_subjects all``
    for the full 14,455 (which then doubles as a general-knowledge probe).

    Few-shot uses the repo's own ``is_few_shot==1`` rows when present; the
    downloaded ``All/test.csv`` has none, so exemplars are drawn from the task
    itself and EXCLUDED from scoring to avoid leakage.
    """
    rng = random.Random(seed)
    path = _require(_resolve(ar_data_dir, "arabicmmlu_test.csv"), "ArabicMMLU All/test.csv")
    rows = _read_any(path)
    if subjects and [s.lower() for s in subjects] == ["all"]:
        keep_subjects = None
    else:
        keep_subjects = set(subjects or ARABICMMLU_ARAB_SUBJECTS)

    parsed, shot_pool = [], []
    for i, row in enumerate(rows):
        subject = _clean(row.get("Subject"))
        if keep_subjects is not None and subject not in keep_subjects:
            continue
        opts = [_clean(row.get(f"Option {k}")) for k in range(1, 6)]
        opts = [o for o in opts if o]                       # 2-5 options, in order
        if len(opts) < 2:
            continue
        gold = _resolve_gold_index(row.get("Answer Key"), LETTERS[:len(opts)])
        if gold is None:                                    # e.g. key 'E' with 4 options
            continue
        item = (_clean(row.get("Question")), opts, gold,
                _clean(row.get("ID")) or str(i), subject,
                _clean(row.get("Country")), _clean(row.get("Context")))
        (shot_pool if _clean(row.get("is_few_shot")) == "1" else parsed).append(item)

    prefix, exclude = "", set()
    if num_fewshot > 0:
        pool = shot_pool or parsed
        shots = rng.sample(pool, min(num_fewshot, len(pool)))
        if not shot_pool:                                   # drawn from the eval set -> exclude
            exclude = {s[3] for s in shots}
        prefix = "".join(f"{_letter_block(q, o, ctx or None)} {LETTERS[g]}\n\n"
                         for q, o, g, _, _, _, ctx in shots)

    examples = [
        MCExample(qid=f"{subject}/{qid}", context=prefix + _letter_block(q, opts, ctx or None),
                  options=[" " + LETTERS[j] for j in range(len(opts))], gold=gold,
                  meta={"subject": subject, "country": country,
                        "n_options": len(opts), "option_text": opts})
        for q, opts, gold, qid, subject, country, ctx in parsed if qid not in exclude
    ]
    if kb_path:
        examples = _drop_kb_overlap(examples, kb_path)
    if limit:
        examples = examples[:limit]
    return MCTask(name="arabicmmlu", examples=examples, score_mode="letter")


# --------------------------------------------------------------------------- #
# 4. Global-PIQA, Arabic varieties (non-parallel = culturally grounded)
# --------------------------------------------------------------------------- #
def _load_global_piqa_ar(ar_data_dir: Optional[str], subdir: str, varieties: List[str],
                         n_solutions: int, name: str, cultural_only: bool,
                         num_fewshot: int, limit: Optional[int], seed: int,
                         kb_path: Optional[str]) -> MCTask:
    rng = random.Random(seed)
    parsed = []
    for var in varieties:
        fname = f"{'nonparallel' if subdir == 'global_piqa' else 'parallel'}_{var}.tsv"
        path = _require(_resolve(ar_data_dir, subdir, fname), f"Global-PIQA {var}")
        for i, row in enumerate(_read_any(path)):
            if cultural_only and _clean(row.get("approx_cultural_score")) not in ("1", "1.0"):
                continue
            sols = [_clean(row.get(f"solution{k}")) for k in range(n_solutions)]
            sols = [s for s in sols if s]
            if len(sols) < 2:
                continue
            gold = _resolve_gold_index(row.get("label"), sols)
            if gold is None:
                continue
            parsed.append((_clean(row.get("prompt")), sols, gold,
                           _clean(row.get("example_id")) or f"{var}/{i}", var))

    prefix = _mc_fewshot_prefix(
        [(GLOBAL_PIQA_AR_TEMPLATE.format(prompt=p), s, g) for p, s, g, _, _ in parsed],
        num_fewshot, rng,
    )
    examples = [
        MCExample(qid=qid, context=prefix + GLOBAL_PIQA_AR_TEMPLATE.format(prompt=p),
                  options=sols, gold=gold, meta={"variety": var, "options": sols})
        for p, sols, gold, qid, var in parsed
    ]
    if kb_path:
        examples = _drop_kb_overlap(examples, kb_path)
    if limit:
        examples = examples[:limit]
    return MCTask(name=name, examples=examples, score_mode="continuation")


def load_global_piqa_ar(ar_data_dir: Optional[str] = None,
                        varieties: Optional[List[str]] = None,
                        cultural_only: bool = True, num_fewshot: int = 0,
                        limit: Optional[int] = None, seed: int = 42,
                        kb_path: Optional[str] = None) -> MCTask:
    """Global-PIQA non-parallel, 13 Arabic varieties x 100 items.

    Natively authored per variety; the two solutions are minimal pairs, so
    length bias is negligible — still scored ``continuation`` (``acc_norm``) for
    consistency with the Hindi Global-PIQA task. ``cultural_only=True`` keeps the
    1,099 items with ``approx_cultural_score == 1`` (drops physics-only items).
    Report per-variety from ``meta['variety']`` for dialect coverage.
    """
    return _load_global_piqa_ar(ar_data_dir, "global_piqa",
                                varieties or GLOBAL_PIQA_AR_VARIETIES, 2,
                                "global_piqa_ar", cultural_only, num_fewshot,
                                limit, seed, kb_path)


def load_global_piqa_ar_parallel(ar_data_dir: Optional[str] = None,
                                 varieties: Optional[List[str]] = None,
                                 num_fewshot: int = 0, limit: Optional[int] = None,
                                 seed: int = 42, kb_path: Optional[str] = None) -> MCTask:
    """Global-PIQA parallel arb/ary/arz, 103 items x 3, 4 options each.

    CONTROL, not a culture score: the content is culture-agnostic physical
    commonsense (``categories: object_properties_interactions``). Its job is to
    show that Arabic CPT did not damage basic Arabic reasoning — a drop here
    alongside a rise in ArabCulture is the regression signal to watch.
    """
    return _load_global_piqa_ar(ar_data_dir, "global_piqa_parallel",
                                varieties or GLOBAL_PIQA_AR_PARALLEL_VARIETIES, 4,
                                "global_piqa_ar_parallel", False, num_fewshot,
                                limit, seed, kb_path)


# --------------------------------------------------------------------------- #
# 5. Alyah (TII) — Emirati dialect + norms + figurative language
# --------------------------------------------------------------------------- #
def _alyah_parsed(ar_data_dir: Optional[str], categories: Optional[List[str]]):
    path = _require(_resolve(ar_data_dir, "alyah.parquet"), "Alyah test parquet")
    keep = set(categories) if categories else None
    out = []
    for i, row in enumerate(_read_any(path)):
        category = _clean(row.get("category"))
        if keep is not None and category not in keep:
            continue
        opts = [_clean(row.get(f"option_{k}")) for k in range(1, 5)]
        opts = [o for o in opts if o]
        # `correct_answer` is 1-indexed into option_1..option_4.
        try:
            gold = int(row["correct_answer"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= gold < len(opts)):
            continue
        out.append((_clean(row.get("query")), opts, gold, f"alyah/{i}", category))
    return out


def load_alyah(ar_data_dir: Optional[str] = None, categories: Optional[List[str]] = None,
               num_fewshot: int = 0, limit: Optional[int] = None, seed: int = 42,
               kb_path: Optional[str] = None) -> MCTask:
    """tiiuae/alyah-emirati-benchmark — 1,173 items, 4 options, no duplicate queries.

    Schema: ``query``, ``option_1..option_4``, ``correct_answer`` (1-indexed),
    ``category``. Manually collected from native Emirati speakers.

    ``score_mode="continuation"`` is MANDATORY here: the correct option is the
    longest one 57.5 % of the time (chance 25 %), mean gold length 22.8 chars vs
    18.7 overall, so raw summed log-prob rewards the length prior rather than the
    model. Read ``acc_norm``, not ``acc``.
    """
    rng = random.Random(seed)
    parsed = _alyah_parsed(ar_data_dir, categories)
    prefix = _mc_fewshot_prefix(
        [(AR_QA_TEMPLATE.format(question=q), [" " + o for o in opts], g)
         for q, opts, g, _, _ in parsed], num_fewshot, rng)
    examples = [
        MCExample(qid=qid, context=prefix + AR_QA_TEMPLATE.format(question=q),
                  options=[" " + o for o in opts], gold=gold,
                  meta={"category": category, "source": "alyah", "options": opts})
        for q, opts, gold, qid, category in parsed
    ]
    if kb_path:
        examples = _drop_kb_overlap(examples, kb_path)
    if limit:
        examples = examples[:limit]
    return MCTask(name="alyah", examples=examples, score_mode="continuation")


# --------------------------------------------------------------------------- #
# 6. DziriEval — Algerian Darija
# --------------------------------------------------------------------------- #
def _dzirieval_parsed(ar_data_dir: Optional[str], domains: Optional[List[str]]):
    """Parse DziriEval, deduplicating on the QUESTION TEXT (not ``id``).

    The file has 1,000 rows and 950 distinct questions: 50 items appear twice,
    tagged with a second ``domain``. Deduplicating on ``id`` looks natural and is
    WRONG — ``id`` is not a unique key. There are only 850 distinct ids, because
    50 ids are reused for *genuinely different* questions (49 of those pairs also
    disagree on the gold letter), so an id-keyed dedup silently deletes 100 real
    items. Keying on the question text yields the correct 950.

    Dedup runs AFTER the domain filter, since for a doubly-tagged item the second
    row is what makes it visible under its second domain.
    """
    path = _require(_resolve(ar_data_dir, "dzirieval.jsonl"), "DziriEval jsonl")
    keep = set(domains) if domains else None
    out, seen = [], set()
    for i, row in enumerate(_read_any(path)):
        domain = _clean(row.get("domain"))
        if keep is not None and domain not in keep:
            continue
        question = _clean(row.get("question"))
        if question in seen:
            continue
        seen.add(question)
        qid = f"{_clean(row.get('id')) or 'row'}_{i}"
        choices = row.get("choices") or {}
        if isinstance(choices, str):
            choices = json.loads(choices)
        letters = [c for c in LETTERS[:4] if _clean(choices.get(c))]
        opts = [_clean(choices[c]) for c in letters]
        gold = _resolve_gold_index(row.get("answer"), letters)
        if gold is None or len(opts) < 2:
            continue
        out.append((_clean(row.get("question")), opts, gold, f"dzirieval/{qid}", domain))
    return out


def load_dzirieval(ar_data_dir: Optional[str] = None, domains: Optional[List[str]] = None,
                   num_fewshot: int = 0, limit: Optional[int] = None, seed: int = 42,
                   kb_path: Optional[str] = None) -> MCTask:
    """touati-kamel/DziriEval — 950 unique Algerian-Darija items, 4 options.

    Schema: ``{id, category, domain, question, choices: {A..D}, answer, explanation}``.
    20 domains x 50. Reported as SECONDARY evidence: no license tag, no paper,
    single author — good coverage of the Maghreb, weak provenance.
    """
    rng = random.Random(seed)
    parsed = _dzirieval_parsed(ar_data_dir, domains)
    prefix = _mc_fewshot_prefix(
        [(AR_QA_TEMPLATE.format(question=q), [" " + o for o in opts], g)
         for q, opts, g, _, _ in parsed], num_fewshot, rng)
    examples = [
        MCExample(qid=qid, context=prefix + AR_QA_TEMPLATE.format(question=q),
                  options=[" " + o for o in opts], gold=gold,
                  meta={"domain": domain, "source": "dzirieval", "options": opts})
        for q, opts, gold, qid, domain in parsed
    ]
    if kb_path:
        examples = _drop_kb_overlap(examples, kb_path)
    if limit:
        examples = examples[:limit]
    return MCTask(name="dzirieval", examples=examples, score_mode="continuation")


# --------------------------------------------------------------------------- #
# 7. ar_figurative — Dimension 3, the clean figurative slice
# --------------------------------------------------------------------------- #
def load_ar_figurative(ar_data_dir: Optional[str] = None, num_fewshot: int = 0,
                       limit: Optional[int] = None, seed: int = 42,
                       kb_path: Optional[str] = None) -> MCTask:
    """The contamination-free Arabic figurative-understanding set (314 items).

    Alyah  'Imagery & Figurative Meaning' (121) + 'Poetry & Creative Expression'
    (32) + 'Greetings & Daily Expressions' (61) = 214, plus DziriEval
    'Métaphores et Idiomes' (50) + 'Proverbes et Sagesses' (50) = 100.

    Why it is this small, stated plainly: **there is no Arabic ChID**, and the
    only Arabic idiom cloze that exists (``menaattia/Kinayat``) is 96.6 %
    contained in our own training KB, as is Jawaher (99.5 % of its test split).
    314 items give a +/-5.5 pp standard error at 50 % accuracy — enough for a
    large CPT effect, not a subtle one. Two ways to strengthen it are recorded
    in plans/arabic_pipeline_plan.md §5: score Kinayat separately as a labelled
    *memorization ceiling*, and mine a genuine cloze set from the 7,973
    KB-disjoint MIDAS idioms against the FineWeb-2 held-out split.

    Meaning-selection, unequal option lengths -> ``acc_norm``.
    """
    rng = random.Random(seed)
    parsed = ([(*p, "alyah") for p in _alyah_parsed(ar_data_dir, ALYAH_FIGURATIVE_CATEGORIES)]
              + [(*p, "dzirieval") for p in _dzirieval_parsed(ar_data_dir,
                                                              DZIRIEVAL_FIGURATIVE_DOMAINS)])
    prefix = _mc_fewshot_prefix(
        [(AR_QA_TEMPLATE.format(question=q), [" " + o for o in opts], g)
         for q, opts, g, _, _, _ in parsed], num_fewshot, rng)
    examples = [
        MCExample(qid=qid, context=prefix + AR_QA_TEMPLATE.format(question=q),
                  options=[" " + o for o in opts], gold=gold,
                  meta={"slice": slice_, "source": source, "options": opts})
        for q, opts, gold, qid, slice_, source in parsed
    ]
    if kb_path:
        examples = _drop_kb_overlap(examples, kb_path)
    if limit:
        examples = examples[:limit]
    return MCTask(name="ar_figurative", examples=examples, score_mode="continuation")


LOADERS_AR = {
    "ar_figurative": load_ar_figurative,
    "arabculture": load_arabculture,
    "arabic_cultural_qa": load_arabic_cultural_qa,
    "arabicmmlu": load_arabicmmlu,
    "global_piqa_ar": load_global_piqa_ar,
    "global_piqa_ar_parallel": load_global_piqa_ar_parallel,
    "alyah": load_alyah,
    "dzirieval": load_dzirieval,
}
