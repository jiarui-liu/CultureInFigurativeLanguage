"""Dataset loaders + prompt templating for the four Chinese benchmarks.

Companion to :mod:`culture.evaluation.tasks` (the Hindi loaders): the shared
dataclasses (:class:`MCExample`, :class:`MCTask`) and helpers (``_read_rows``,
``_first``, ``_resolve_gold_index``, ``_mc_fewshot_prefix``, ``_load_hf``) are
imported from that module rather than re-implemented here.

All four tasks are base-model MULTIPLE-CHOICE / CLOZE, scored by log-likelihood
via :meth:`culture.evaluation.scorer.HFModel.loglikelihood` and evaluated by the
existing :func:`culture.evaluation.run_eval.eval_mc` (no new model/scoring code):

- ``chid``          : ChID chengyu cloze (fill-the-blank; N-way, scored ``acc``).
- ``chengyu_bench`` : Chengyu-Bench connotation / appropriateness (binary).
- ``cmmlu``         : CMMLU China-specific subjects (4-choice, MMLU-style letter).
- ``ccpm``          : Chinese Classical Poetry Matching (4-choice, ``acc_norm``).

Data access is *local-file-first* where a local schema is documented; ChID and
CMMLU also fall back to HuggingFace (lazy ``datasets`` import) when no local path
is given. Chengyu-Bench and CCPM are GitHub-only (``git clone``): pass the cloned
directory / file. Download commands are in ``download_zh.sh`` and
``docs/plans/eval_benchmarks_download.md``.
"""

import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional

# Reuse the shared dataclasses + helpers from the Hindi task module (do not fork).
from culture.evaluation.tasks import (
    LETTERS,
    MCExample,
    MCTask,
    _first,
    _load_hf,
    _mc_fewshot_prefix,
    _read_rows,
    _resolve_gold_index,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Templates (edit here to tune prompting; kept explicit for transparency)
# --------------------------------------------------------------------------- #
# ChID cloze: no template — the passage itself is the context. Other blanks in
# the same passage are replaced by this neutral placeholder when scoring one blank.
CHID_PLACEHOLDER = "____"
CHID_BLANK_RE = re.compile(r"#idiom\d*#")           # matches "#idiom#" and "#idiom000000#"

# Chengyu-Bench — connotation (感情色彩): 成语「X」的感情色彩是： 褒义/贬义
CHENGYU_CONNOTATION_TEMPLATE = "成语「{idiom}」的感情色彩是："
CHENGYU_CONNOTATION_OPTIONS = [" 褒义", " 贬义"]      # [0]=positive (褒义), [1]=negative (贬义)
# Chengyu-Bench — appropriateness: passage (idiom marked) → 恰当/不恰当
CHENGYU_APPROPRIATENESS_TEMPLATE = "{passage}\n上文中成语的使用是否恰当？答："
CHENGYU_APPROPRIATENESS_OPTIONS = [" 恰当", " 不恰当"]  # [0]=appropriate, [1]=inappropriate

# CMMLU — MMLU-style, score the answer letter (A/B/C/D). "题目/答案" = question/answer.
CMMLU_TEMPLATE = "题目：{question}\n{options_block}\n答案："

# CCPM — Chinese Classical Poetry Matching: modern paraphrase → classical line.
CCPM_TEMPLATE = "现代文：{translation}\n对应的诗句是："

# The 16 China-specific CMMLU subject configs (culturally-loaded ones).
CMMLU_DEFAULT_SUBJECTS = [
    "ancient_chinese", "chinese_history", "chinese_literature",
    "chinese_civil_service_exam", "chinese_driving_rule", "chinese_food_culture",
    "chinese_foreign_policy", "chinese_teacher_qualification",
    "construction_project_management", "elementary_chinese",
    "elementary_commonsense", "ethnology", "high_school_politics",
    "modern_chinese", "traditional_chinese_medicine", "marxist_theory",
]


# --------------------------------------------------------------------------- #
# HF helper (CMMLU needs trust_remote_code + a parquet-mirror fallback)
# --------------------------------------------------------------------------- #
def _load_hf_trust(repo: str, split: str, config: Optional[str] = None,
                   fallback_repo: Optional[str] = None) -> List[Dict[str, Any]]:
    """Like :func:`tasks._load_hf` but passes ``trust_remote_code=True`` and, if
    the (script-based) ``repo`` fails to load, retries a parquet mirror.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("`datasets` is required for HF loading; `pip install datasets`.") from e
    token = os.environ.get("HF_TOKEN")
    logger.info("Loading HF dataset %s (config=%s, split=%s, trust_remote_code=True)",
                repo, config, split)
    try:
        ds = load_dataset(repo, config, split=split, token=token, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 — script repos break on newer `datasets`
        if not fallback_repo:
            raise
        logger.warning("Loading %s (config=%s) failed (%s); falling back to mirror %s",
                       repo, config, e, fallback_repo)
        ds = load_dataset(fallback_repo, config, split=split, token=token)
    return [dict(r) for r in ds]


# --------------------------------------------------------------------------- #
# 1. ChID (chengyu cloze / fill-in-the-blank)
# --------------------------------------------------------------------------- #
def _chid_unwrap(row: Dict[str, Any]) -> Dict[str, Any]:
    """thu-coai/chid stores each example as a JSON string in a ``text`` column;
    unwrap it to the underlying {content, candidates, groundTruth} dict."""
    if isinstance(row.get("text"), str):
        try:
            obj = json.loads(row["text"])
        except (json.JSONDecodeError, TypeError):
            return row
        if isinstance(obj, dict) and ("content" in obj or "candidates" in obj):
            return obj
    return row


def _chid_golds(row: Dict[str, Any]) -> List[Any]:
    """Return the per-blank gold answers (idiom strings or indices) in order.

    Defensive across ChID variants: original ChID uses ``groundTruth`` (list of
    idiom strings); CLUE-chid uses ``answers`` (a dict with ``candidate_id`` /
    ``text`` lists) or a flat ``answer``/``label``.
    """
    gt = row.get("groundTruth", row.get("ground_truth"))
    if gt is None:
        gt = _first(row, ["answers", "answer", "labels", "label"])
    if gt is None:
        return []
    if isinstance(gt, dict):                       # CLUE: {"candidate_id": [...], "text": [...]}
        ids = gt.get("candidate_id", gt.get("candidate_ids"))
        if ids is not None:
            return list(ids)
        return list(gt.get("text", []))
    if isinstance(gt, list):
        return gt
    return [gt]


def _chid_resolve_gold(raw: Any, candidates: List[str]) -> Optional[int]:
    """Map one blank's gold (idiom text / index) to an index into ``candidates``."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raw = int(raw)
    if isinstance(raw, int):
        return raw if 0 <= raw < len(candidates) else None
    s = str(raw).strip()
    if s.isdigit():
        i = int(s)
        return i if 0 <= i < len(candidates) else None
    for i, c in enumerate(candidates):
        if c.strip() == s:
            return i
    return None


def _chid_split(passage: str, matches: List[re.Match], k: int) -> tuple:
    """Split ``passage`` at the k-th blank; return (LEFT, RIGHT) with every OTHER
    blank replaced by :data:`CHID_PLACEHOLDER`."""
    target = matches[k]
    left = CHID_BLANK_RE.sub(CHID_PLACEHOLDER, passage[:target.start()])
    right = CHID_BLANK_RE.sub(CHID_PLACEHOLDER, passage[target.end():])
    return left, right


def _load_chid_answers(path: str):
    """Load a ChID answer file → dict {blank_tag: gold} or ordered list [gold, ...].

    The original ChID release (GitHub ``chujiezheng/ChID-Dataset``) ships gold
    labels SEPARATELY from the passages: a ``*_answer.json`` (dict keyed by the
    numbered blank tag, e.g. ``{"#idiom000000#": 3}``) or a ``*_answer.csv``
    (rows ``tag,index`` — or a single ``index`` column in blank order).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        d, flat = {}, []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    d[parts[0]] = parts[1]
                elif parts:
                    flat.append(parts[0])
        return d or flat
    with open(path, encoding="utf-8") as f:
        if ext == ".jsonl":
            items = [json.loads(l) for l in f if l.strip()]
            if items and all(isinstance(x, dict) for x in items):
                merged = {}
                for x in items:
                    merged.update(x)
                return merged
            return items
        return json.load(f)                        # dict {tag: gold} or list [gold, ...]


def load_chid(data_path: Optional[str] = None, answer_path: Optional[str] = None,
              num_fewshot: int = 0, limit: Optional[int] = None, seed: int = 42,
              hf_repo: str = "thu-coai/chid", hf_split: str = "validation") -> MCTask:
    """ChID chengyu cloze. One :class:`MCExample` per blank.

    For each blank we split the passage into LEFT / RIGHT; the context is LEFT and
    each option's continuation is ``candidate + RIGHT`` (no leading space — Chinese
    text is unspaced, so the idiom must sit flush against LEFT). Candidates are
    4-char chengyu sharing the identical RIGHT, so the RIGHT log-prob cancels in
    the arg-max: **``acc`` (raw summed log-prob) is the primary metric**, hence
    ``score_mode="continuation"`` (which also reports ``acc_norm``).

    Gold source (in priority order):
    - ``answer_path``: a separate ChID answer file (dict keyed by blank tag, or an
      ordered list) — REQUIRED for a scorable eval, because the HF mirror
      ``thu-coai/chid`` ships **no** gold in any split. Get it from the original
      ``chujiezheng/ChID-Dataset`` (``*_answer.json`` / ``*_answer.csv``).
    - else in-row ``groundTruth`` (some local dumps carry it).

    ``candidates`` may be a single shared list (the ChID competition format; one
    index per blank into that list) or a list-of-lists (per-blank candidates).
    """
    rng = random.Random(seed)
    rows = _read_rows(data_path) if data_path else _load_hf(hf_repo, split=hf_split)
    answer_map = _load_chid_answers(answer_path) if answer_path else None
    ans_is_dict = isinstance(answer_map, dict)

    parsed: List[tuple] = []                        # (left, options, gold, qid)
    n_missing = 0
    gblank = 0                                      # global blank index (list-mode answers)
    for i, row in enumerate(rows):
        row = _chid_unwrap(row)
        content = _first(row, ["content", "passage", "sentence"])
        cands = _first(row, ["candidates", "options", "choices"])
        if isinstance(cands, str):
            cands = json.loads(cands)
        if not content or not cands:
            continue
        per_blank = isinstance(cands[0], list)      # list-of-lists → per-blank candidates
        if not per_blank:
            cands = [str(c) for c in cands]
        in_row_golds = _chid_golds(row)             # fallback when no answer_map
        rid = str(_first(row, ["id", "qid", "idx"], i))
        passages = content if isinstance(content, list) else [content]
        blank_k = 0                                 # blank index within this row
        for p_idx, passage in enumerate(passages):
            matches = list(CHID_BLANK_RE.finditer(passage))
            for m_idx in range(len(matches)):
                tag = matches[m_idx].group(0)
                blank_cands = [str(c) for c in cands[blank_k]] if per_blank else cands
                if answer_map is not None:
                    gold_raw = (answer_map.get(tag) if ans_is_dict
                                else (answer_map[gblank] if gblank < len(answer_map) else None))
                else:
                    gold_raw = in_row_golds[blank_k] if blank_k < len(in_row_golds) else None
                gold = _chid_resolve_gold(gold_raw, blank_cands)
                if gold is None:
                    n_missing += 1
                    gold = 0
                left, right = _chid_split(passage, matches, m_idx)
                options = [c + right for c in blank_cands]
                parsed.append((left, options, gold, f"{rid}-{p_idx}-{m_idx}"))
                blank_k += 1
                gblank += 1

    if parsed and n_missing == len(parsed):
        raise ValueError(
            "ChID: no gold answers resolved (all defaulted). thu-coai/chid ships no "
            "groundTruth; download the original chujiezheng/ChID-Dataset and pass "
            "--chid_path <dev.json/jsonl> and --chid_answer_path <dev_answer.json/csv>.")
    if n_missing:
        logger.warning("ChID: %d/%d blank(s) had no resolvable gold (defaulted to 0).",
                       n_missing, len(parsed))

    # Few-shot demonstrations are filled-in passages (LEFT + correct idiom + RIGHT).
    prefix = ""
    if num_fewshot > 0 and parsed:
        shots = rng.sample(parsed, min(num_fewshot, len(parsed)))
        prefix = "".join(left + opts[g] + "\n\n" for left, opts, g, _ in shots)

    examples = [MCExample(qid=qid, context=prefix + left, options=options, gold=gold, meta={})
                for left, options, gold, qid in parsed]
    if limit:
        examples = examples[:limit]
    return MCTask(name="chid", examples=examples, score_mode="continuation")


# --------------------------------------------------------------------------- #
# 2. Chengyu-Bench (connotation + appropriateness binary subtasks)
# --------------------------------------------------------------------------- #
# NOTE: Chengyu-Bench (GitHub `sofyc/ChengyuBench`, arXiv 2506.18105) is a
# `git clone`-only release; GitHub was unreachable while writing this loader, so
# the exact JSON field/file names below are a DEFENSIVE best-guess based on the
# paper's task descriptions. **Confirm the real schema after `git clone`** and
# extend the field aliases / label maps here if they differ.
_CONNOTATION_POS = {"褒义", "褒", "positive", "pos", "commendatory", "good"}
_CONNOTATION_NEG = {"贬义", "贬", "negative", "neg", "derogatory", "bad"}
_APPROPRIATE = {"恰当", "appropriate", "correct", "right", "yes", "true"}
_INAPPROPRIATE = {"不恰当", "inappropriate", "incorrect", "wrong", "no", "false"}

_CHENGYU_BENCH_ALIASES = {
    "connotation": ["connotation", "sentiment", "感情色彩", "polarity"],
    "appropriateness": ["appropriateness", "appropriate", "恰当", "usage", "suitability"],
}


def _connotation_gold(raw: Any) -> int:
    """0 = 褒义 (positive), 1 = 贬义 (negative). Numeric fallback assumes 0/1
    (CONFIRM after clone)."""
    if raw is None:
        raise ValueError("Chengyu-Bench connotation item has no label")
    if isinstance(raw, bool):
        return 0 if raw else 1
    s = str(raw).strip().lower()
    if s in _CONNOTATION_POS:
        return 0
    if s in _CONNOTATION_NEG:
        return 1
    if s.isdigit():
        return 0 if int(s) == 0 else 1
    raise ValueError(f"Cannot map connotation label {raw!r}")


def _appropriateness_gold(raw: Any) -> int:
    """0 = 恰当 (appropriate), 1 = 不恰当 (inappropriate). Numeric fallback assumes
    1=appropriate, 0=inappropriate (CONFIRM after clone)."""
    if raw is None:
        raise ValueError("Chengyu-Bench appropriateness item has no label")
    if isinstance(raw, bool):
        return 0 if raw else 1
    s = str(raw).strip().lower()
    if s in _APPROPRIATE:
        return 0
    if s in _INAPPROPRIATE:
        return 1
    if s.isdigit():
        return 0 if int(s) == 1 else 1
    raise ValueError(f"Cannot map appropriateness label {raw!r}")


def _find_chengyu_bench_file(root: str, subtask: str) -> str:
    """Find the json/jsonl file for ``subtask`` under the cloned repo ``root``."""
    aliases = _CHENGYU_BENCH_ALIASES[subtask]
    found: List[str] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            low = fn.lower()
            if low.endswith((".json", ".jsonl")) and any(a.lower() in low for a in aliases):
                found.append(os.path.join(dirpath, fn))
    if not found:
        raise FileNotFoundError(
            f"No {subtask} json/jsonl under {root!r} (looked for names containing "
            f"{aliases}). Confirm the Chengyu-Bench layout after `git clone "
            f"https://github.com/sofyc/ChengyuBench` and adjust --chengyu_bench_dir "
            f"or the aliases in tasks_zh._CHENGYU_BENCH_ALIASES.")
    found.sort()
    return found[0]


def load_chengyu_bench(chengyu_bench_dir: Optional[str] = None,
                       subtask: str = "connotation", num_fewshot: int = 0,
                       limit: Optional[int] = None, seed: int = 42,
                       data_path: Optional[str] = None) -> MCTask:
    """Chengyu-Bench binary subtask (``connotation`` or ``appropriateness``).

    Pass ``--chengyu_bench_dir`` (the cloned ``sofyc/ChengyuBench`` repo) and the
    loader picks the matching json/jsonl; or pass ``data_path`` to a specific file.
    Both subtasks are 2-way and scored by log-likelihood (``score_mode="continuation"``).
    """
    if subtask not in ("connotation", "appropriateness"):
        raise ValueError(f"Unknown Chengyu-Bench subtask {subtask!r} "
                         "(choose connotation | appropriateness)")
    if data_path:
        path = data_path
    elif chengyu_bench_dir:
        path = _find_chengyu_bench_file(chengyu_bench_dir, subtask)
    else:
        raise ValueError("Chengyu-Bench requires --chengyu_bench_dir (git clone "
                         "https://github.com/sofyc/ChengyuBench) or a --data_path.")
    rows = _read_rows(path)
    rng = random.Random(seed)

    if subtask == "connotation":
        options = CHENGYU_CONNOTATION_OPTIONS

        def parse(row: Dict[str, Any], i: int) -> tuple:
            idiom = _first(row, ["idiom", "chengyu", "word", "成语", "query"])
            gold = _connotation_gold(_first(row, ["label", "connotation", "sentiment",
                                                  "polarity", "感情色彩", "answer"]))
            ctx = CHENGYU_CONNOTATION_TEMPLATE.format(idiom=idiom)
            return ctx, gold, str(_first(row, ["id", "qid"], i)), {"idiom": idiom}
    else:
        options = CHENGYU_APPROPRIATENESS_OPTIONS

        def parse(row: Dict[str, Any], i: int) -> tuple:
            passage = _first(row, ["passage", "content", "sentence", "text", "context"])
            gold = _appropriateness_gold(_first(row, ["label", "appropriate", "correct",
                                                      "恰当", "answer"]))
            ctx = CHENGYU_APPROPRIATENESS_TEMPLATE.format(passage=passage)
            return ctx, gold, str(_first(row, ["id", "qid"], i)), {"passage": passage}

    parsed = [parse(r, i) for i, r in enumerate(rows)]
    prefix = _mc_fewshot_prefix([(ctx, options, g) for ctx, g, _, _ in parsed],
                                num_fewshot, rng)
    examples = [MCExample(qid=qid, context=prefix + ctx, options=list(options),
                          gold=gold, meta={"subtask": subtask, **meta})
                for ctx, gold, qid, meta in parsed]
    if limit:
        examples = examples[:limit]
    return MCTask(name="chengyu_bench", examples=examples, score_mode="continuation")


# --------------------------------------------------------------------------- #
# 3. CMMLU (China-specific cultural subjects, MMLU-style 4-way letter scoring)
# --------------------------------------------------------------------------- #
def load_cmmlu(subjects: Optional[List[str]] = None, num_fewshot: int = 5,
               limit: Optional[int] = None, seed: int = 42,
               cmmlu_dir: Optional[str] = None,
               hf_repo: str = "haonan-li/cmmlu",
               fallback_repo: str = "lmlmcat/cmmlu") -> MCTask:
    """CMMLU over the China-specific subject configs, concatenated.

    Mirrors :func:`tasks.load_milu`: score the answer letter (A/B/C/D),
    ``score_mode="letter"``. Few-shot exemplars come from each subject's own
    ``dev`` split (no test leakage). Columns: ``Question, A, B, C, D, Answer``.

    Data source (in priority order):
    - ``cmmlu_dir``: **local CSV mode** — reads ``<cmmlu_dir>/test/<subject>.csv``
      and ``<cmmlu_dir>/dev/<subject>.csv`` directly (the layout produced by
      ``huggingface-cli download haonan-li/cmmlu``). Use this when the HF *script*
      loader is unavailable (it is removed in ``datasets>=4.0``).
    - else HuggingFace via ``trust_remote_code=True``, falling back to the parquet
      mirror ``lmlmcat/cmmlu`` if the script repo fails.
    """
    rng = random.Random(seed)
    subjects = subjects or list(CMMLU_DEFAULT_SUBJECTS)
    # Sentinel: --cmmlu_subjects all -> every subject present (all 67 in full CMMLU).
    if [s.lower() for s in subjects] == ["all"]:
        import glob
        if not cmmlu_dir:
            raise ValueError("--cmmlu_subjects all requires --cmmlu_dir (local CSVs).")
        subjects = sorted(os.path.splitext(os.path.basename(p))[0]
                          for p in glob.glob(os.path.join(cmmlu_dir, "test", "*.csv")))

    def parse(row: Dict[str, Any], i: int, subj: str) -> tuple:
        question = _first(row, ["Question", "question"])
        opts = [_first(row, [c, c.lower()]) for c in LETTERS[:4]]
        opts = [str(o) for o in opts if o is not None]
        gold = _resolve_gold_index(_first(row, ["Answer", "answer", "label", "target"]), opts)
        return question, opts, gold, str(_first(row, ["id", "qid", "Question_id"], i)), subj

    def block(q: str, opts: List[str]) -> str:
        lines = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        return CMMLU_TEMPLATE.format(question=q, options_block=lines)

    examples: List[MCExample] = []
    for subj in subjects:
        if cmmlu_dir:
            test_rows = _read_rows(os.path.join(cmmlu_dir, "test", f"{subj}.csv"))
            dev_rows = _read_rows(os.path.join(cmmlu_dir, "dev", f"{subj}.csv"))
        else:
            test_rows = _load_hf_trust(hf_repo, "test", subj, fallback_repo)
            dev_rows = _load_hf_trust(hf_repo, "dev", subj, fallback_repo)
        test_parsed = [parse(r, i, subj) for i, r in enumerate(test_rows)]
        dev_parsed = [parse(r, i, subj) for i, r in enumerate(dev_rows)]

        prefix = ""
        if num_fewshot > 0 and dev_parsed:
            shots = rng.sample(dev_parsed, min(num_fewshot, len(dev_parsed)))
            prefix = "".join(f"{block(q, o)} {LETTERS[g]}\n\n" for q, o, g, _, _ in shots)

        for q, opts, gold, qid, _ in test_parsed:
            ctx = prefix + block(q, opts)
            examples.append(MCExample(
                qid=f"{subj}/{qid}", context=ctx,
                options=[" " + LETTERS[j] for j in range(len(opts))],
                gold=gold, meta={"subject": subj, "option_text": opts},
            ))
    if limit:
        examples = examples[:limit]
    return MCTask(name="cmmlu", examples=examples, score_mode="letter")


# --------------------------------------------------------------------------- #
# 4. CCPM (Chinese Classical Poetry Matching, 4-way MC)
# --------------------------------------------------------------------------- #
def load_ccpm(data_path: Optional[str] = None, num_fewshot: int = 0,
              limit: Optional[int] = None, seed: int = 42) -> MCTask:
    """CCPM (Chinese Classical Poetry Matching). JSONL schema per line:
    ``{"translation": <modern paraphrase>, "choices": [4 lines], "answer": <index>}``.

    Options differ in length, so ``acc_norm`` is the fair metric
    (``score_mode="continuation"``). GitHub-only: pass ``--ccpm_path`` to the
    cloned ``THUNLP-AIPoet/CCPM`` JSONL.
    """
    if not data_path:
        raise ValueError("CCPM has no clean HF release; pass --ccpm_path to the CCPM "
                         "JSONL (git clone https://github.com/THUNLP-AIPoet/CCPM; "
                         "see download_zh.sh).")
    rows = _read_rows(data_path)
    rng = random.Random(seed)

    def parse(row: Dict[str, Any], i: int) -> tuple:
        translation = _first(row, ["translation", "modern", "paraphrase", "trans"])
        choices = _first(row, ["choices", "options", "candidates"])
        if isinstance(choices, str):
            choices = json.loads(choices)
        choices = [str(c) for c in choices]
        gold = _resolve_gold_index(_first(row, ["answer", "label", "gold"]), choices)
        return translation, choices, gold, str(_first(row, ["id", "qid"], i))

    parsed = [parse(r, i) for i, r in enumerate(rows)]
    prefix = _mc_fewshot_prefix(
        [(CCPM_TEMPLATE.format(translation=t), [" " + c for c in ch], g)
         for t, ch, g, _ in parsed],
        num_fewshot, rng,
    )
    examples = []
    for translation, choices, gold, qid in parsed:
        ctx = prefix + CCPM_TEMPLATE.format(translation=translation)
        examples.append(MCExample(
            qid=qid, context=ctx, options=[" " + c for c in choices],
            gold=gold, meta={"translation": translation, "choices": choices},
        ))
    if limit:
        examples = examples[:limit]
    return MCTask(name="ccpm", examples=examples, score_mode="continuation")


LOADERS_ZH = {
    "chid": load_chid,
    "chengyu_bench": load_chengyu_bench,
    "cmmlu": load_cmmlu,
    "ccpm": load_ccpm,
}
