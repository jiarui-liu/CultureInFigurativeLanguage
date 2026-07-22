"""Dataset loaders + prompt templating for the four Hindi benchmarks.

Each multiple-choice loader returns an :class:`MCTask` (a list of
:class:`MCExample`, each already templated for the requested few-shot count).
The IdiomCE loader returns a :class:`GenTask` of :class:`GenExample`.

Data access is *local-file-first*: pass ``--data_path`` to a file in the schema
documented in the README. Where a clean HuggingFace id exists (MILU, Global
PIQA), the loader will fall back to ``datasets.load_dataset`` when no local path
is given. MABL and IdiomCE have no clean HF release, so a local file is required
(download instructions are in the README).
"""

import csv
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LETTERS = ["A", "B", "C", "D", "E", "F"]


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class MCExample:
    qid: str
    context: str            # fully templated prompt (incl. few-shot prefix)
    options: List[str]      # continuations scored by log-likelihood
    gold: int               # index into `options`
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenExample:
    qid: str
    prompt: str             # raw-text prompt fed to the base model
    source: str             # source (English) sentence
    reference: Optional[str] = None
    idiom_en: Optional[str] = None
    idiom_hi: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCTask:
    name: str
    examples: List[MCExample]
    score_mode: str         # "continuation" (length-normalize) | "letter"


@dataclass
class GenTask:
    name: str
    examples: List[GenExample]


# --------------------------------------------------------------------------- #
# Templates (edit here to tune prompting; kept explicit for transparency)
# --------------------------------------------------------------------------- #
MABL_TEMPLATE = "वाक्य: {startphrase}\nइसका अर्थ है:"          # "Sentence: ...\nIt means:"
GLOBAL_PIQA_TEMPLATE = "प्रश्न: {goal}\nउत्तर:"                # "Question: ...\nAnswer:"
MILU_TEMPLATE = "प्रश्न: {question}\n{options_block}\nउत्तर:"    # "Question:\n<A. ..>\nAnswer:"
IDIOMCE_FEWSHOT = [
    ("It's raining cats and dogs outside.", "बाहर मूसलाधार बारिश हो रही है।"),
    ("Don't beat around the bush, tell me the truth.", "इधर-उधर की बात मत करो, मुझे सच बताओ।"),
    ("He finally decided to bury the hatchet with his brother.", "आख़िरकार उसने अपने भाई से गिले-शिकवे मिटाने का फ़ैसला किया।"),
]
IDIOMCE_INSTRUCTION = "अंग्रेज़ी वाक्य का हिंदी में मुहावरेदार अनुवाद कीजिए।"  # "Translate idiomatically into Hindi."


# --------------------------------------------------------------------------- #
# Small IO / field helpers
# --------------------------------------------------------------------------- #
def _read_rows(path: str) -> List[Dict[str, Any]]:
    """Read .jsonl / .json / .csv / .tsv into a list of dict rows."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("data", data.get("rows", []))
    if ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter=delim))
    raise ValueError(f"Unsupported file extension for {path!r} (use jsonl/json/csv/tsv)")


def _first(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _resolve_gold_index(raw: Any, options: List[str]) -> int:
    """Map a gold answer (letter / index / option text) to an option index."""
    if isinstance(raw, int):
        # Datasets use either 0-based or 1-based indexing; normalize.
        return raw - 1 if raw >= len(options) else raw
    s = str(raw).strip()
    if s.isdigit():
        i = int(s)
        return i - 1 if i >= len(options) else i
    if len(s) == 1 and s.upper() in LETTERS:
        return LETTERS.index(s.upper())
    # Match against option text.
    for i, opt in enumerate(options):
        if opt.strip() == s:
            return i
    raise ValueError(f"Cannot resolve gold answer {raw!r} against {len(options)} options")


# --------------------------------------------------------------------------- #
# MABL (Hindi figurative-meaning inference, 2-choice)
# --------------------------------------------------------------------------- #
def load_mabl(data_path: str, num_fewshot: int = 0, limit: Optional[int] = None,
              seed: int = 42) -> MCTask:
    """MABL Hindi. Expects Fig-QA schema: startphrase, ending1, ending2, labels(0/1)."""
    if not data_path:
        raise ValueError("MABL has no clean HF release; pass --data_path to the Hindi "
                         "Fig-QA CSV (see README).")
    rows = _read_rows(data_path)
    rng = random.Random(seed)

    def raw_ex(row, i):
        startphrase = _first(row, ["startphrase", "sentence", "premise"])
        e1 = _first(row, ["ending1", "ending_1", "option1", "meaning1"])
        e2 = _first(row, ["ending2", "ending_2", "option2", "meaning2"])
        gold = int(_first(row, ["labels", "label", "answer", "gold"]))
        return startphrase, [e1, e2], gold, str(_first(row, ["qid", "id"], i))

    parsed = [raw_ex(r, i) for i, r in enumerate(rows)]
    prefix = _mc_fewshot_prefix(
        [(MABL_TEMPLATE.format(startphrase=sp), opts, g) for sp, opts, g, _ in parsed],
        num_fewshot, rng,
    )
    examples = []
    for sp, opts, gold, qid in parsed:
        ctx = prefix + MABL_TEMPLATE.format(startphrase=sp)
        examples.append(MCExample(qid=qid, context=ctx,
                                  options=[" " + o.strip() for o in opts],
                                  gold=gold, meta={"startphrase": sp}))
    if limit:
        examples = examples[:limit]
    return MCTask(name="mabl", examples=examples, score_mode="continuation")


# --------------------------------------------------------------------------- #
# Global PIQA (Hindi cultural physical commonsense, 2-choice)
# --------------------------------------------------------------------------- #
def load_global_piqa(data_path: str, num_fewshot: int = 0, limit: Optional[int] = None,
                     seed: int = 42, hf_config: str = "hin_deva") -> MCTask:
    """Global PIQA Hindi. Schema: prompt/goal, solution0, solution1, label(0/1)."""
    rng = random.Random(seed)
    if data_path:
        rows = _read_rows(data_path)
    else:
        rows = _load_hf("mrlbenchmarks/global-piqa-nonparallel", split="test",
                        config=hf_config)

    def raw_ex(row, i):
        goal = _first(row, ["prompt", "goal", "question"])
        s0 = _first(row, ["solution0", "sol1", "choice1", "option0"])
        s1 = _first(row, ["solution1", "sol2", "choice2", "option1"])
        gold = int(_first(row, ["label", "answer", "gold"]))
        return goal, [s0, s1], gold, str(_first(row, ["id", "qid"], i))

    parsed = [raw_ex(r, i) for i, r in enumerate(rows)]
    prefix = _mc_fewshot_prefix(
        [(GLOBAL_PIQA_TEMPLATE.format(goal=g), opts, gd) for g, opts, gd, _ in parsed],
        num_fewshot, rng,
    )
    examples = []
    for goal, opts, gold, qid in parsed:
        ctx = prefix + GLOBAL_PIQA_TEMPLATE.format(goal=goal)
        examples.append(MCExample(qid=qid, context=ctx,
                                  options=[" " + o.strip() for o in opts],
                                  gold=gold, meta={"goal": goal}))
    if limit:
        examples = examples[:limit]
    return MCTask(name="global_piqa", examples=examples, score_mode="continuation")


# --------------------------------------------------------------------------- #
# MILU (Hindi cultural-knowledge exam QA, 4-choice, MMLU-style)
# --------------------------------------------------------------------------- #
def load_milu(data_path: str, num_fewshot: int = 5, limit: Optional[int] = None,
              seed: int = 42, hf_config: str = "Hindi",
              fewshot_path: Optional[str] = None) -> MCTask:
    """MILU Hindi. MMLU-style: score the answer letter (A/B/C/D)."""
    rng = random.Random(seed)
    if data_path:
        rows = _read_rows(data_path)
    else:
        rows = _load_hf("ai4bharat/MILU", split="test", config=hf_config)

    def parse(row, i):
        question = _first(row, ["question", "Question"])
        opts = _first(row, ["options", "choices"])
        if isinstance(opts, str):
            opts = json.loads(opts)
        if not opts:
            opts = [_first(row, [f"option{j}", f"option_{j}", f"option{c.lower()}",
                                 f"option{c}", c]) for j, c in enumerate(LETTERS[:4], 1)]
            opts = [o for o in opts if o is not None]
        opts = [str(o) for o in opts]
        gold_raw = _first(row, ["answer", "target", "label", "correct_option", "gold"])
        gold = _resolve_gold_index(gold_raw, opts)
        return question, opts, gold, str(_first(row, ["id", "qid"], i)), \
            _first(row, ["domain", "subject"], "")

    parsed = [parse(r, i) for i, r in enumerate(rows)]

    # Few-shot exemplars from a separate pool (validation) to avoid test leakage.
    pool = parsed
    if fewshot_path:
        pool = [parse(r, i) for i, r in enumerate(_read_rows(fewshot_path))]

    def block(q, opts):
        lines = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        return MILU_TEMPLATE.format(question=q, options_block=lines)

    prefix = ""
    if num_fewshot > 0:
        shots = rng.sample(pool, min(num_fewshot, len(pool)))
        prefix = "".join(f"{block(q, o)} {LETTERS[g]}\n\n" for q, o, g, _, _ in shots)

    examples = []
    for q, opts, gold, qid, dom in parsed:
        ctx = prefix + block(q, opts)
        examples.append(MCExample(
            qid=qid, context=ctx,
            options=[" " + LETTERS[j] for j in range(len(opts))],
            gold=gold, meta={"domain": dom, "option_text": opts},
        ))
    if limit:
        examples = examples[:limit]
    return MCTask(name="milu", examples=examples, score_mode="letter")


# --------------------------------------------------------------------------- #
# IdiomCE (English->Hindi idiomatic translation, generation + judge)
# --------------------------------------------------------------------------- #
def load_idiomce(data_path: str, limit: Optional[int] = None,
                 num_fewshot: int = 3) -> GenTask:
    """IdiomCE. Expects JSONL rows: {source, reference?, idiom_en?, idiom_hi?}."""
    if not data_path:
        raise ValueError("IdiomCE has no clean HF release; pass --data_path to a JSONL "
                         "with fields {source, reference?, idiom_en?, idiom_hi?} (see README).")
    rows = _read_rows(data_path)

    shots = IDIOMCE_FEWSHOT[:num_fewshot]
    prefix = f"{IDIOMCE_INSTRUCTION}\n\n"
    for en, hi in shots:
        prefix += f"English: {en}\nHindi: {hi}\n\n"

    examples = []
    for i, row in enumerate(rows):
        src = _first(row, ["source", "source_en", "english", "en", "sentence"])
        prompt = prefix + f"English: {src}\nHindi:"
        examples.append(GenExample(
            qid=str(_first(row, ["id", "qid"], i)),
            prompt=prompt,
            source=src,
            reference=_first(row, ["reference", "reference_hi", "target", "hindi", "hi"]),
            idiom_en=_first(row, ["idiom_en", "idiom", "en_idiom"]),
            idiom_hi=_first(row, ["idiom_hi", "hi_idiom"]),
        ))
    if limit:
        examples = examples[:limit]
    return GenTask(name="idiomce", examples=examples)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _mc_fewshot_prefix(templated, num_fewshot: int, rng: random.Random) -> str:
    """Build a few-shot prefix for continuation-scored MC tasks.

    `templated` is a list of (context, options, gold). Each exemplar is rendered
    as context + correct option + blank line.
    """
    if num_fewshot <= 0:
        return ""
    shots = rng.sample(templated, min(num_fewshot, len(templated)))
    return "".join(f"{ctx} {opts[g].strip()}\n\n" for ctx, opts, g in shots)


def _load_hf(repo: str, split: str, config: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load a HuggingFace dataset split as a list of dict rows (lazy import)."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("`datasets` is required for HF loading; `pip install datasets` "
                          "or pass --data_path.") from e
    token = os.environ.get("HF_TOKEN")
    logger.info("Loading HF dataset %s (config=%s, split=%s)", repo, config, split)
    ds = load_dataset(repo, config, split=split, token=token) if config \
        else load_dataset(repo, split=split, token=token)
    return [dict(r) for r in ds]


LOADERS = {
    "mabl": load_mabl,
    "milu": load_milu,
    "global_piqa": load_global_piqa,
    "idiomce": load_idiomce,
}
