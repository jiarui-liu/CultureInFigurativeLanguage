"""
Generate Cultural Instruction Tuning Dataset
=============================================

Creates instruction tuning data for cultural alignment of language models by
integrating CultureBank contexts with idiom data and cross-lingual cultural analysis.

For each CultureBank context (e.g., "Americans tip servers in restaurants"):
  1. Draw a batch of candidate idioms from a shuffled pool (each idiom used at most once)
  2. Present candidates + figurative meanings to the LLM alongside the cultural context
  3. LLM picks the best-fitting idiom and generates a culture-focused QA pair

English prompts for English idioms, Chinese prompts for Chinese idioms.

Usage:
    python generate_cultural_instruction_dataset.py --language en --num_samples 5000 \
        --model gpt-4o --provider openai --batch_size 20

    python generate_cultural_instruction_dataset.py --language both --num_samples 10000 \
        --model gpt-4o --provider openai
"""

import json
import random
import argparse
import asyncio
import re
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import Counter

import pandas as pd
from tqdm import tqdm

# Add parent directories for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent.parent))  # src/culture level

try:
    from culture.models.llm_utils import ChatModel
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Paths
# ============================================================================
_DATA_ROOT = _SCRIPT_DIR.parent.parent.parent.parent / "culture" / "data"
CULTUREBANK_DIR = _DATA_ROOT / "CultureBank"
IDIOMS_DIR = _DATA_ROOT / "idioms"
CULTURAL_ANALYSIS_PATH = IDIOMS_DIR / "cross_lingual_analysis" / "cultural_analysis_clusters.json"
OUTPUT_DIR = _DATA_ROOT / "training"

IDIOM_PATHS = {
    "en": IDIOMS_DIR / "en" / "idioms_merged_llm_formatted_figurative_only.jsonl",
    "zh": IDIOMS_DIR / "zh" / "idioms_merged_llm_formatted.jsonl",
}


# ============================================================================
# Answer Types — all focused on CULTURE, not idiom definitions
# ============================================================================
class AnswerType(str, Enum):
    CULTURAL_BEHAVIOR_EXPLANATION = "cultural_behavior_explanation"
    CULTURAL_VALUE_ANALYSIS = "cultural_value_analysis"
    CULTURAL_NORM_REASONING = "cultural_norm_reasoning"


ANSWER_TYPE_LIST = list(AnswerType)


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class CultureBankContext:
    cultural_group: str
    context: str
    goal: str
    relation: str
    topic: str
    actor_behavior: str
    eval_whole_desc: str = ""
    source: str = ""


@dataclass
class IdiomEntry:
    idiom: str
    entities: List[str] = field(default_factory=list)
    literal_meanings: List[str] = field(default_factory=list)
    figurative_meanings: List[str] = field(default_factory=list)


@dataclass
class CulturalAnalysis:
    english_primary_meanings: List[str] = field(default_factory=list)
    chinese_primary_meanings: List[str] = field(default_factory=list)
    shared_meanings: List[str] = field(default_factory=list)
    english_unique_aspects: List[str] = field(default_factory=list)
    chinese_unique_aspects: List[str] = field(default_factory=list)
    cultural_explanation: str = ""
    summary: str = ""


# ============================================================================
# Data Loading
# ============================================================================
def _flatten(items) -> List[str]:
    out: List[str] = []
    for item in (items or []):
        if isinstance(item, list):
            out.extend(_flatten(item))
        elif isinstance(item, str):
            out.append(item)
        else:
            out.append(str(item))
    return out


def load_culturebank() -> pd.DataFrame:
    """Load CultureBank CSV files (reddit + tiktok)."""
    dfs = []
    for name in ("culturebank_reddit.csv", "culturebank_tiktok.csv"):
        path = CULTUREBANK_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            df["source"] = name.split("_")[1].split(".")[0]
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CultureBank data found in {CULTUREBANK_DIR}")
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"cultural group": "cultural_group"})
    text_cols = ["cultural_group", "context", "goal", "relation", "topic",
                 "actor_behavior", "eval_whole_desc"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def filter_culturebank_by_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Keep only CultureBank entries whose cultural_group matches the language."""
    keyword = "chinese" if language == "zh" else "english"
    mask = df["cultural_group"].str.lower().str.contains(keyword, na=False)
    filtered = df[mask].reset_index(drop=True)
    logger.info("Filtered CultureBank for %s ('%s'): %d / %d entries",
                language, keyword, len(filtered), len(df))
    return filtered


def load_idioms(language: str) -> List[IdiomEntry]:
    """Load idioms from JSONL, keeping only those with figurative meanings."""
    path = IDIOM_PATHS.get(language)
    if not path or not path.exists():
        raise FileNotFoundError(f"Idiom file not found for language={language}: {path}")
    idioms: List[IdiomEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            output = data.get("output") or {}
            figurative = _flatten(output.get("figurative_meanings", []))
            if figurative:
                idioms.append(IdiomEntry(
                    idiom=data["idiom"],
                    entities=output.get("entities", []) or [],
                    literal_meanings=_flatten(output.get("literal_meanings", [])),
                    figurative_meanings=figurative,
                ))
    return idioms


def _parse_analysis_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    try:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            text = re.sub(r"```\w*\s*", "", text).replace("```", "")
        return json.loads(text)
    except (json.JSONDecodeError, Exception):
        return None


def load_cultural_analysis() -> Dict[str, CulturalAnalysis]:
    """Load cross-lingual cultural analysis, indexed by entity."""
    if not CULTURAL_ANALYSIS_PATH.exists():
        logger.warning("Cultural analysis file not found: %s", CULTURAL_ANALYSIS_PATH)
        return {}
    with open(CULTURAL_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    index: Dict[str, CulturalAnalysis] = {}
    for cluster in clusters:
        parsed = _parse_analysis_json(cluster.get("cultural_analysis", ""))
        if not parsed:
            continue
        ca = CulturalAnalysis(
            english_primary_meanings=parsed.get("english_primary_meanings", []),
            chinese_primary_meanings=parsed.get("chinese_primary_meanings", []),
            shared_meanings=parsed.get("shared_meanings", []),
            english_unique_aspects=parsed.get("english_unique_aspects", []),
            chinese_unique_aspects=parsed.get("chinese_unique_aspects", []),
            cultural_explanation=parsed.get("cultural_explanation", ""),
            summary=parsed.get("summary", ""),
        )
        for entity in cluster.get("en_cluster", {}).get("cluster_entities", []):
            index[entity] = ca
        for entity in cluster.get("zh_cluster", {}).get("cluster_entities", []):
            index[entity] = ca
    return index


# ============================================================================
# Shuffled Idiom Pool — each idiom used at most once
# ============================================================================
class IdiomPool:
    """A shuffled pool of idioms. Draw candidates in batches; each idiom is
    consumed at most once, guaranteeing maximum coverage and no repeats."""

    def __init__(self, idioms: List[IdiomEntry], seed: int = 42):
        self._all = list(idioms)
        rng = random.Random(seed)
        rng.shuffle(self._all)
        self._cursor = 0

    def draw(self, n: int) -> List[IdiomEntry]:
        """Draw the next *n* idioms from the pool. Returns fewer if exhausted."""
        batch = self._all[self._cursor:self._cursor + n]
        self._cursor += len(batch)
        return batch

    @property
    def remaining(self) -> int:
        return max(0, len(self._all) - self._cursor)

    @property
    def total(self) -> int:
        return len(self._all)

    @property
    def consumed(self) -> int:
        return self._cursor


# ============================================================================
# Answer Type Selection
# ============================================================================
class AnswerTypeSelector:
    def __init__(self):
        self._idx = 0

    def select(self, has_cultural_analysis: bool) -> AnswerType:
        if random.random() < 0.2:
            return random.choice(ANSWER_TYPE_LIST)
        t = ANSWER_TYPE_LIST[self._idx % len(ANSWER_TYPE_LIST)]
        self._idx += 1
        return t


# ============================================================================
# Prompts — English (culture-focused, idiom is background context only)
# ============================================================================
EN_SYSTEM_PROMPT = (
    "You are a native English speaker from Britain or America. "
    "You understand English-speaking culture deeply — not as an academic, "
    "but as someone who grew up in it. You create natural QA pairs that help "
    "language learners understand cultural behaviors, norms, and values.\n\n"
    "Rules:\n"
    "- From the idioms provided, pick ONLY 1-3 that genuinely fit. Ignore the rest.\n"
    "- Never force an idiom into the answer. If it doesn't fit naturally, skip it.\n"
    "- Write naturally, like explaining to a curious friend who is learning English.\n"
    "- Do NOT use stiff framing like 'In English-speaking culture...' or "
    "'English people tend to...' — just talk normally.\n"
    "- Do NOT reveal that you were given idioms or background context to work with.\n"
    "- The 'input' question should sound like a real person asked it — "
    "someone learning English or exploring the culture.\n"
    "- The 'output' should directly and naturally answer the question, "
    "weaving in relevant idioms to help explain.\n"
    "- Never fabricate cultural claims.\n\n"
    "Output valid JSON only: {{\"input\": \"...\", \"output\": \"...\"}}"
)

# Instruction template stored in training output — includes idioms + meanings
EN_INSTRUCTION_TEMPLATE = (
    "Answer the following question about cultural behaviors or values. "
    "Where they naturally fit, you may use the idioms below to help explain.\n\n"
    "Relevant Idioms:\n{idiom_knowledge}"
)
ZH_INSTRUCTION_TEMPLATE = (
    "请回答以下关于文化行为或价值观的问题。"
    "在合适的地方，可以引用下面的成语来辅助说明。\n\n"
    "相关成语：\n{idiom_knowledge}"
)

EN_USER_PROMPTS = {
    AnswerType.CULTURAL_BEHAVIOR_EXPLANATION: """Generate a natural QA pair where someone asks about a cultural behavior they've noticed.

## Background (use for context only — do NOT reproduce metadata in the output)
- People: {cultural_group}
- Setting: {context}
- Topic: {topic}
- Behavior: {actor_behavior}
- Relationship: {relation}
- Goal: {goal}
{desc_block}
## Idiom Pool (pick ONLY 1-3 that genuinely fit; ignore the rest)
{idiom_knowledge_block}

## Requirements
"input": A natural question from someone learning English or getting to know the culture. Should sound like a real person — "I noticed that...", "Why do people...", "How come my flatmate always...". Keep it specific and personal, not academic. Do NOT mention metadata like "cultural group" or "topic".

"output": A natural, friendly answer (150-250 words) that:
1. Directly explains why people behave this way
2. Naturally uses 1-3 fitting idioms to illustrate the cultural point — briefly explain what each means and why it captures this attitude
3. Sounds like a friend explaining over coffee, not a textbook
4. Does NOT start with "In English culture..." or "This behavior reflects..."
5. Does NOT mention that idioms were "provided" or reference any generation process

Output JSON only:
{{"input": "...", "output": "..."}}""",

    AnswerType.CULTURAL_VALUE_ANALYSIS: """Generate a natural QA pair exploring the deeper values behind a cultural pattern.

## Background (use for context only — do NOT reproduce metadata in the output)
- People: {cultural_group}
- Setting: {context}
- Topic: {topic}
- Behavior: {actor_behavior}
{desc_block}
## Idiom Pool (pick ONLY 1-3 that genuinely fit; ignore the rest)
{idiom_knowledge_block}

## Requirements
"input": A genuine, curious question about why people think or feel a certain way about {topic}. Should sound like someone trying to understand the mindset — not an essay prompt. Example tone: "What's the deal with...", "I've always wondered why...", "Why is it that...".

"output": A natural, insightful answer (150-250 words) that:
1. Identifies 1-2 core values or attitudes at play
2. Uses 1-3 fitting idioms to illustrate these values — explain what they mean and what they reveal about the cultural mindset
3. Connects the idioms' meanings to the actual behavior or attitude being discussed
4. Reads naturally, without academic framing or stiff transitions like "The idiom X reflects..."

Output JSON only:
{{"input": "...", "output": "..."}}""",

    AnswerType.CULTURAL_NORM_REASONING: """Generate a natural QA pair about navigating cultural norms in a specific situation.

## Background (use for context only — do NOT reproduce metadata in the output)
- People: {cultural_group}
- Setting: {context}
- Topic: {topic}
- Social dynamics: {actor_behavior}
- Relationship: {relation}
- Goal: {goal}
{desc_block}
## Idiom Pool (pick ONLY 1-3 that genuinely fit; ignore the rest)
{idiom_knowledge_block}

## Requirements
"input": A natural question from someone unsure how to act in a social situation related to {topic}. Write it as a real person would ask — "I've been invited to... and I'm not sure if...", "My colleague did X, should I...?". Should feel like something a language/culture learner would genuinely ask.

"output": A helpful, natural answer (150-250 words) that:
1. Explains what's expected and why
2. Uses 1-3 fitting idioms to shed light on the cultural reasoning — explain what they mean and how they relate
3. Gives practical, friendly advice rather than a cultural lecture
4. Does NOT start with "The cultural norm here is..." or "In this society..."

Output JSON only:
{{"input": "...", "output": "..."}}""",
}


# ============================================================================
# Prompts — Chinese (culture-focused, native Chinese)
# ============================================================================
ZH_SYSTEM_PROMPT = (
    "你是一个土生土长的中国人，对中国文化有深刻而直觉的理解——"
    "不是作为学者，而是作为一个在这种文化中长大的人。"
    "你负责创建自然的问答对，帮助学中文的人理解中国文化的行为、规范和价值观。\n\n"
    "规则：\n"
    "- 从提供的成语中，只选1-3个真正贴切的，其余忽略。\n"
    "- 绝不生硬地把成语塞进回答。不合适就不用。\n"
    "- 用自然的语气——像给一个正在学中文的好奇朋友解释，而不是做学术报告。\n"
    "- 不要动不动就用'在中国文化中……'开头。说人话。\n"
    "- 不要暴露你被提供了成语或背景信息这件事。\n"
    "- 'input'里的问题要像真人会问的——一个正在学中文或了解中国文化的人。\n"
    "- 'output'要直接、自然地回答问题，在合适的地方引用成语来帮助解释。\n"
    "- 全部使用纯中文输出，不得夹杂英文。\n"
    "- 不得编造文化主张。\n\n"
    "只输出有效的JSON：{{\"input\": \"...\", \"output\": \"...\"}}"
)

ZH_USER_PROMPTS = {
    AnswerType.CULTURAL_BEHAVIOR_EXPLANATION: """请生成一个自然的问答对，关于某个文化行为的疑问。

## 背景信息（仅供参考，不要在输出中复述这些元数据）
- 人群：{cultural_group}
- 场景：{context}
- 话题：{topic}
- 行为：{actor_behavior}
- 关系：{relation}
- 目的：{goal}
{desc_block}
## 成语候选（只选1-3个真正合适的，其余忽略）
{idiom_knowledge_block}

## 要求
"input"：一个正在学中文或了解中国文化的人自然提出的问题。要像真人说话——"我发现……"、"为什么大家都……"、"我朋友总是……这是怎么回事？"。具体、真实，不要提及元数据。

"output"：自然流畅的解释（150-250字）：
1. 直接回答为什么会有这种行为
2. 自然地用1-3个贴切的成语来帮助说明——简单解释含义，说明它为什么能体现这种文化态度
3. 像朋友聊天一样，不是做学术报告
4. 不要用"在中国文化中……"这样的套话开头

只输出JSON：
{{"input": "...", "output": "..."}}""",

    AnswerType.CULTURAL_VALUE_ANALYSIS: """请生成一个探讨文化深层价值观的自然问答对。

## 背景信息（仅供参考，不要在输出中复述这些元数据）
- 人群：{cultural_group}
- 场景：{context}
- 话题：{topic}
- 行为：{actor_behavior}
{desc_block}
## 成语候选（只选1-3个真正合适的，其余忽略）
{idiom_knowledge_block}

## 要求
"input"：一个有深度但自然的问题，想理解中国人在{topic}方面的思维方式或价值观。要像真正好奇的人会问的，不是论文题目。

"output"：自然、有洞察力的回答（150-250字）：
1. 点出1-2个核心价值观或态度
2. 用1-3个贴切的成语来体现这些价值观——解释含义，说明它们为什么能代表这种文化心态
3. 把成语的内涵和实际行为自然联系起来
4. 读起来流畅自然，不要学术腔，不要用"成语X反映了……"这样的句式

只输出JSON：
{{"input": "...", "output": "..."}}""",

    AnswerType.CULTURAL_NORM_REASONING: """请生成一个关于社交场景中文化规范的自然问答对。

## 背景信息（仅供参考，不要在输出中复述这些元数据）
- 人群：{cultural_group}
- 场景：{context}
- 话题：{topic}
- 社交动态：{actor_behavior}
- 关系：{relation}
- 目的：{goal}
{desc_block}
## 成语候选（只选1-3个真正合适的，其余忽略）
{idiom_knowledge_block}

## 要求
"input"：一个人在和{topic}相关的社交场景中不确定该怎么做。像真人会问的——"我被邀请去……该怎么做？"、"我同事总是……我应该跟着做吗？"。像一个正在学习这个文化的人会提出的问题。

"output"：实用、自然的回答（150-250字）：
1. 说清楚该怎么做以及为什么
2. 用1-3个贴切的成语来说明背后的道理——解释含义，说明它为什么能解释这个文化逻辑
3. 像朋友给建议一样说话，不要写分析报告
4. 不要用"这里的文化规范是……"这样的套话开头

只输出JSON：
{{"input": "...", "output": "..."}}""",
}


# ============================================================================
# Prompt Builder
# ============================================================================
def _join_list(items: List[str], max_items: int = 5) -> str:
    if not items:
        return "N/A"
    return "; ".join(items[:max_items])


def _build_analysis_block(ca: Optional[CulturalAnalysis], language: str) -> str:
    # Cultural analysis is cross-cultural comparison data — not included.
    # We only use idioms + their figurative/literal meanings.
    return ""


def _format_idiom_knowledge(candidates: List[IdiomEntry], language: str) -> str:
    """Format idioms and their meanings as a pool of cultural knowledge."""
    lines = []
    for ie in candidates:
        fig = "; ".join(ie.figurative_meanings[:3])
        lit = "; ".join(ie.literal_meanings[:2]) if ie.literal_meanings else ""
        if language == "zh":
            entry = f"- {ie.idiom}: {fig}"
            if lit:
                entry += f" (出处: {lit})"
        else:
            entry = f'- "{ie.idiom}": {fig}'
            if lit:
                entry += f" (literal: {lit})"
        lines.append(entry)
    return "\n".join(lines)


def build_prompt_messages(
    answer_type: AnswerType,
    language: str,
    ctx: CultureBankContext,
    candidates: List[IdiomEntry],
    ca: Optional[CulturalAnalysis],
) -> List[Dict[str, str]]:
    """Build system + user messages for the LLM.

    *candidates* is a list of idiom options; the LLM picks the best fit.
    """
    if language == "zh":
        system = ZH_SYSTEM_PROMPT
        template = ZH_USER_PROMPTS[answer_type]
    else:
        system = EN_SYSTEM_PROMPT
        template = EN_USER_PROMPTS[answer_type]

    if ctx.eval_whole_desc:
        if language == "zh":
            desc_block = "- " + ctx.eval_whole_desc[:600] + "\n\n"
        else:
            desc_block = "- Full Description: " + ctx.eval_whole_desc[:600] + "\n\n"
    else:
        desc_block = "\n"

    analysis_block = _build_analysis_block(ca, language)
    idiom_knowledge_block = _format_idiom_knowledge(candidates, language)

    # For ZH, translate the cultural_group field to Chinese
    cultural_group = ctx.cultural_group
    if language == "zh" and cultural_group:
        _ZH_GROUP_MAP = {
            "chinese": "中国人", "chinese people": "中国人",
            "chinese students": "中国学生", "chinese immigrants": "中国移民",
            "chinese americans": "华裔美国人", "malaysian chinese": "马来西亚华人",
        }
        mapped = _ZH_GROUP_MAP.get(cultural_group.lower().strip())
        if mapped:
            cultural_group = mapped
        elif "chinese" in cultural_group.lower():
            cultural_group = cultural_group.lower().replace("chinese", "中国").strip()

    user_content = template.format(
        cultural_group=cultural_group or ("General" if language == "en" else "一般"),
        context=ctx.context or ("everyday situations" if language == "en" else "日常生活"),
        topic=ctx.topic or ("Culture" if language == "en" else "文化"),
        actor_behavior=ctx.actor_behavior or ("various behaviors" if language == "en" else "各种行为"),
        goal=ctx.goal or "N/A",
        relation=ctx.relation or "N/A",
        idiom_knowledge_block=idiom_knowledge_block,
        desc_block=desc_block,
        analysis_block=analysis_block,
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ============================================================================
# LLM Generation
# ============================================================================
def _parse_llm_response(text: str) -> Optional[Dict[str, str]]:
    text = text.strip()
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        text = re.sub(r"```\w*\s*", "", text).replace("```", "")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "input" in obj and "output" in obj:
            return {"input": str(obj["input"]), "output": str(obj["output"])}
    except json.JSONDecodeError:
        pass
    return None


def _validate_example(result: Dict[str, str], language: str) -> bool:
    inp = result.get("input", "")
    out = result.get("output", "")
    if not inp or not out:
        return False
    if inp == out:
        return False
    min_out_len = 50 if language == "zh" else 100
    if len(out) < min_out_len:
        return False
    return True


# ============================================================================
# Main Pipeline
# ============================================================================
async def generate_dataset(
    language: str = "zh",
    num_samples: int = 1000,
    model: str = "gpt-4o",
    provider: str = "openai",
    output_file: Optional[str] = None,
    batch_size: int = 20,
    seed: int = 42,
    candidates_per_context: int = 5,
) -> List[Dict]:
    """Generate the cultural instruction tuning dataset.

    Idiom selection strategy:
      - Shuffle all idioms into a pool.
      - For each CultureBank context, draw ``candidates_per_context`` idioms.
      - Pass all candidates (with figurative meanings) to the LLM.
      - The LLM picks the best-fitting one and generates the QA pair.
      - Drawn idioms are consumed and never reused.
    """
    random.seed(seed)

    # --- Load data ---
    logger.info("Loading CultureBank...")
    cb_df = load_culturebank()
    cb_df = filter_culturebank_by_language(cb_df, language)

    logger.info("Loading %s idioms...", language)
    idioms = load_idioms(language)
    logger.info("Loaded %d idioms with figurative meanings", len(idioms))

    logger.info("Loading cultural analysis...")
    ca_index = load_cultural_analysis()
    logger.info("Loaded analysis for %d entities", len(ca_index))

    # --- Init ---
    pool = IdiomPool(idioms, seed=seed)
    type_selector = AnswerTypeSelector()

    if not HAS_LLM:
        raise RuntimeError(
            "LLM is required for generation. Install culture.models.llm_utils "
            "and provide --model / --provider."
        )
    logger.info("Initializing %s via %s...", model, provider)
    chat_model = ChatModel(model=model, provider=provider)

    # --- Sample CultureBank entries (shuffled) ---
    n = min(num_samples, len(cb_df))
    sampled = cb_df.sample(n=n, random_state=seed).reset_index(drop=True)
    logger.info("Sampled %d CultureBank entries", n)

    # How many contexts we can actually serve (pool may be smaller)
    max_contexts = pool.total // candidates_per_context
    if n > max_contexts:
        logger.warning(
            "Requested %d samples but only %d idioms / %d candidates = %d contexts possible. "
            "Capping at %d.", n, pool.total, candidates_per_context, max_contexts, max_contexts,
        )
        sampled = sampled.iloc[:max_contexts]

    # --- Prepare work items ---
    WorkItem = Tuple[int, CultureBankContext, List[IdiomEntry], Optional[CulturalAnalysis], AnswerType]
    work_items: List[WorkItem] = []

    for idx, row in sampled.iterrows():
        candidates = pool.draw(candidates_per_context)
        if not candidates:
            break  # pool exhausted

        # Look up cultural analysis for any candidate's entities
        ca = None
        for ie in candidates:
            for entity in ie.entities:
                if entity in ca_index:
                    ca = ca_index[entity]
                    break
            if ca:
                break

        has_ca = ca is not None
        atype = type_selector.select(has_ca)

        ctx = CultureBankContext(
            cultural_group=row.get("cultural_group", ""),
            context=row.get("context", ""),
            goal=row.get("goal", ""),
            relation=row.get("relation", ""),
            topic=row.get("topic", ""),
            actor_behavior=row.get("actor_behavior", ""),
            eval_whole_desc=row.get("eval_whole_desc", ""),
            source=row.get("source", ""),
        )
        work_items.append((idx, ctx, candidates, ca, atype))

    logger.info("Prepared %d work items (idioms consumed: %d / %d)",
                len(work_items), pool.consumed, pool.total)

    # --- Batch LLM generation ---
    dataset: List[Dict] = []
    success_count = 0
    fail_count = 0

    logger.info("Generating with LLM in batches of %d...", batch_size)
    for batch_start in tqdm(range(0, len(work_items), batch_size), desc="LLM batches"):
        batch = work_items[batch_start:batch_start + batch_size]
        batch_messages = []
        idx_to_msgs: Dict[int, List[Dict[str, str]]] = {}
        for idx, ctx, candidates, ca, atype in batch:
            msgs = build_prompt_messages(atype, language, ctx, candidates, ca)
            batch_messages.append((idx, msgs))
            idx_to_msgs[idx] = msgs

        results = await chat_model.batch_generate_with_indices(batch_messages)

        for item, batch_result in zip(batch, results):
            idx, ctx, candidates, ca, atype = item
            _, response_text, error = batch_result

            result = None
            if response_text and not error:
                result = _parse_llm_response(response_text)
                if result and not _validate_example(result, language):
                    result = None

            if result:
                success_count += 1
                idiom_knowledge = _format_idiom_knowledge(candidates, language)
                if language == "zh":
                    instruction = ZH_INSTRUCTION_TEMPLATE.format(idiom_knowledge=idiom_knowledge)
                else:
                    instruction = EN_INSTRUCTION_TEMPLATE.format(idiom_knowledge=idiom_knowledge)

                dataset.append({
                    "instruction": instruction,
                    "input": result["input"],
                    "output": result["output"],
                    "metadata": {
                        "idioms_provided": [ie.idiom for ie in candidates],
                        "language": language,
                        "answer_type": atype.value,
                        "cultural_group": ctx.cultural_group,
                        "context": ctx.context,
                        "topic": ctx.topic,
                        "actor_behavior": ctx.actor_behavior,
                        "relation": ctx.relation,
                        "goal": ctx.goal,
                        "has_cultural_analysis": ca is not None,
                        "source": ctx.source,
                    },
                    "generation_prompt": idx_to_msgs[idx],
                })
            else:
                fail_count += 1
                if error:
                    logger.debug("LLM error for item %d: %s", idx, error)

    # --- Save ---
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for ex in dataset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved %d examples to %s", len(dataset), output_file)

    # --- Stats ---
    type_dist = Counter(d["metadata"]["answer_type"] for d in dataset)
    ca_dist = Counter(d["metadata"]["has_cultural_analysis"] for d in dataset)

    logger.info("=== Generation Statistics ===")
    logger.info("Total examples: %d (success: %d, failed: %d)", len(dataset), success_count, fail_count)
    logger.info("Idiom pool: %d total, %d consumed (%d per context), %d remaining",
                pool.total, pool.consumed, candidates_per_context, pool.remaining)
    logger.info("Answer type distribution: %s", dict(type_dist))
    logger.info("With cultural analysis: %d | Without: %d", ca_dist.get(True, 0), ca_dist.get(False, 0))

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate cultural instruction tuning dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_cultural_instruction_dataset.py --language en --num_samples 5000 \\
      --model gpt-4o --provider openai

  python generate_cultural_instruction_dataset.py --language both --num_samples 10000 \\
      --model gpt-4o --provider openai --batch_size 20
        """,
    )
    parser.add_argument("--language", type=str, default="zh", choices=["en", "zh", "both"])
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--candidates_per_context", type=int, default=5,
                        help="Number of idioms (with meanings) to provide per context (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    languages = ["en", "zh"] if args.language == "both" else [args.language]

    for lang in languages:
        if args.output_file and len(languages) == 1:
            out_path = args.output_file
        else:
            out_path = os.path.join(args.output_dir, f"cultural_instruction_{lang}.jsonl")

        logger.info("=" * 60)
        logger.info("Generating %s dataset (%d samples)", lang.upper(), args.num_samples)
        logger.info("=" * 60)

        dataset = asyncio.run(generate_dataset(
            language=lang,
            num_samples=args.num_samples,
            model=args.model,
            provider=args.provider,
            output_file=out_path,
            batch_size=args.batch_size,
            seed=args.seed,
            candidates_per_context=args.candidates_per_context,
        ))

        logger.info("Done: %d %s examples\n", len(dataset), lang.upper())


if __name__ == "__main__":
    main()
