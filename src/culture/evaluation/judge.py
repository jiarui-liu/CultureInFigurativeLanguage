"""OpenAI LLM-as-judge for the IdiomCE generation task.

IdiomCE is reference-less by design (one English idiom maps to many valid Hindi
renderings), so we grade each model translation with an OpenAI judge on a 1-5
rubric for *idiomatic adequacy* and fluency, plus a boolean for whether the
figurative meaning was rendered idiomatically (vs. translated literally).

The judge is reference-guided when a reference translation is available. Single-
answer grading is used (not pairwise), so position bias does not apply; set
``temperature=0`` for determinism.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from culture.models.llm_utils import ChatModel

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are an expert bilingual (English-Hindi) evaluator of idiomatic translation. "
    "The English source contains an idiom or figurative expression. A GOOD Hindi "
    "translation conveys the FIGURATIVE meaning naturally -- ideally with a Hindi "
    "idiom/muhavara or a natural paraphrase -- rather than a literal word-for-word "
    "rendering that loses the figurative sense. Judge strictly and consistently."
)

JUDGE_USER_TEMPLATE = """Evaluate the following English->Hindi translation.

English source: {source}
{idiom_line}Model's Hindi translation: {hypothesis}
{reference_line}
Score on these axes:
- idiom_score (1-5): how well the figurative/idiomatic meaning is conveyed in Hindi
  (5 = fully natural idiomatic rendering; 1 = literal or wrong meaning).
- fluency (1-5): grammatical fluency and naturalness of the Hindi.
- idiom_rendered (true/false): true if the figurative meaning is rendered
  idiomatically/naturally, false if it is a literal word-for-word translation.

Respond with ONLY a JSON object:
{{"idiom_score": <int>, "fluency": <int>, "idiom_rendered": <bool>, "rationale": "<short>"}}"""


def _build_messages(rec: Dict[str, Any]) -> List[Dict[str, str]]:
    idiom_line = f'English idiom: "{rec["idiom_en"]}"\n' if rec.get("idiom_en") else ""
    reference_line = (f'A valid reference Hindi translation: {rec["reference"]}\n'
                      if rec.get("reference") else "")
    user = JUDGE_USER_TEMPLATE.format(
        source=rec["source"],
        hypothesis=rec["hypothesis"],
        idiom_line=idiom_line,
        reference_line=reference_line,
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def _parse(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {"idiom_score": None, "fluency": None, "idiom_rendered": None,
                "rationale": "empty judge response", "parse_error": True}
    try:
        # Tolerate code fences / surrounding prose.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
        return {
            "idiom_score": int(obj["idiom_score"]),
            "fluency": int(obj["fluency"]),
            "idiom_rendered": bool(obj["idiom_rendered"]),
            "rationale": str(obj.get("rationale", "")),
            "parse_error": False,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to parse judge output: %s | raw=%r", e, raw)
        return {"idiom_score": None, "fluency": None, "idiom_rendered": None,
                "rationale": f"parse error: {e}", "parse_error": True}


def judge_translations(
    records: List[Dict[str, Any]],
    judge_model: str = "gpt-4o",
    provider: str = "openai",
    batch_size: int = 20,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """Grade a list of records (each with source/hypothesis/[reference]/[idiom_en]).

    Returns the same records augmented with the parsed judge fields.
    """
    model = ChatModel(model=judge_model, provider=provider)
    indexed = [(i, _build_messages(rec)) for i, rec in enumerate(records)]

    gen_kwargs: Dict[str, Any] = {"temperature": temperature,
                                  "response_format": {"type": "json_object"}}
    results = model.batch_generate_with_indices_sync(indexed, batch_size=batch_size, **gen_kwargs)

    verdicts: Dict[int, str] = {}
    for idx, response, err in results:
        if err is not None:
            logger.warning("Judge error on record %s: %s", idx, err)
        verdicts[idx] = response

    out = []
    for i, rec in enumerate(records):
        out.append({**rec, **_parse(verdicts.get(i))})
    return out


def aggregate(judged: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean idiom_score, mean fluency, and idiomatic-rendering rate."""
    idiom = [r["idiom_score"] for r in judged if r.get("idiom_score") is not None]
    fluency = [r["fluency"] for r in judged if r.get("fluency") is not None]
    rendered = [r["idiom_rendered"] for r in judged if r.get("idiom_rendered") is not None]
    n_err = sum(1 for r in judged if r.get("parse_error"))
    return {
        "n": len(judged),
        "n_judged": len(idiom),
        "n_parse_error": n_err,
        "idiom_score_mean": round(sum(idiom) / len(idiom), 4) if idiom else None,
        "fluency_mean": round(sum(fluency) / len(fluency), 4) if fluency else None,
        "idiom_rendered_rate": round(sum(rendered) / len(rendered), 4) if rendered else None,
    }
