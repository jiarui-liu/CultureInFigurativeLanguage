"""Local open-weight LLMs (vLLM offline batch) with an on-disk cache in the same format as
culture.bidirectional.llm_api (one JSON line {"h": sha1, "r": response} per call).

Why local: the Gemini free tier is 20 requests/day/project/model and was exhausted, and the
OpenRouter account has no credits, so all LLM steps of the v2 analyses run on local models:
  PRIMARY = Qwen/Qwen3.5-27B-FP8        (typology, entity translation, meaning paraphrases, pair judge)
  SECOND  = Qwen/Qwen3.5-9B             (second annotator, same family)
  THIRD   = CohereLabs/aya-expanse-8b   (second annotator, different model family)
Decoding: greedy (temperature 0), thinking disabled via chat_template_kwargs.
Tensor parallelism from $VLLM_TP (default 1).
"""
import hashlib
import json
import os

CACHE_DIR = os.environ.get("LLM_CACHE_DIR", "/data/group_data/r3lit_culture_pretrain/culture/bidir/llm_cache")
PRIMARY = "/data/group_data/r3lit_culture_pretrain/models/Qwen/Qwen3.5-27B-FP8"
SECOND = "/data/group_data/r3lit_culture_pretrain/models/Qwen/Qwen3.5-9B"
THIRD = "/data/group_data/r3lit_culture_pretrain/models/CohereLabs/aya-expanse-8b"
NAMES = {PRIMARY: "Qwen3.5-27B-FP8", SECOND: "Qwen3.5-9B", THIRD: "aya-expanse-8b"}
_llm = {}


def _cache(tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, f"{tag}.jsonl")
    d = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try:
                o = json.loads(l)
                d[o["h"]] = o["r"]
            except Exception:
                pass
    return p, d


def _load(model):
    if model in _llm:
        return _llm[model]
    from vllm import LLM
    kw = dict(model=model, dtype="bfloat16", tensor_parallel_size=int(os.environ.get("VLLM_TP", "1")),
              gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM", "0.85")),
              max_model_len=int(os.environ.get("VLLM_MAX_LEN", "8192")), trust_remote_code=True,
              max_num_seqs=int(os.environ.get("VLLM_MAX_SEQS", "64")))
    if "gemma-4" in model or "Qwen3.5" in model:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0} if "gemma-4" in model else {"image": 0, "video": 0}
    _llm[model] = LLM(**kw)
    return _llm[model]


def generate(prompts, tag, model=PRIMARY, max_tokens=1024, temperature=0.0):
    """Chat completion for a list of user prompts; cached by (model, prompt, temperature)."""
    path, cache = _cache(tag)
    keys = [hashlib.sha1(json.dumps(["vllm", NAMES.get(model, model), p, temperature]).encode()).hexdigest()
            for p in prompts]
    todo = sorted({k: p for k, p in zip(keys, prompts) if k not in cache}.items())
    if todo:
        from vllm import SamplingParams
        llm = _load(model)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        msgs = [[{"role": "user", "content": p}] for _, p in todo]
        outs = llm.chat(msgs, sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
        with open(path, "a", encoding="utf-8") as f:
            for (k, _), o in zip(todo, outs):
                r = o.outputs[0].text
                cache[k] = r
                f.write(json.dumps({"h": k, "r": r}, ensure_ascii=False) + "\n")
    return [cache[k] for k in keys]
