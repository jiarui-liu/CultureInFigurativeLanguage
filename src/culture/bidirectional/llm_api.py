"""Minimal cached, concurrent LLM client for the analysis scripts.

Providers:
  gemini      Google Generative Language API, keys from $gemini_api_key_1..7 (rotated)
  openrouter  OpenRouter chat completions, key from $OPEN_ROUTER_API_KEY
  anthropic   Anthropic Messages API, key from $ANTHROPIC_API_KEY or ~/.env

Every (provider, model, prompt, temperature) result is cached on disk as one JSON
line, so re-running an analysis never re-queries the API.
"""
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

CACHE_DIR = os.environ.get(
    "LLM_CACHE_DIR", "/data/group_data/r3lit_culture_pretrain/culture/bidir/llm_cache")
_lock = threading.Lock()
_mem = {}


def _keys():
    ks = [os.environ[k] for k in sorted(os.environ) if k.startswith("gemini_api_key_")]
    if not ks:
        raise RuntimeError("no gemini_api_key_* in environment")
    return ks


def _cache_path(tag):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{tag}.jsonl")


def _load_cache(tag):
    if tag in _mem:
        return _mem[tag]
    d = {}
    p = _cache_path(tag)
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try:
                o = json.loads(l)
                d[o["h"]] = o["r"]
            except Exception:
                pass
    _mem[tag] = d
    return d


def _call_gemini(model, prompt, temperature, max_tokens, json_mode):
    ks = _keys()
    cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
    for attempt in range(8):
        k = random.choice(ks)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={k}"
        try:
            r = requests.post(url, json=body, timeout=180)
            if r.status_code == 200:
                c = r.json()["candidates"][0]
                return "".join(p.get("text", "") for p in c["content"]["parts"] if not p.get("thought"))
            if r.status_code in (429, 500, 503):
                time.sleep(2 ** attempt + random.random())
                continue
            raise RuntimeError(f"gemini {r.status_code}: {r.text[:300]}")
        except (requests.RequestException, KeyError) as e:
            time.sleep(2 ** attempt)
            last = e
    raise RuntimeError(f"gemini failed after retries")


def _call_openrouter(model, prompt, temperature, max_tokens, json_mode):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    h = {"Authorization": f"Bearer {os.environ['OPEN_ROUTER_API_KEY']}"}
    for attempt in range(8):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                              headers=h, timeout=300)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"] or ""
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt + random.random())
                continue
            raise RuntimeError(f"openrouter {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("openrouter failed after retries")


def _anthropic_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k and os.path.exists(os.path.expanduser("~/.env")):
        for l in open(os.path.expanduser("~/.env")):
            if l.strip().startswith("ANTHROPIC_API_KEY"):
                k = l.split("=", 1)[1].strip().strip('"').strip("'")
    return k


def _call_anthropic(model, prompt, temperature, max_tokens, json_mode):
    h = {"x-api-key": _anthropic_key(), "anthropic-version": "2023-06-01",
         "content-type": "application/json"}
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(8):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=h, timeout=300)
            if r.status_code == 200:
                return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
            if r.status_code in (429, 500, 529, 503):
                time.sleep(2 ** attempt + random.random())
                continue
            raise RuntimeError(f"anthropic {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("anthropic failed after retries")


def complete(prompt, model="gemini-3.8-flash", provider="gemini", temperature=0.0,
             max_tokens=4096, json_mode=False, tag="default"):
    h = hashlib.sha1(json.dumps([provider, model, prompt, temperature, json_mode]).encode()).hexdigest()
    cache = _load_cache(tag)
    if h in cache:
        return cache[h]
    fn = {"gemini": _call_gemini, "openrouter": _call_openrouter,
          "anthropic": _call_anthropic}[provider]
    r = fn(model, prompt, temperature, max_tokens, json_mode)
    with _lock:
        cache[h] = r
        with open(_cache_path(tag), "a", encoding="utf-8") as f:
            f.write(json.dumps({"h": h, "r": r}, ensure_ascii=False) + "\n")
    return r


_VLLM = {}


def _vllm_many(prompts, model, temperature=0.0, max_tokens=4096, tag="default", tp=None, **_):
    """Batch generation with a local vLLM engine (one engine per model per process)."""
    cache = _load_cache(tag)
    keys = [hashlib.sha1(json.dumps(["vllm", model, p, temperature, False]).encode()).hexdigest()
            for p in prompts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        from vllm import LLM, SamplingParams
        if model not in _VLLM:
            import torch
            tp = tp or int(os.environ.get("VLLM_TP", torch.cuda.device_count()))
            extra = {"quantization": os.environ["LLM_QUANT"]} if os.environ.get("LLM_QUANT") else {}
            _VLLM[model] = LLM(model=model, tensor_parallel_size=tp, **extra,
                               max_model_len=16384, gpu_memory_utilization=0.9, max_num_seqs=128,
                               enable_prefix_caching=True)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=0)
        msgs = [[{"role": "user", "content": prompts[i]}] for i in todo]
        try:
            outs = _VLLM[model].chat(msgs, sp, use_tqdm=True,
                                     chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            outs = _VLLM[model].chat(msgs, sp, use_tqdm=True)
        with _lock, open(_cache_path(tag), "a", encoding="utf-8") as f:
            for i, o in zip(todo, outs):
                cache[keys[i]] = o.outputs[0].text
                f.write(json.dumps({"h": keys[i], "r": cache[keys[i]]}, ensure_ascii=False) + "\n")
    return [cache[k] for k in keys]


def complete_many(prompts, workers=16, **kw):
    """Run ``complete`` over many prompts concurrently; failures return None.

    With provider="vllm" (model = local path) all prompts go to one local batch."""
    if kw.get("provider") == "vllm":
        kw = {k: v for k, v in kw.items() if k not in ("provider", "json_mode")}
        return _vllm_many(prompts, **kw)
    def one(p):
        try:
            return complete(p, **kw)
        except Exception as e:  # keep going; caller filters None
            print("LLM error:", str(e)[:200])
            return None
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))


def parse_json(txt):
    if txt is None:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    s, e = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1), max(t.rfind("}"), t.rfind("]"))
    if s < 0 or e < 0:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None
