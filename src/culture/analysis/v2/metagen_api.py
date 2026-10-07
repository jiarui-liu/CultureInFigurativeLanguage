"""Cached, concurrent client for the MetaGen / Llama OpenAI-compatible gateway.

Mirrors the contract of local_llm.generate so the analysis scripts can swap backends:
one JSON line per call in an on-disk cache keyed by (model, prompt, temperature), so
re-running an analysis never re-queries the API.

Credentials, resolved in docs/plans/hindi_cpt_eval_progress.md:
  base  https://api.llama.com/experimental/compat/openai/v1   (post-redirect URL)
  key   $LLAMA_API_KEY   -- the "LLM|<id>|<secret>" one.  $METAGEN_API_KEY is NOT
        entitled on this endpoint and returns 401, despite the name.
  model gpt-5-4-mini-genai-responses
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = os.environ.get("METAGEN_API_BASE",
                      "https://api.llama.com/experimental/compat/openai/v1")
MODEL = os.environ.get("METAGEN_MODEL", "gpt-5-4-mini-genai-responses")
CACHE_DIR = os.environ.get(
    "LLM_CACHE_DIR",
    "/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/llm_cache")

_lock = threading.Lock()
_mem: dict = {}


def _key() -> str:
    k = os.environ.get("LLAMA_API_KEY") or os.environ.get("METAGEN_API_KEY")
    if not k:
        raise RuntimeError("set LLAMA_API_KEY (preferred) or METAGEN_API_KEY")
    return k


def _cache(tag):
    if tag in _mem:
        return _mem[tag]
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, f"{tag}.jsonl")
    d = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try:
                o = json.loads(line)
                d[o["h"]] = o["r"]
            except Exception:
                pass
    _mem[tag] = d
    return d


def _one(prompt, model, temperature, max_tokens, json_mode, retries=5):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        # the gateway rejects max_output_tokens < 16
        "max_tokens": max(16, max_tokens),
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last = ""
    for a in range(retries):
        try:
            r = requests.post(f"{BASE}/chat/completions",
                              headers={"Authorization": f"Bearer {_key()}",
                                       "Content-Type": "application/json"},
                              json=body, timeout=180)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"{r.status_code} {r.text[:200]}"
            # 429/5xx are worth waiting out; 4xx otherwise are not
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:  # network blips
            last = str(e)[:200]
        time.sleep(min(2 ** a, 30))
    raise RuntimeError(f"metagen call failed: {last}")


def generate(prompts, tag, model=MODEL, max_tokens=1024, temperature=0.0,
             json_mode=False, workers=8):
    """Return one completion per prompt, cached on disk by (model, prompt, temperature)."""
    path = os.path.join(CACHE_DIR, f"{tag}.jsonl")
    cache = _cache(tag)
    keys = [hashlib.sha1(json.dumps(["metagen", model, p, temperature]).encode()).hexdigest()
            for p in prompts]
    todo = sorted({k: p for k, p in zip(keys, prompts) if k not in cache}.items())
    if todo:
        os.makedirs(CACHE_DIR, exist_ok=True)
        done = [0]

        def run(kp):
            k, p = kp
            try:
                r = _one(p, model, temperature, max_tokens, json_mode)
            except Exception as e:
                r = ""
                print(f"  [warn] {e}")
            with _lock:
                cache[k] = r
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"h": k, "r": r}, ensure_ascii=False) + "\n")
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == len(todo):
                    print(f"  [metagen] {done[0]}/{len(todo)}", flush=True)
            return r

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run, todo))
    return [cache.get(k, "") for k in keys]
