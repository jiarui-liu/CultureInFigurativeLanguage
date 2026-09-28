#!/usr/bin/env python3
"""LLM annotation of culture-specificity (0-5) for a random sample of pool documents.

The annotations train the culture classifier (``culture_classifier.py``), in the
style of FineWeb-Edu's educational-value annotations. The rubric deliberately
ignores idioms, proverbs and sayings, so that the classifier does not learn to
select idiom-dense text.

Usage:
  python -m culture.bidirectional.annotate_culture --lang ar --pool_dir $B/pool/ar \
      --n 12000 --model <instruct model> --tp 2 --out $B/annot/ar.jsonl
"""
import argparse
import glob
import gzip
import json
import os
import random
import re

CULTURE = {
    "ar": ("Arabic", "Arab / Arabic-speaking communities (any Arab country or region, including dialect communities)"),
    "zh": ("Chinese", "Chinese communities (mainland China, Taiwan, Hong Kong, and the Chinese diaspora)"),
    "hi": ("Hindi", "Indian / Hindi-speaking communities (North India and the Indian diaspora)"),
}

PROMPT = """You are rating web documents for a study of cultural knowledge in language models.

The document below is written in {lang}. Rate how much it conveys knowledge that is SPECIFIC TO THE CULTURE of {community}, as opposed to generic, global, commercial, or technical content.

Cultural knowledge includes: customs and rituals; festivals and holidays; food and cuisine; religion and religious practice; social norms, etiquette, family and kinship practices; values and beliefs; folklore, myths, legends; history and historical figures as they matter to the community; traditional arts, literature, poetry, music, dress, crafts; local institutions and ways of life.

Do NOT give credit for idioms, proverbs, or sayings that appear in the document; rate only the content described above. Merely being written in {lang}, or mentioning a place name or a local company, is not cultural knowledge.

Scale:
0 = no culture-specific content (e.g., product pages, software, sports scores, generic health or finance advice, international news).
1 = local context only incidental (place names, local brands, local politics) with no cultural knowledge.
2 = some culture-specific information appears, but the focus is elsewhere.
3 = a substantial part of the document conveys culture-specific knowledge.
4 = the document is primarily about the community's culture and explains practices, meanings, or traditions.
5 = rich, in-depth account of cultural traditions, beliefs, values, or heritage that would teach an outsider a lot about this culture.

Document:
<<<
{doc}
>>>

Answer with a JSON object only, no other text:
{{"domains": [<up to 3 short English labels of cultural domains present, e.g. "food", "religion", "festival", "history", "arts", "social norms", "folklore"; empty list if none>], "score": <integer 0-5>}}"""


def sample_docs(pool_dir, n, seed, max_chars):
    files = sorted(glob.glob(os.path.join(pool_dir, "*.jsonl.gz")))
    rng = random.Random(seed)
    per_file = max(1, n // len(files) + 1)
    out = []
    for f in files:
        # reservoir sample per shard so that every shard contributes equally
        res, k = [], 0
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                k += 1
                if len(res) < per_file:
                    res.append(line)
                else:
                    j = rng.randrange(k)
                    if j < per_file:
                        res[j] = line
        out.extend(json.loads(l) for l in res)
    rng.shuffle(out)
    out = out[:n]
    for d in out:
        d["text_trunc"] = d["text"][:max_chars]
    return out


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        s = int(o["score"])
        if 0 <= s <= 5:
            return {"score": s, "domains": [str(x) for x in o.get("domains", [])][:3]}
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_chars", type=int, default=3000)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    docs = sample_docs(a.pool_dir, a.n, a.seed, a.max_chars)
    lang, community = CULTURE[a.lang]
    msgs = [[{"role": "user", "content": PROMPT.format(lang=lang, community=community,
                                                        doc=d["text_trunc"])}] for d in docs]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=8192,
              gpu_memory_utilization=0.90, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=120)
    outs = llm.chat(msgs, sp, use_tqdm=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n_ok = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for d, o in zip(docs, outs):
            txt = o.outputs[0].text
            r = parse(txt)
            n_ok += r is not None
            fo.write(json.dumps({"id": d["id"], "source": d["source"], "idioms": d["idioms"],
                                 "text": d["text_trunc"], "raw": txt,
                                 **(r or {"score": None, "domains": []})},
                                ensure_ascii=False) + "\n")
    print(f"parsed {n_ok}/{len(docs)}")


if __name__ == "__main__":
    main()
