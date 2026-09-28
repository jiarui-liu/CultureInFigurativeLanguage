#!/usr/bin/env python3
"""Generate cultural notes for culture-selected documents (the Culture+notes arm).

The notes are the culture-side analogue of the idiom meaning tags: a short block
appended to a document that makes its cultural content explicit. For each
document, an instruction-tuned LLM lists 2-5 cultural elements that the document
mentions and explains what each means or signifies in the culture, in the
document's language. Idioms, proverbs and sayings are forbidden in the prompt, and
any note line that still contains an IdiomAtlas idiom is removed afterwards with
the same matcher used to build the pool, so the arm stays idiom-free.

Output: one jsonl.gz per input part, {"id", "notes": [...], "text": <doc + block>}.

Usage:
  python -m culture.bidirectional.culture_notes --lang ar --input $B/arms/ar/culture_docs.jsonl.gz \
      --model <instruct model> --tp 2 --part 0 --nparts 4 --out_dir $B/arms/ar/notes
"""
import argparse
import gzip
import json
import os
import re

HEADER = {
    "zh": "【文化注释】",
    "hi": "सांस्कृतिक टिप्पणियाँ:",
    "ar": "ملاحظات ثقافية حول النص:",
}
LANG = {"zh": "Chinese", "hi": "Hindi", "ar": "Arabic"}
COMMUNITY = {
    "zh": "Chinese culture",
    "hi": "Indian / Hindi-speaking culture",
    "ar": "Arab culture",
}

PROMPT = """Read the {lang} document below. Identify 2 to 5 elements of {community} that the document mentions or relies on (for example a custom, ritual, festival, dish, religious practice, social norm, value, historical figure or event, art form, garment, institution, or folk belief).

For each element, write one line in {lang} of the form
- <element>: <one or two sentences explaining what it is and what it means or signifies in {community}>

Rules:
- Only explain elements that actually appear in the document; add background knowledge that a reader from another culture would need.
- Do NOT quote, explain, or mention any idiom, proverb, or saying.
- Be factual and concise (at most 120 words in total). Write in {lang} only.
- If the document contains no cultural element, output exactly: NONE

Document:
<<<
{doc}
>>>"""


def parse_notes(txt):
    if txt.strip().upper().startswith("NONE"):
        return []
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if re.match(r"^[-•*]\s*\S", line):
            out.append("- " + re.sub(r"^[-•*]\s*", "", line))
    return out[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_chars", type=int, default=3000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"notes_{a.part:03d}.jsonl.gz")
    if os.path.exists(out):
        print("exists", out)
        return

    docs = []
    with gzip.open(a.input, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % a.nparts == a.part:
                docs.append(json.loads(line))
    from culture.bidirectional.build_pool import make_matcher
    matcher = make_matcher(a.lang)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=8192,
              gpu_memory_utilization=0.90, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=400, seed=0)
    msgs = [[{"role": "user", "content": PROMPT.format(lang=LANG[a.lang], community=COMMUNITY[a.lang],
                                                        doc=d["text"][:a.max_chars])}] for d in docs]
    res = llm.chat(msgs, sp, use_tqdm=True)
    st = {"docs": len(docs), "with_notes": 0, "lines": 0, "lines_dropped_idiom": 0}
    tmp = out + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fo:
        for d, r in zip(docs, res):
            notes = parse_notes(r.outputs[0].text)
            kept = []
            for n in notes:
                if matcher.match(n):
                    st["lines_dropped_idiom"] += 1
                else:
                    kept.append(n)
            st["lines"] += len(kept)
            st["with_notes"] += bool(kept)
            text = d["text"] + ("\n\n" + HEADER[a.lang] + "\n" + "\n".join(kept) if kept else "")
            fo.write(json.dumps({"id": d["id"], "notes": kept, "text": text}, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    json.dump(st, open(out.replace(".jsonl.gz", ".stats.json"), "w"))
    print(st)


if __name__ == "__main__":
    main()
