#!/usr/bin/env python3
"""Answerability check for the symbolism probe: strong instruction-tuned models answer the
items as a letter-choice question WITHOUT the evidence idioms. Reports accuracy and lure rate.
  python -m culture.bidirectional.probe_ceiling --model <path> --name <tag>"""
import argparse
import json
import re

from culture.bidirectional.llm_api import complete_many

B = "/data/group_data/r3lit_culture_pretrain/culture/bidir"
Q = """{context}
A.{o0}
B.{o1}
C.{o2}
D.{o3}
Answer with the letter of the best option only."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", default="")
    a = ap.parse_args()
    out = {}
    for L in ["zh", "hi", "ar"]:
        it = [json.loads(l) for l in open(f"{B}/eval_data/mc/symbolism{a.version}_{L}.jsonl")]
        res = complete_many([Q.format(context=x["context"], o0=x["options"][0], o1=x["options"][1],
                                      o2=x["options"][2], o3=x["options"][3]) for x in it],
                            provider="vllm", model=a.model, max_tokens=8, tag=f"probe_ceiling_{a.name}{a.version}_{L}")
        acc = lure = 0
        for x, r in zip(it, res):
            m = re.search(r"[ABCD]", r or "")
            k = "ABCD".index(m.group(0)) if m else -1
            acc += k == x["gold"]
            lure += k == x["meta"]["lure"]
        out[L] = {"n": len(it), "acc": round(acc / len(it), 3), "lure": round(lure / len(it), 3)}
        print(a.name, L, out[L], flush=True)
    json.dump(out, open(f"{B}/probe/ceiling_{a.name}{a.version}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
