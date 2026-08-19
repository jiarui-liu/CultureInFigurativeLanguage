"""Build two Chinese Dim-1 perplexity probes:
  zh_wiki_heldout.jsonl     - clean independent Chinese (zh Wikipedia), not in CPT corpus
  zh_chengyu_heldout.jsonl  - in-domain slice from the zh CPT corpus (CONTAMINATED)
"""
import glob, json, os, random
OUT = "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/zh"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(42)

# 1) in-domain (contaminated) from train_zh
shards = sorted(glob.glob("/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/train_zh/*.jsonl"))
texts = []
for line in open(shards[-1], encoding="utf-8"):
    r = json.loads(line); t = r.get("text", "")
    if isinstance(t, str) and 80 <= len(t) <= 8000:
        texts.append(t)
rng.shuffle(texts); sel = texts[:2000]
with open(f"{OUT}/zh_chengyu_heldout.jsonl", "w", encoding="utf-8") as f:
    for t in sel: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
print(f"zh_chengyu_heldout (in-domain, CONTAMINATED): {len(sel)} docs from {shards[-1].split('/')[-1]}")

# 2) clean independent from zh Wikipedia
from datasets import load_dataset
got = []
for cfg in ["20231101.zh"]:
    try:
        ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True, token=os.environ.get("HF_TOKEN"))
        ds = ds.shuffle(seed=42, buffer_size=20000)
        for r in ds:
            t = (r.get("text") or "").strip()
            if 80 <= len(t) <= 8000:
                got.append(t)
            if len(got) >= 2000: break
        break
    except Exception as e:
        print("wiki cfg", cfg, "failed:", type(e).__name__, str(e)[:120])
with open(f"{OUT}/zh_wiki_heldout.jsonl", "w", encoding="utf-8") as f:
    for t in got: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
print(f"zh_wiki_heldout (clean, independent): {len(got)} docs")
