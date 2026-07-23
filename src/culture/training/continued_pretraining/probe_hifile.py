import os
from huggingface_hub import HfFileSystem
fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
p = "datasets/ai4bharat/IndicCorpV2/data/hi.txt"
print("exists:", fs.exists(p), "size(GB):", round(fs.info(p)["size"]/1e9, 2))
n = 0
with fs.open(p, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if len(line) >= 150:
            print("LINE chars=%d :: %s" % (len(line), line[:90]))
            n += 1
        if n >= 3:
            break
