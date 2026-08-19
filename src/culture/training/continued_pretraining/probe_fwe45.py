import os
from huggingface_hub import HfApi
from datasets import load_dataset
api=HfApi(token=os.environ.get("HF_TOKEN"))
files=[f for f in api.list_repo_files("opencsg/Fineweb-Edu-Chinese-V2.1", repo_type="dataset") if f.endswith(".parquet")]
tiers=sorted(set(f.split("/")[0] for f in files if "/" in f))
print("tiers:", tiers)
print("4_5 parquet count:", sum(1 for f in files if f.startswith("4_5/")))
ds=load_dataset("opencsg/Fineweb-Edu-Chinese-V2.1", data_files="4_5/*.parquet", split="train", streaming=True, token=os.environ.get("HF_TOKEN"))
r=next(iter(ds)); print("keys:", list(r.keys())); print("text[:80]:", str(r.get("text",""))[:80])
