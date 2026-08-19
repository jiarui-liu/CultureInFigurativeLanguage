import os
from huggingface_hub import HfApi
api=HfApi(token=os.environ.get("HF_TOKEN"))
repo="opencsg/Fineweb-Edu-Chinese-V2.1"
files=api.list_repo_files(repo, repo_type="dataset")
print("total files:", len(files))
# show directory structure (top 2 levels)
import collections
dirs=collections.Counter("/".join(f.split("/")[:2]) for f in files if "/" in f)
for d,c in sorted(dirs.items())[:25]: print(f"  {d}/  ({c} files)")
print("sample files:", [f for f in files if f.endswith((".parquet",".jsonl",".json",".txt",".gz"))][:6])
# try to get configs
try:
    from datasets import get_dataset_config_names
    print("configs:", get_dataset_config_names(repo)[:20])
except Exception as e:
    print("configs err:", type(e).__name__, str(e)[:100])
