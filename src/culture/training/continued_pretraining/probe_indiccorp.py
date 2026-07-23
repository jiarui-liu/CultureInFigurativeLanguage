import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
files = [f for f in api.list_repo_files("ai4bharat/IndicCorpV2", repo_type="dataset") if f.endswith(".txt")]
for f in sorted(files): print(f)
