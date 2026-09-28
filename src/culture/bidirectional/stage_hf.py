#!/usr/bin/env python3
"""Download a model straight from the Hugging Face Hub to node-local /scratch and print
its local path (the Hub is much faster than our NFS). Spec formats:
  Qwen/Qwen3.5-2B-Base                                   (model repo)
  dataset:Jerry9999/CultureInFigurativeLanguage:models/qwen3p5-9b-ar-cpt   (subfolder of a dataset repo)
Reuses a complete copy. Usage: python stage_hf.py <spec>
"""
import fcntl
import os
import sys

from huggingface_hub import snapshot_download

spec = sys.argv[1]
root = "/scratch/jiaruil5/hf"
if spec.startswith("dataset:"):
    _, repo, sub = spec.split(":", 2)
    name = sub.strip("/").replace("/", "__")
    local = os.path.join(root, repo.replace("/", "__"))
    path = os.path.join(local, sub)
    kw = dict(repo_type="dataset", allow_patterns=[sub.rstrip("/") + "/*"])
else:
    repo, name = spec, spec.replace("/", "__")
    local = path = os.path.join(root, name)
    kw = {}
marker = path.rstrip("/") + ".complete"
os.makedirs(root, exist_ok=True)
with open(os.path.join(root, "." + name + ".lock"), "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)  # one downloader per node and model; others wait
    import glob
    if os.path.exists(marker) and not glob.glob(os.path.join(path, "*.safetensors")):
        os.remove(marker)  # a previous, interrupted download left no weights
    if not os.path.exists(marker):
        os.makedirs(local, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        snapshot_download(repo, local_dir=local, max_workers=16,
                          token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"), **kw)
        open(marker, "w").close()
print(path)
