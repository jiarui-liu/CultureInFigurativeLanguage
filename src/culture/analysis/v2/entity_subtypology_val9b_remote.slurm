#!/bin/bash
# Rerun of step 3 of entity_subtypology_remote.slurm. In job 399162 the Qwen3.5-9B second
# annotator died with
#   httpcore.RemoteProtocolError: peer closed connection without sending complete message body
#     (received 4615986056 bytes, expected 5335161512)
# i.e. a truncated HF weight download, not a modelling problem. Pre-fetch the weights with
# retries before touching vLLM, then run only the Qwen3.5-9B annotator; the script merges into
# the existing entity_subtypology_validation.json, so the aya-expanse-8b result is preserved.
#   sbatch entity_subtypology_val9b_remote.slurm
#SBATCH --job-name=ent-subtyp-9b
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v2
#SBATCH --output=logs/ent-subtyp-9b-%j.out
#SBATCH --error=logs/ent-subtyp-9b-%j.err
set -euo pipefail
mkdir -p logs

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL

REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH="${REPO}/src:${REPO}/src/culture/analysis/v2"
VENV="/storage/home/jiaruiliu/local/git-repos/monitorability/.venv"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME=/lustre-storage/fsx_it_0/users/jiaruiliu/hf_home
export HF_HUB_DISABLE_XET=1
export TMPDIR=/lustre-storage/fsx_it_0/users/jiaruiliu/tmp

export CULTURE_REPO="$REPO"
export CULTURE_LLM_PRIMARY=Qwen/Qwen3.5-27B-FP8
export CULTURE_LLM_SECOND=Qwen/Qwen3.5-9B
export CULTURE_LLM_THIRD=CohereLabs/aya-expanse-8b
export CULTURE_EVAL_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval
export LLM_CACHE_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/llm_cache
export VLLM_TP=2

echo "=== 1/2 pre-fetch Qwen3.5-9B weights (resumes the truncated shard) ==="
for attempt in 1 2 3 4 5; do
  echo "--- download attempt ${attempt} ---"
  if python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-9B", max_workers=4)
print("snapshot complete:", p)
PY
  then
    echo "--- weights present ---"
    break
  fi
  if [ "${attempt}" = "5" ]; then
    echo "download failed after 5 attempts" >&2
    exit 1
  fi
  sleep $((attempt * 30))
done

echo "=== 2/2 validation: second annotator Qwen3.5-9B ==="
python entity_subtypology_validate.py --only Qwen3.5-9B

echo "=== done ==="
