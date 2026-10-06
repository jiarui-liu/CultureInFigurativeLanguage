#!/bin/bash
# Larger second-annotator pass for the culture-layer taxonomy. The primary (Qwen3.5-27B) labels
# are already cached, so only aya-expanse-8b loads here. With --agree_n 550 each of the eleven
# sources gets ~50 relabelled items instead of 10, which is what settles the per-source
# reliability -- in particular whether the cultural-notes row really sits with the benchmarks.
#   sbatch taxonomy_agreement_remote.slurm
#SBATCH --job-name=subtyp-gemma
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v2
#SBATCH --output=logs/subtyp-gemma-%j.out
#SBATCH --error=logs/subtyp-gemma-%j.err
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
export CULTURE_LLM_THIRD=CohereLabs/aya-expanse-8b
export CULTURE_EVAL_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval
export LLM_CACHE_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/llm_cache
export VLLM_TP=1
export VLLM_GPU_MEM=0.92
export VLLM_MAX_LEN=4096

export CULTURE_LLM_FOURTH=google/gemma-4-26B-A4B-it

echo "=== subtypology validation, 4th annotator (gemma-4-26B-A4B-it) ==="
python entity_subtypology_validate.py --only gemma

echo "=== done ==="
