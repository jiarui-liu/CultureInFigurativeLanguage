#!/bin/bash
# Direct zh-hi / zh-ar / hi-ar same-entity divergence (entity_divergence_cross.py) on the H100
# cluster. Models (Qwen3.5-27B-FP8 gloss LLM, Qwen3-Embedding-0.6B) are pulled straight from the
# Hub by vLLM/sentence-transformers on the compute node, same pattern as notes_remote.slurm.
#   sbatch entity_divergence_cross_remote.slurm
#SBATCH --job-name=div-cross
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v2
#SBATCH --output=logs/div-cross-%j.out
#SBATCH --error=logs/div-cross-%j.err
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
export CULTURE_EMB_MODEL=Qwen/Qwen3-Embedding-0.6B
export LLM_CACHE_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/llm_cache
export VLLM_TP=2

python entity_divergence_cross.py
