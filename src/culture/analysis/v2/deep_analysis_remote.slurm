#!/bin/bash
# Deep Section-4 analyses that need an LLM: the culture-layer taxonomy (idioms vs culture
# benchmarks) and cross-language entity valence. Models are pulled from the Hub by vLLM on the
# compute node, same pattern as entity_divergence_cross_remote.slurm.
#   sbatch deep_analysis_remote.slurm
#SBATCH --job-name=deep-analysis
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v2
#SBATCH --output=logs/deep-analysis-%j.out
#SBATCH --error=logs/deep-analysis-%j.err
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
export CULTURE_EVAL_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval
export LLM_CACHE_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/llm_cache
export VLLM_TP=2

# vLLM cannot initialise a second model inside one process (job 397644 died that way after the
# 27B engine shut down), so each pass runs as its own invocation. The on-disk LLM cache makes
# any pass that already completed essentially free on a rerun.
echo "=== [1/4] taxonomy, primary (Qwen3.5-27B) ==="
python culture_layer_taxonomy.py --per_source 400 --skip_second

echo "=== [2/4] taxonomy, second annotator (aya-expanse-8b) ==="
python culture_layer_taxonomy.py --per_source 400

echo "=== [3/4] valence, primary (Qwen3.5-27B) ==="
python entity_valence.py --skip_second

echo "=== [4/4] valence, second annotator (aya-expanse-8b) ==="
python entity_valence.py

echo "=== done ==="
