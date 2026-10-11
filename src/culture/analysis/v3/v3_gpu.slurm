#!/bin/bash
# Pass-3 GPU analyses: per-item culture-layer labels, corpus<->benchmark affinity and the
# culture-score profile of every training corpus, and the neighbourhood-generalisation test.
#   sbatch v3_gpu.slurm
#SBATCH --job-name=v3-gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v3
#SBATCH --output=logs/v3-gpu-%j.out
#SBATCH --error=logs/v3-gpu-%j.err
set -uo pipefail
mkdir -p logs

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL

REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH="${REPO}/src:${REPO}/src/culture/analysis/v2:${REPO}/src/culture/analysis/v3"
VENV="/storage/home/jiaruiliu/local/git-repos/monitorability/.venv"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME=/lustre-storage/fsx_it_0/users/jiaruiliu/hf_home
export HF_HUB_DISABLE_XET=1
export TMPDIR=/lustre-storage/fsx_it_0/users/jiaruiliu/tmp

export CULTURE_REPO="$REPO"
export CULTURE_LLM_PRIMARY=Qwen/Qwen3.5-27B-FP8
export CULTURE_LLM_THIRD=CohereLabs/aya-expanse-8b
export CULTURE_LLM_FOURTH=google/gemma-4-26B-A4B-it
export EMB_MODEL=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3-Embedding-0.6B
export CULTURE_EVAL_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval
export LLM_CACHE_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/llm_cache
export CULTURE_STATS_V3="${REPO}/docs/paper_stats/v3"

echo "=== [1/4] corpus culture-score profile + benchmark affinity (embedding only) ==="
CUDA_VISIBLE_DEVICES=0 python corpus_affinity.py || echo "[fail] corpus_affinity"

echo "=== [2/4] neighbourhood generalisation (embedding only) ==="
CUDA_VISIBLE_DEVICES=0 python generalization.py || echo "[fail] generalization"

echo "=== [3/4] per-item culture-layer labels, primary (Qwen3.5-27B-FP8, TP=2) ==="
VLLM_TP=2 VLLM_MAX_LEN=4096 VLLM_MAX_SEQS=128 python item_taxonomy.py --cap 4000 \
  --model primary --out item_taxonomy.json || echo "[fail] item_taxonomy primary"

echo "=== [4/4] per-item labels, second annotator (gemma-4-26B-A4B-it, TP=1) on a subsample ==="
VLLM_TP=1 VLLM_GPU_MEM=0.92 VLLM_MAX_LEN=4096 python item_taxonomy.py --cap 4000 --subsample 150 \
  --model fourth --out item_taxonomy_gemma.json || echo "[fail] item_taxonomy gemma"

echo "=== done ==="
