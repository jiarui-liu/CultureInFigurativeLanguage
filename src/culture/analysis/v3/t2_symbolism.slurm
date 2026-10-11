#!/bin/bash
# T2 step 1 (generalised): write what each entity symbolizes, from the knowledge base only.
#   sbatch t2_symbolism.slurm hi
#   sbatch t2_symbolism.slurm ar --include_probe
#SBATCH --job-name=t2-sym
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v3
#SBATCH --output=logs/t2-sym-%j.out
#SBATCH --error=logs/t2-sym-%j.err
set -uo pipefail
mkdir -p logs
LANG_ARG="${1:?usage: sbatch t2_symbolism.slurm <ar|hi> [--include_probe]}"
shift || true
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL
REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH="${REPO}/src:${REPO}/src/culture/analysis/v2:${REPO}/src/culture/analysis/v3"
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability/.venv/bin:${PATH}"
export HF_HOME=/lustre-storage/fsx_it_0/users/jiaruiliu/hf_home
export HF_HUB_DISABLE_XET=1
export TMPDIR=/lustre-storage/fsx_it_0/users/jiaruiliu/tmp
export CULTURE_REPO="$REPO"
export CULTURE_LLM_PRIMARY=Qwen/Qwen3.5-27B-FP8
export LLM_CACHE_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/llm_cache
export VLLM_TP=2 VLLM_MAX_LEN=8192 VLLM_MAX_SEQS=64

python -u t2_symbolism_summaries.py --lang "$LANG_ARG" --min_idioms 4 "$@"
echo "=== done ==="
