#!/bin/bash
# T1 corpus build: the three token-matched Chinese corpora (random / culture-restricted /
# culture-unrestricted).  Needs one GPU for the classifier pass over the idiom-bearing pool.
#   sbatch build_zh_t1.slurm
#SBATCH --job-name=t1-build
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v3
#SBATCH --output=logs/t1-build-%j.out
#SBATCH --error=logs/t1-build-%j.err
set -uo pipefail
mkdir -p logs
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL

REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH="${REPO}/src:${REPO}/src/culture/analysis/v2:${REPO}/src/culture/analysis/v3"
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:${PATH}"
export HF_HOME=/lustre-storage/fsx_it_0/users/jiaruiliu/hf_home
export TMPDIR=/lustre-storage/fsx_it_0/users/jiaruiliu/tmp
export CULTURE_REPO="$REPO"
export EMB_MODEL=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3-Embedding-0.6B

python -u build_zh_t1.py --target_tokens 1.1e9 --pool 1200000
echo "=== done ==="
