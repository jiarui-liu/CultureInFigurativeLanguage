#!/bin/bash
# Score raw web files for the full-scale (9B) Culture corpus on the H100 cluster.
# One part per array task; run count is controlled by --array and must match NPARTS.
#   sbatch --array=0-15 --export=ALL,LANG_=zh,REPO_ID=opencsg/Fineweb-Edu-Chinese-V2.1,PATTERN='4_5/*.parquet',NPARTS=16,MIN_SCORE=0.5,RG_SPLIT=1 stream_select_remote.slurm
#SBATCH --job-name=stream-select
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/bidirectional
#SBATCH --output=logs/stream-%x-%A_%a.out
#SBATCH --error=logs/stream-%x-%A_%a.err
set -euo pipefail
mkdir -p logs

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL

REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH="${REPO}/src"
VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME=/lustre-storage/fsx_2/user/jiaruiliu/hf_home

OUT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir
mkdir -p "$OUT/full/${LANG_}"

python -m culture.bidirectional.stream_select_culture score --lang "$LANG_" \
  --repo "$REPO_ID" --pattern "$PATTERN" \
  --clf "$OUT/clf/${LANG_}.pkl" --min_score "${MIN_SCORE:-1.5}" \
  --out_dir "$OUT/full/${LANG_}" \
  --part "${SLURM_ARRAY_TASK_ID:-0}" --nparts "${NPARTS:-8}" \
  --workers 11 --rg_split "${RG_SPLIT:-1}" \
  --tmp_dir "/tmp/stream_${SLURM_ARRAY_TASK_ID:-0}"
