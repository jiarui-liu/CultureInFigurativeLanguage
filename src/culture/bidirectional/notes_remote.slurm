#!/bin/bash
# Cultural notes for the culture-ranked documents of one language, one part per array task,
# on the H100 cluster. Model is pulled straight from the Hub by vLLM on the compute node
# (which has real internet access, unlike the login-node huggingface-cli downloads that
# failed with xet/disk errors on 2026-09-29).
#   sbatch --array=0-7 --export=ALL,LANG_=ar,NPARTS=8 notes_remote.slurm
#SBATCH --job-name=notes-remote
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/bidirectional
#SBATCH --output=logs/notes-%x-%A_%a.out
#SBATCH --error=logs/notes-%x-%A_%a.err
set -euo pipefail
mkdir -p logs

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL

REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH="${REPO}/src"
VENV="/storage/home/jiaruiliu/local/git-repos/monitorability/.venv"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME=/lustre-storage/fsx_it_0/users/jiaruiliu/hf_home
export HF_HUB_DISABLE_XET=1
export TMPDIR=/lustre-storage/fsx_it_0/users/jiaruiliu/tmp

OUT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir

python -m culture.bidirectional.culture_notes --lang "$LANG_" \
  --input "$OUT/full/${LANG_}/culture_docs_ranked.jsonl.gz" \
  --model "${NOTES_MODEL:-google/gemma-4-26B-A4B-it}" --tp "${NOTES_TP:-2}" \
  --part "${SLURM_ARRAY_TASK_ID:-0}" --nparts "${NPARTS:-8}" \
  --out_dir "$OUT/full/${LANG_}/notes"
