#!/bin/bash
# Threshold sensitivity of the en-zh meaning pairing (paper thresholds 0.70 / 0.95).
#   sbatch pairing_sensitivity.slurm
#SBATCH --job-name=pairsens
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pairsens-%j.out
#SBATCH --error=logs/pairsens-%j.err
#SBATCH --partition=cpu
set -uo pipefail
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
export CULTURE_REPO="$CIFL"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
python src/culture/analysis/v2/pairing_threshold_sensitivity.py
