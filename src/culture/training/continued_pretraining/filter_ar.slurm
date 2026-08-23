#!/bin/bash
# Build the AUGMENTED Arabic CPT corpus (ar_amthal): stream FineWeb-2 arb_Arab,
# apply the quality gates + idiom matcher, window over-long docs, and append the
# Arabic knowledge block. Writes tagged_*.json.gz. Runbook: plans/arabic_pipeline_plan.md §9.3.
#
#   sbatch filter_ar.slurm
#
# Streams the corpus (no bulk download) on a cpu node that has HF egress once the
# sandbox proxy is unset -- same mechanism as sample_unfiltered_zh.slurm.
#SBATCH --job-name=ar-filter-tag
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ar/filter-%j.out
#SBATCH --error=logs/ar/filter-%j.out
set -uo pipefail

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
# The compute node reaches HF only with the sandbox proxy unset.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_XET=1
export HF_HOME="/lustre-storage/fsx_0/user/jiaruiliu/hfcache_ar"

DATA_ROOT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data
KB="$DATA_ROOT/ar-kb/data/idioms/ar/idioms_merged_llm_formatted.jsonl"
OUT="$DATA_ROOT/ar-amthal-cpt/data/fineweb2"

python src/culture/training/mC4/filter_and_tag_ar.py filter \
    --idioms "$KB" \
    --dataset HuggingFaceFW/fineweb-2 --config arb_Arab --split train \
    --out "$OUT" \
    --max_docs_per_idiom 10000

echo "FILTER_DONE rc=$? out=$OUT"
