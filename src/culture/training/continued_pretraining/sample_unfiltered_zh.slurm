#!/bin/bash
#SBATCH --job-name=zh-unf-sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/unf-zh-%x-%j.out
#SBATCH --error=logs/unf-zh-%x-%j.err
#SBATCH --partition=cpu
set -uo pipefail
# args: $1=label $2=token_budget $3=seed $4=data_files_globs(comma)
LABEL=$1; BUDGET=$2; SEED=$3; DF=$4
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
source ~/.bashrc 2>/dev/null; export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HOME="/lustre-storage/fsx_0/user/jiaruiliu/hfcache_unf"
OUT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/train_zh_unfiltered
python -m culture.training.continued_pretraining.build_unfiltered_hi \
  --path opencsg/Fineweb-Edu-Chinese-V2.1 --data_files "$DF" --split train \
  --source_label "$LABEL" --token_budget "$BUDGET" --seed "$SEED" --out_dir "$OUT" \
  --tokenizer /lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
echo "DONE $LABEL"
