#!/bin/bash
#SBATCH --job-name=hi-unf-sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/unf-sample-%x-%j.out
#SBATCH --error=logs/unf-sample-%x-%j.err
#SBATCH --partition=cpu
set -uo pipefail
# args: $1=path $2=name(or -) $3=split $4=source_label $5=token_budget
PATH_=$1; NAME=$2; SPLIT=$3; LABEL=$4; BUDGET=$5
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
source ~/.bashrc 2>/dev/null; export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HOME="/lustre-storage/fsx_0/user/jiaruiliu/hfcache_unf"
OUT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/train_hi_unfiltered
NAME_ARG=(); [ "$NAME" != "-" ] && NAME_ARG=(--name "$NAME")
python -m culture.training.continued_pretraining.build_unfiltered_hi \
  --path "$PATH_" "${NAME_ARG[@]}" --split "$SPLIT" --source_label "$LABEL" \
  --token_budget "$BUDGET" --out_dir "$OUT" \
  --tokenizer /lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
echo "DONE $LABEL"
