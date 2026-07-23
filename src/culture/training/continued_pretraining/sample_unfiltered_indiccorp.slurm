#!/bin/bash
#SBATCH --job-name=hi-unf-indiccorp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/unf-sample-indiccorp-%j.out
#SBATCH --error=logs/unf-sample-indiccorp-%j.err
#SBATCH --partition=cpu
set -uo pipefail
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
source ~/.bashrc 2>/dev/null; export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HOME="/lustre-storage/fsx_0/user/jiaruiliu/hfcache_unf"
OUT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/train_hi_unfiltered
python -m culture.training.continued_pretraining.build_unfiltered_hi \
  --hf_repo ai4bharat/IndicCorpV2 --hf_text_files data/hi-1.txt,data/hi-2.txt,data/hi-3.txt \
  --source_label indiccorp-hi --token_budget 110000000 --out_dir "$OUT" \
  --tokenizer /lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
echo "DONE indiccorp-hi"
