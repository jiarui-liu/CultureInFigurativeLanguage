#!/bin/bash
# Sample the UNFILTERED Arabic baseline corpus: the SAME FineWeb-2 arb_Arab source
# as the augmented run, with NO idiom filtering and NO tagging, to a matched Qwen
# token budget. Mirrors sample_unfiltered_zh.slurm.
#   sbatch sample_unfiltered_ar.slurm <token_budget>
# <token_budget> = the exact Qwen-token count of the augmented ar_amthal corpus
# (measured after 9.3/9.4). Streams the corpus on a cpu node with HF egress.
#SBATCH --job-name=ar-unf-sample
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ar/unf-sample-%j.out
#SBATCH --error=logs/ar/unf-sample-%j.out
#SBATCH --partition=cpu
set -uo pipefail
# arg: $1 = token_budget (required)
BUDGET="${1:?usage: sbatch sample_unfiltered_ar.slurm <token_budget>}"
SEED="${2:-42}"

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_XET=1
export HF_HOME="/lustre-storage/fsx_0/user/jiaruiliu/hfcache_ar"

OUT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/train_ar_unfiltered
python -m culture.training.continued_pretraining.build_unfiltered_hi \
  --path HuggingFaceFW/fineweb-2 --name arb_Arab --split train \
  --source_label fineweb2-ar --token_budget "$BUDGET" --seed "$SEED" \
  --out_dir "$OUT" \
  --tokenizer /lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
echo "SAMPLE_DONE budget=$BUDGET out=$OUT"
