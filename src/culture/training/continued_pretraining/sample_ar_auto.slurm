#!/bin/bash
# DAG step 3: build the baseline corpus matched to the augmented token budget,
# then decompress to train_ar_unfiltered_prepared. Reads the budget written by
# prepare_ar.slurm. Depends (afterok) on prepare_ar.
#   sbatch --dependency=afterok:<prepare_jobid> sample_ar_auto.slurm
#SBATCH --job-name=ar-unf-auto
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ar/unf-auto-%j.out
#SBATCH --error=logs/ar/unf-auto-%j.out
set -uo pipefail

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_XET=1
export HF_HOME="/lustre-storage/fsx_0/user/jiaruiliu/hfcache_ar"

DATA_ROOT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data
BUDGET_FILE="$DATA_ROOT/train_ar.token_budget.txt"   # sibling of train_ar (NOT inside it)
OUT="$DATA_ROOT/train_ar_unfiltered"
PREP="$DATA_ROOT/train_ar_unfiltered_prepared"
MODEL=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B

if [[ ! -f "$BUDGET_FILE" ]]; then echo "FATAL: no token budget at $BUDGET_FILE"; exit 1; fi
BUDGET=$(cat "$BUDGET_FILE"); echo "matched token budget = $BUDGET"
[[ "$BUDGET" =~ ^[0-9]+$ && "$BUDGET" -gt 0 ]] || { echo "FATAL: bad budget '$BUDGET'"; exit 1; }

# --- sample the same FineWeb-2 arb_Arab, no filter/no tag, to that budget ------
python -m culture.training.continued_pretraining.build_unfiltered_hi \
  --path HuggingFaceFW/fineweb-2 --name arb_Arab --split train \
  --source_label fineweb2-ar --token_budget "$BUDGET" --seed 42 \
  --out_dir "$OUT" \
  --tokenizer "$MODEL"

# --- decompress train_*.jsonl.gz -> train_*.jsonl (matches how
#     train_{hi,zh}_unfiltered_prepared were produced) --------------------------
mkdir -p "$PREP"; rm -f "$PREP"/train_*.jsonl
n=0
for gz in "$OUT"/train_*.jsonl.gz; do
  [[ -e "$gz" ]] || { echo "FATAL: no shards in $OUT"; exit 1; }
  base=$(basename "$gz" .gz)
  zcat "$gz" > "$PREP/$base"
  n=$((n+1))
done
echo "decompressed $n shards -> $PREP"
python src/culture/training/continued_pretraining/prepare_data.py --verify_only --out_dir "$PREP" || true
echo "SAMPLE_AUTO_DONE budget=$BUDGET out=$PREP"
