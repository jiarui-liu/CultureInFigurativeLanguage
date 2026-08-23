#!/bin/bash
# PARALLEL augmented-corpus filter: one array task per FineWeb-2 arb_Arab parquet
# file (25 files). Replaces the single-threaded filter_ar.slurm to cut wall-clock
# ~Nx (the matcher/quality gates are CPU-bound single-thread). Each task streams
# ONE parquet file, writes tagged_p<taskid>_NNNNN.json.gz + filter_report_p<id>.json
# into a shared out dir. Per-idiom cap is divided by N so the total stays ~10000.
#   sbatch filter_ar_parallel.slurm
#SBATCH --job-name=ar-filter-par
#SBATCH --array=0-24%16
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --partition=cpu,all
#SBATCH --output=logs/ar/filterpar-%A_%a.out
#SBATCH --error=logs/ar/filterpar-%A_%a.out
set -uo pipefail

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_XET=1
export HF_HOME="/lustre-storage/fsx_0/user/jiaruiliu/hfcache_ar_${SLURM_ARRAY_TASK_ID}"

DATA_ROOT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data
KB="$DATA_ROOT/ar-kb/data/idioms/ar/idioms_merged_llm_formatted.jsonl"
OUT="$DATA_ROOT/ar-amthal-cpt/data/fineweb2_par"       # shared across tasks
FILELIST="$DATA_ROOT/ar_parquet_files.txt"

# pick this task's parquet file (1 file per task)
LINE=$(( SLURM_ARRAY_TASK_ID + 1 ))
PARQUET=$(sed -n "${LINE}p" "$FILELIST")
if [[ -z "$PARQUET" ]]; then echo "no parquet for task $SLURM_ARRAY_TASK_ID (>${FILELIST} len)"; exit 0; fi
echo "task $SLURM_ARRAY_TASK_ID -> $PARQUET"

python src/culture/training/mC4/filter_and_tag_ar.py filter \
    --idioms "$KB" \
    --dataset HuggingFaceFW/fineweb-2 --split train \
    --data_files "$PARQUET" \
    --shard_prefix "p${SLURM_ARRAY_TASK_ID}_" \
    --out "$OUT" \
    --max_docs_per_idiom 400
echo "FILTER_PAR_DONE task=$SLURM_ARRAY_TASK_ID rc=$? out=$OUT"
