#!/bin/bash
# One-time Arabic eval prereqs: download benchmarks + build BPB probes into the
# repo-root data/eval/ar (== lustre culture-pretraining-data/eval/ar, which
# eval_ar.slurm reads). Networked node, proxy unset. Runbook §9.6.
#   sbatch eval_prereqs_ar.slurm
#SBATCH --job-name=ar-eval-prep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu,all
#SBATCH --output=logs/ar/eval-prep-%j.out
#SBATCH --error=logs/ar/eval-prep-%j.out
set -uo pipefail

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_DISABLE_XET=1
export HF_HOME="/lustre-storage/fsx_0/user/jiaruiliu/hfcache_ar_eval"

EVAL_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/ar   # == repo-root data/eval/ar
TAGGED="/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/ar-amthal-cpt/data/fineweb2_par/tagged_*.json.gz"
mkdir -p "$EVAL_DIR/raw"

echo "=== 1/3 download benchmarks -> $EVAL_DIR/raw ==="
DATA_DIR="$EVAL_DIR/raw" bash src/culture/evaluation/download_ar.sh || { echo "FATAL: download_ar.sh failed"; exit 1; }

echo "=== 2/3 build BPB probes (fineweb2 heldout + wiki, decontaminated) -> $EVAL_DIR ==="
python src/culture/evaluation/build_ar_probes.py \
    --out_dir "$EVAL_DIR" \
    --exclude_urls "$TAGGED" || { echo "FATAL: build_ar_probes failed"; exit 1; }

echo "=== 3/3 verify loaders (must print expected counts) ==="
PYTHONPATH=src python -c "
from culture.evaluation.tasks_ar import LOADERS_AR as L
for k,f in L.items():
    try: print(k, len(f(ar_data_dir='$EVAL_DIR/raw').examples))
    except Exception as e: print(k, 'ERROR', e)
" || { echo "FATAL: loader verification failed"; exit 1; }

# probes present?
for p in ar_fineweb2_heldout ar_wiki_heldout; do
  [[ -s "$EVAL_DIR/$p.jsonl" ]] || { echo "FATAL: missing probe $p.jsonl"; exit 1; }
done

touch "$CIFL/src/culture/training/continued_pretraining/logs/ar/EVAL_STARTED"
echo "EVAL_PREREQS_DONE ok -> $EVAL_DIR"
