#!/bin/bash
# Hindi eval for the R4 exact-budget Random-CPT checkpoint (qwen3p5-9b-hi-cpt-unfiltered-matched).
#
# This checkpoint finished training but was never evaluated -- it is the only arm in the
# whole 9B matrix with zero eval records. R4 exists because plain hi Random-CPT processed
# 4.09B tokens vs 4.40B for Idiom-CPT (plan §3.4); without this run the Hindi forward/reverse
# comparisons rest on a 7%-mismatched control.
#
# Same task list as eval_hi.slurm but with symbolism v2 (the current task list, §3.1b).
#   sbatch eval_hi_matched.slurm
#SBATCH --job-name=hi-matched
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval-hi-matched-%j.out
#SBATCH --error=logs/eval-hi-matched-%j.err
#SBATCH --partition=h100
set -uo pipefail
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

RUN=unfiltered-matched
M=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt-unfiltered-matched
[ -f "$M/config.json" ] || { echo "no checkpoint at $M"; exit 1; }

DATA=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/hi
MC=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi/$RUN
mkdir -p "$OUT"

python -m culture.evaluation.run_eval \
  --model_path "$M" --run_name "$RUN" \
  --tasks mabl,global_piqa,milu,jsonl:idiomatlas_mc_hi_seen,jsonl:idiomatlas_mc_hi_unseen,jsonl:symbolism_v2_hi_letter \
  --mabl_path "$DATA/mabl_hi.csv" --global_piqa_path "$DATA/global_piqa_hi.tsv" \
  --milu_path "$DATA/milu_hi_test.jsonl" --milu_fewshot_path "$DATA/milu_hi_val.jsonl" --milu_num_fewshot 5 \
  --mc_jsonl_dir "$MC" \
  --num_fewshot 0 \
  --batch_size 8 \
  --output_dir "$OUT"
echo "DONE $RUN -> $OUT"
