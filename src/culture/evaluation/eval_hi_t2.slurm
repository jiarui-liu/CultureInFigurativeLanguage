#!/bin/bash
# Evaluate one Hindi T2 arm (pass-4 symbolism-tag triple) on the Hindi suite.
#   sbatch eval_hi_t2.slurm <untagged|dict|sym>
#SBATCH --job-name=hi-t2-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --output=logs/hi-t2-eval-%x-%j.out
#SBATCH --error=logs/hi-t2-eval-%x-%j.err
set -uo pipefail
ARM="${1:?usage: sbatch eval_hi_t2.slurm <untagged|dict|sym>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
M="$CK/qwen3p5-9b-hi-t2-${ARM}"
[ -f "$M/config.json" ] || { echo "no checkpoint at $M"; exit 1; }
DATA=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/hi
MC=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi_t2/$ARM
mkdir -p "$OUT"

# held-out perplexity: proverb-bearing Hindi (the trained domain) and generic Hindi
for D in hi_proverbs_heldout hi_samanantar_heldout; do
  python -m culture.evaluation.perplexity --model_path "$M" --run_name "t2_$ARM" \
    --data_path "$DATA/$D.jsonl" --text_field text --output_dir "$OUT/ppl_$D" \
    || echo "[warn] ppl $D"
done

python -m culture.evaluation.run_eval --model_path "$M" --run_name "t2_$ARM" \
  --tasks mabl,global_piqa,milu,jsonl:idiomatlas_mc_hi_seen,jsonl:idiomatlas_mc_hi_unseen,jsonl:symbolism_v2_hi_letter,jsonl:symbolism_hi,jsonl:global_piqa_hi,jsonl:global_piqa_hi_cultural,jsonl:global_piqa_hi_parallel4,jsonl:parambench_hi_culture,jsonl:parambench_hi_other \
  --mabl_path "$DATA/mabl_hi.csv" --global_piqa_path "$DATA/global_piqa_hi.tsv" \
  --milu_path "$DATA/milu_hi_test.jsonl" --milu_fewshot_path "$DATA/milu_hi_val.jsonl" \
  --milu_num_fewshot 5 --mc_jsonl_dir "$MC" \
  --num_fewshot 0 --batch_size 8 --output_dir "$OUT"
echo "T2_EVAL_DONE hi $ARM"
