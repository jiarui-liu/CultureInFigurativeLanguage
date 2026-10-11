#!/bin/bash
# Evaluate one T2 arm (Arabic symbolism-tag experiment) on the Arabic suite.
#   sbatch eval_ar_t2.slurm <untagged|dict|sym>
#SBATCH --job-name=ar-t2-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --partition=h100
#SBATCH --output=logs/ar-t2-eval-%x-%j.out
#SBATCH --error=logs/ar-t2-eval-%x-%j.err
set -uo pipefail
ARM="${1:?usage: sbatch eval_ar_t2.slurm <untagged|dict|sym>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
M="$CK/qwen3p5-9b-ar-t2-${ARM}"
[ -f "$M/config.json" ] || { echo "no checkpoint at $M"; exit 1; }
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/ar
KB=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/ar-kb/data/idioms/ar/idioms_merged_llm_formatted.jsonl
MC=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar_t2/$ARM
mkdir -p "$OUT"

python -m culture.evaluation.perplexity --model_path "$M" --run_name "t2_$ARM" \
  --data_path "$DATA/ar_wiki_heldout.jsonl" --text_field text --output_dir "$OUT/ppl_wiki_heldout" \
  || echo "[warn] wiki ppl"

python -m culture.evaluation.run_eval --model_path "$M" --run_name "t2_$ARM" \
  --tasks kinayat_meaning,kinayat_cloze,ar_figurative,arabculture,alyah,dzirieval,arabic_cultural_qa,arabicmmlu,global_piqa_ar,jsonl:idiomatlas_mc_ar_seen,jsonl:idiomatlas_mc_ar_unseen,jsonl:symbolism_v2_ar_letter,jsonl:symbolism_ar \
  --ar_data_dir "$DATA/raw" --ar_kb_path "$KB" --mc_jsonl_dir "$MC" \
  --acqa_dialects msa --ar_num_fewshot 0 --batch_size 8 --output_dir "$OUT"
echo "T2_EVAL_DONE $ARM"
