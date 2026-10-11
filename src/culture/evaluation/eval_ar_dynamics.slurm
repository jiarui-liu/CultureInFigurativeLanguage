#!/bin/bash
# Training dynamics: evaluate an intermediate Arabic checkpoint on a compact task set.
# The paper reports only end-of-training numbers, so it cannot say when item-specific idiom
# knowledge appears, whether it saturates, or whether a cultural effect appears transiently
# and is then washed out.  Arabic is the right language for this: 1,608 steps, and it is the
# only language with a reverse-direction signal.
#   sbatch --array=0-11 eval_ar_dynamics.slurm
#SBATCH --job-name=ar-dyn
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=h100
#SBATCH --output=logs/ar-dyn-%A_%a.out
#SBATCH --error=logs/ar-dyn-%A_%a.err
set -uo pipefail
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

ARMS=(cpt unfiltered culture)
STEPS=(200 400 800 1200)
i=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$((i / 4))]}
ST=${STEPS[$((i % 4))]}

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$ARM" in
  cpt)        D="$CK/qwen3p5-9b-ar-cpt";;
  unfiltered) D="$CK/qwen3p5-9b-ar-cpt-unfiltered";;
  culture)    D="$CK/qwen3p5-9b-ar-cpt-culture";;
esac
M="$D/checkpoint-$ST"
[ -f "$M/config.json" ] || { echo "no checkpoint $M"; exit 0; }
# intermediate checkpoints carry no tokenizer; take it from the final directory
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json \
         generation_config.json chat_template.jinja; do
  [ -f "$D/$f" ] && [ ! -f "$M/$f" ] && cp "$D/$f" "$M/" 2>/dev/null
done

DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/ar
KB=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/ar-kb/data/idioms/ar/idioms_merged_llm_formatted.jsonl
MC=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar_dynamics/${ARM}_$ST
mkdir -p "$OUT"

python -m culture.evaluation.run_eval --model_path "$M" --run_name "${ARM}_$ST" \
  --tasks kinayat_meaning,ar_figurative,arabculture,alyah,jsonl:idiomatlas_mc_ar_seen,jsonl:idiomatlas_mc_ar_unseen,jsonl:symbolism_v2_ar_letter \
  --ar_data_dir "$DATA/raw" --ar_kb_path "$KB" --mc_jsonl_dir "$MC" \
  --acqa_dialects msa --ar_num_fewshot 0 --batch_size 8 --output_dir "$OUT"
echo "DYN_DONE $ARM $ST"
