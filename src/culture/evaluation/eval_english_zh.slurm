#!/bin/bash
# English-retention (Dim-1/2) for the CHINESE SFT checkpoints, culture-package harness.
#   sbatch eval_english_zh.slurm <zh-cpt-sft|zh-base-sft>
#SBATCH --job-name=zh-eval-en
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval-zh-en-%x-%j.out
#SBATCH --error=logs/eval-zh-en-%x-%j.out
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_english_zh.slurm <zh-cpt-sft|zh-base-sft>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  zh-cpt-sft)  M="${CK}/qwen3p5-9b-zh-cpt-sft" ;;
  zh-base-sft) M="${CK}/qwen3p5-9b-zh-base-sft" ;;
  *) echo "unknown RUN=$RUN"; exit 1 ;;
esac
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/$RUN/en
OUTGEN=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/$RUN/en_gen
mkdir -p "$OUT" "$OUTGEN"
python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
  --tasks mmlu,boolq --num_fewshot 0 --batch_size 8 --no_judge --output_dir "$OUT" || echo "[warn] mmlu/boolq"
python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
  --tasks gsm8k,humaneval --gen_batch_size 16 --output_dir "$OUTGEN" || echo "[warn] gsm8k/humaneval"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
  --output_dir "$OUT/ppl_wikitext" || echo "[warn] wikitext"
echo "ENG_ZH_DONE $RUN"
