#!/bin/bash
# Point 4: English-retention (Dim-1/2) for the ARABIC checkpoints, via the culture
# package harness (NOT lm_eval, which isn't installed). Base == shared Qwen3.5-9B
# (already measured), so only cpt / unfiltered are needed.
#   sbatch eval_english_ar.slurm <cpt|unfiltered>
#SBATCH --job-name=ar-eval-en
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=h100
#SBATCH --output=logs/ar/eval-en-%x-%j.out
#SBATCH --error=logs/ar/eval-en-%x-%j.out
set -uo pipefail
RUN="${1:?usage: sbatch eval_english_ar.slurm <cpt|unfiltered>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

case "$RUN" in
  cpt)        M="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-ar-cpt" ;;
  unfiltered) M="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-ar-cpt-unfiltered" ;;
  *) echo "unknown RUN=$RUN (use cpt|unfiltered)"; exit 1 ;;
esac
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar/$RUN/en
OUTGEN=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar/$RUN/en_gen
mkdir -p "$OUT" "$OUTGEN"
echo "RUN=$RUN MODEL=$M"

echo "### [Dim2] MMLU + BoolQ (0-shot, loglik)"
python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
  --tasks mmlu,boolq --num_fewshot 0 --batch_size 8 --no_judge \
  --output_dir "$OUT" || echo "[warn] mmlu/boolq failed"

echo "### [Dim2] GSM8K (8-shot CoT) + HumanEval (pass@1, sandboxed)"
python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
  --tasks gsm8k,humaneval --gen_batch_size 16 --output_dir "$OUTGEN" || echo "[warn] gsm8k/humaneval failed"

echo "### [Dim1] WikiText-103 (English retention, BPB/PPL)"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
  --output_dir "$OUT/ppl_wikitext" || echo "[warn] wikitext failed"

echo "ENG_AR_DONE $RUN -> $OUT"
