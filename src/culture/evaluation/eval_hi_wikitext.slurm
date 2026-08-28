#!/bin/bash
# Backfill WikiText-103 English-retention PPL/BPB for a Hindi checkpoint (Salesforce
# loader fix). Fills the Hindi +IT / Untagged WikiText cells (parity with zh/ar).
#   sbatch eval_hi_wikitext.slurm <untagged|hi-cpt-sft>
#SBATCH --job-name=hi-wikitext
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/hi-wikitext-%x-%j.out
#SBATCH --error=logs/hi-wikitext-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?run}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  untagged)    M="$CK/qwen3p5-9b-hi-cpt-untagged" ;;
  hi-cpt-sft)  M="$CK/qwen3p5-9b-hi-cpt-sft" ;;
  *) echo bad; exit 1;;
esac
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi/$RUN
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --hf_dataset Salesforce/wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
  --output_dir "$OUT/ppl_wikitext"
echo "HI_WIKITEXT_DONE $RUN"
