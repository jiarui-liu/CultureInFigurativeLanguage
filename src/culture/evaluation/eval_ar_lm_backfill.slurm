#!/bin/bash
# Arabic LM/English backfill for ONE checkpoint, single H100. Fills the gaps needed
# to make the Arabic table symmetric with Hindi/Chinese:
#   - amthal in-domain PPL/BPB probe (idiom-specific, contaminated) -> ppl_ar_amthal
#   - WikiText-103 English retention PPL/BPB (Salesforce/wikitext loader fix) -> en/ppl_wikitext
#   - untagged only: MMLU/BoolQ (en/) + GSM8K/HumanEval (en_gen/) English retention
#   sbatch src/culture/evaluation/eval_ar_lm_backfill.slurm <base|cpt|unfiltered|untagged|ar-cpt-sft>
#SBATCH --job-name=ar-lm-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ar/ar-lmbackfill-%x-%j.out
#SBATCH --error=logs/ar/ar-lmbackfill-%x-%j.err
#SBATCH --partition=h100

set -uo pipefail

RUN="${1:?usage: sbatch eval_ar_lm_backfill.slurm <base|cpt|unfiltered|untagged|ar-cpt-sft>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  base)        M="/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B" ;;
  cpt)         M="$CK/qwen3p5-9b-ar-cpt" ;;
  unfiltered)  M="$CK/qwen3p5-9b-ar-cpt-unfiltered" ;;
  untagged)    M="$CK/qwen3p5-9b-ar-cpt-untagged" ;;
  ar-cpt-sft)  M="$CK/qwen3p5-9b-ar-cpt-sft" ;;
  *) echo "unknown RUN=$RUN"; exit 1 ;;
esac

EVROOT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar/$RUN
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/ar
OUT="$EVROOT/en"; OUTGEN="$EVROOT/en_gen"
mkdir -p "$EVROOT" "$OUT" "$OUTGEN"
echo "RUN=$RUN MODEL=$M"

# --- Dim2 amthal in-domain probe (idiom-specific, CONTAMINATED upper bound) ---
echo "### [Dim2] ar amthal in-domain (CONTAMINATED)"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --data_path "$DATA/ar_amthal_heldout.jsonl" --text_field text \
  --output_dir "$EVROOT/ppl_ar_amthal" || echo "[warn] amthal ppl failed"

# --- Dim1 English retention: WikiText-103 (Salesforce path fixes the old loader err) ---
echo "### [Dim1] WikiText-103 retention"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --hf_dataset Salesforce/wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
  --output_dir "$OUT/ppl_wikitext" || echo "[warn] wikitext failed"

# --- English MC/gen retention: only untagged is missing (base is the shared model) ---
if [ "$RUN" = "untagged" ]; then
  echo "### [Dim1] MMLU/BoolQ"
  python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
    --tasks mmlu,boolq --num_fewshot 0 --batch_size 8 --no_judge \
    --output_dir "$OUT" || echo "[warn] mmlu/boolq failed"
  echo "### [Dim1] GSM8K/HumanEval"
  python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
    --tasks gsm8k,humaneval --gen_batch_size 16 --output_dir "$OUTGEN" || echo "[warn] gsm8k/humaneval failed"
fi

echo "AR_LM_BACKFILL_DONE $RUN"
