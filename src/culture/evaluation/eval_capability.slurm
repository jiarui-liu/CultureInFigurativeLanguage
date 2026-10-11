#!/bin/bash
# Capability profile for one checkpoint: in-language perplexity probes + English retention.
# The culture arms were never evaluated on either, so the paper's claim that the capability
# cost is "a cost of continued pretraining, not of idiom data" is untested for exactly the
# arm that reads the narrowest slice of the language.
#   sbatch eval_capability.slurm <ar|hi|zh> <base|unfiltered|untagged|cpt|culture|culturenotes>
#SBATCH --job-name=cap-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --partition=h100
#SBATCH --output=logs/cap-%x-%j.out
#SBATCH --error=logs/cap-%x-%j.err
set -uo pipefail
L="${1:?usage: sbatch eval_capability.slurm <lang> <arm>}"
RUN="${2:?usage: sbatch eval_capability.slurm <lang> <arm>}"

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  base)         M=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B;;
  unfiltered)   M="$CK/qwen3p5-9b-$L-cpt-unfiltered";;
  untagged)     M="$CK/qwen3p5-9b-$L-cpt-untagged";;
  cpt)          M="$CK/qwen3p5-9b-$L-cpt";;
  culture)      M="$CK/qwen3p5-9b-$L-cpt-culture";;
  culturenotes) M="$CK/qwen3p5-9b-$L-cpt-culturenotes";;
  *) echo "bad RUN: $RUN"; exit 1;;
esac
[ -f "$M/config.json" ] || { echo "no checkpoint at $M"; exit 1; }

OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/$L/$RUN
mkdir -p "$OUT/en" "$OUT/en_gen"
REPO_DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval
LUS=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval

ppl () {  # ppl <jsonl> <outdir>
  [ -f "$1" ] || { echo "[skip] missing probe $1"; return; }
  [ -f "$2/perplexity.json" ] && { echo "[have] $2"; return; }
  python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
    --data_path "$1" --text_field text --output_dir "$2" || echo "[warn] ppl $1"
}

echo "### in-language perplexity probes"
case "$L" in
  ar) ppl "$REPO_DATA/ar/ar_wiki_heldout.jsonl"      "$OUT/ppl_wiki_heldout"
      ppl "$REPO_DATA/ar/ar_fineweb2_heldout.jsonl"  "$OUT/ppl_fineweb2_heldout"
      ppl "$REPO_DATA/ar/ar_amthal_heldout.jsonl"    "$OUT/ppl_ar_amthal";;
  hi) ppl "$REPO_DATA/hi/hi_samanantar_heldout.jsonl" "$OUT/ppl_hi_samanantar"
      ppl "$LUS/hi/hi_samanantar_heldout.jsonl"       "$OUT/ppl_hi_samanantar"
      ppl "$REPO_DATA/hi/hi_proverbs_heldout.jsonl"   "$OUT/ppl_hi_proverbs"
      ppl "$LUS/hi/hi_proverbs_heldout.jsonl"         "$OUT/ppl_hi_proverbs";;
  zh) ppl "$REPO_DATA/zh/zh_wiki_heldout.jsonl"      "$OUT/ppl_zh_wiki"
      ppl "$REPO_DATA/zh/zh_chengyu_heldout.jsonl"   "$OUT/ppl_zh_chengyu";;
esac

echo "### English retention (needs the Hub)"
python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
  --tasks mmlu,boolq --num_fewshot 0 --batch_size 8 --no_judge \
  --output_dir "$OUT/en" || echo "[warn] mmlu/boolq"
python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
  --tasks gsm8k,humaneval --gen_batch_size 16 \
  --output_dir "$OUT/en_gen" || echo "[warn] gsm8k/humaneval"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
  --output_dir "$OUT/en/ppl_wikitext" || echo "[warn] wikitext"

echo "CAP_DONE $L/$RUN"
