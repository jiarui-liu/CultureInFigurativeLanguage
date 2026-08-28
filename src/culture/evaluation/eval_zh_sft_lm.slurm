#!/bin/bash
# Backfill the LM perplexity probes (WikiText + zh-Wikipedia + zh-chengyu) on a zh SFT
# checkpoint, so the Chinese +IT column's LM rows can be filled (parity with hi/ar).
#   sbatch eval_zh_sft_lm.slurm <zh-cpt-sft|zh-base-sft>
#SBATCH --job-name=zh-sft-lm
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/zh-sft-lm-%x-%j.out
#SBATCH --error=logs/zh-sft-lm-%x-%j.err
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
  zh-cpt-sft)  M="$CK/qwen3p5-9b-zh-cpt-sft" ;;
  zh-base-sft) M="$CK/qwen3p5-9b-zh-base-sft" ;;
  *) echo bad; exit 1;;
esac
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/zh
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/$RUN
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --data_path "$DATA/zh_chengyu_heldout.jsonl" --text_field text --output_dir "$OUT/ppl_zh_chengyu" || echo "[warn] chengyu"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --data_path "$DATA/zh_wiki_heldout.jsonl" --text_field text --output_dir "$OUT/ppl_zh_wiki" || echo "[warn] zhwiki"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --hf_dataset Salesforce/wikitext --hf_config wikitext-103-raw-v1 --hf_split test --output_dir "$OUT/ppl_wikitext" || echo "[warn] wikitext"
echo "ZH_SFT_LM_DONE $RUN"
