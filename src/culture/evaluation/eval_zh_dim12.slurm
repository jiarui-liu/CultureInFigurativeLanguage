#!/bin/bash
# Chinese Dim-1 (LM) + Dim-2 (general English) for ONE checkpoint.
#   sbatch eval_zh_dim12.slurm base   (only the 2 Chinese PPL probes; Dim2/WikiText reused from base)
#   sbatch eval_zh_dim12.slurm cpt    (everything)
#SBATCH --job-name=zh-eval-d12
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval-zh-d12-%x-%j.out
#SBATCH --error=logs/eval-zh-d12-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_zh_dim12.slurm <base|cpt>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}"

BASE=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
CPT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-zh-cpt
case "$RUN" in base) M="$BASE";; cpt) M="$CPT";; unfiltered) M="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-zh-cpt-unfiltered";; *) echo bad RUN; exit 1;; esac
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/zh
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/$RUN
mkdir -p "$OUT"

# --- Dim 1: Chinese LM probes (local, always) ---
echo "### [Dim1] zh Wikipedia (clean, independent)"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --data_path "$DATA/zh_wiki_heldout.jsonl" --text_field text --output_dir "$OUT/ppl_zh_wiki"
echo "### [Dim1] zh chengyu in-domain (CONTAMINATED)"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --data_path "$DATA/zh_chengyu_heldout.jsonl" --text_field text --output_dir "$OUT/ppl_zh_chengyu"

# --- base reuses Dim2 + WikiText from the (identical) Qwen3.5-9B base run; cpt/unfiltered are distinct models so run them ---
if [ "$RUN" = cpt ] || [ "$RUN" = unfiltered ]; then
  echo "### [Dim1] WikiText-103 (English retention)"
  python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
    --hf_dataset Salesforce/wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
    --output_dir "$OUT/ppl_wikitext" || echo "[warn] wikitext failed"
  echo "### [Dim2] MMLU + BoolQ"
  python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
    --tasks mmlu,boolq --num_fewshot 0 --batch_size 8 --no_judge --output_dir "$OUT/en" || echo "[warn] mmlu/boolq failed"
  echo "### [Dim2] GSM8K + HumanEval"
  python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
    --tasks gsm8k,humaneval --gen_batch_size 16 --output_dir "$OUT/en_gen" || echo "[warn] gsm8k/humaneval failed"
fi
echo "DONE $RUN"
