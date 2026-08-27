#!/bin/bash
# Chinese CPT-eval pipeline applied to the SFT checkpoints (SAME tasks/harness as
# eval_zh.slurm). sbatch eval_zh_sft.slurm zh-cpt-sft ; sbatch eval_zh_sft.slurm zh-base-sft
#SBATCH --job-name=zh-eval-sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval-zh-%x-%j.out
#SBATCH --error=logs/eval-zh-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_zh_sft.slurm <zh-cpt-sft|zh-base-sft>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  zh-cpt-sft)  M="${CK}/qwen3p5-9b-zh-cpt-sft" ;;
  zh-base-sft) M="${CK}/qwen3p5-9b-zh-base-sft" ;;
  *) echo bad RUN; exit 1;;
esac
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/zh
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/$RUN
mkdir -p "$OUT"
python -m culture.evaluation.run_eval \
  --model_path "$M" --run_name "$RUN" \
  --tasks chid,chengyu_bench,cmmlu,ccpm \
  --chid_path "$DATA/chid_eval3000.jsonl" --chid_answer_path "" \
  --chengyu_bench_dir "$DATA/ChengyuBench" --chengyu_bench_subtask connotation \
  --ccpm_path "$DATA/CCPM/valid.jsonl" \
  --cmmlu_dir "$DATA/cmmlu" --cmmlu_subjects all \
  --num_fewshot 0 --cmmlu_num_fewshot 5 \
  --batch_size 8 \
  --output_dir "$OUT"
echo "DONE $RUN -> $OUT"
