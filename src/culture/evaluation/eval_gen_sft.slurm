#!/bin/bash
# Hindi generative English-retention eval (GSM8K/HumanEval) for SFT checkpoints.
# Same as eval_gen.slurm, SFT run-name mapping.
#   sbatch eval_gen_sft.slurm hi-cpt-sft ; sbatch eval_gen_sft.slurm hi-base-sft
#SBATCH --job-name=hi-eval-gen-sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval-gen-%x-%j.out
#SBATCH --error=logs/eval-gen-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_gen_sft.slurm <hi-cpt-sft|hi-base-sft>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  hi-cpt-sft)  M="${CK}/qwen3p5-9b-hi-cpt-sft" ;;
  hi-base-sft) M="${CK}/qwen3p5-9b-hi-base-sft" ;;
  *) echo "unknown RUN=$RUN"; exit 1 ;;
esac
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi/$RUN/en_gen
python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
  --tasks gsm8k,humaneval --gen_batch_size 16 --output_dir "$OUT"
echo "DONE $RUN"
