#!/bin/bash
#SBATCH --job-name=hi-eval-gen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval-gen-%x-%j.out
#SBATCH --error=logs/eval-gen-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_gen.slurm <base|cpt>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
BASE=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
CPT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt
UNF=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt-unfiltered
case "$RUN" in
  base) M="$BASE" ;; cpt) M="$CPT" ;; unfiltered) M="$UNF" ;;
  *) echo "unknown RUN=$RUN (use base|cpt|unfiltered)"; exit 1 ;;
esac
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi/$RUN/en_gen
python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
  --tasks gsm8k,humaneval --gen_batch_size 16 --output_dir "$OUT"
echo "DONE $RUN"
