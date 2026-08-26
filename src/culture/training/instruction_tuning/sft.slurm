#!/bin/bash
# Generic SFT launcher for the 4-node Qwen3.5-9B instruction-tuning runs.
# Usage:  sbatch sft.slurm configs/qwen3p5_9b_sft_<name>.yaml
# Mirrors ../continued_pretraining/cpt_ar.slurm (same distributed wiring).
#SBATCH --job-name=qwen-sft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning
#SBATCH --output=logs/sft-%x-%j.out
#SBATCH --error=logs/sft-%x-%j.err

set -euo pipefail
mkdir -p logs

IT_DIR="/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning"
CPT_DIR="/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/continued_pretraining"
CONFIG="${1:?usage: sbatch sft.slurm <config.yaml>}"
[[ "$CONFIG" = /* ]] || CONFIG="${IT_DIR}/${CONFIG}"

VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"

source "${CPT_DIR}/env.sh"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}"
export WANDB_MODE=offline
export WANDB_DIR="${IT_DIR}/logs"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

OUT=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG}'))['output_dir'])")
mkdir -p "$OUT"
echo "CONFIG=$CONFIG"; echo "OUTPUT_DIR=$OUT"

srun bash -c '
  export PATH="'"${VENV}"'/bin:${PATH}"
  export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_CACHE="/tmp/hf_cache_'"${SLURM_JOB_ID}"'"
  export WANDB_MODE=offline
  export WANDB_DIR="'"${IT_DIR}"'/logs"
  export TRITON_CACHE_DIR="/tmp/triton_'"${SLURM_JOB_ID}"'"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  FORCE_TORCHRUN=1 \
  NNODES='"${SLURM_NNODES}"' NODE_RANK=${SLURM_NODEID} \
  MASTER_ADDR='"${MASTER_ADDR}"' MASTER_PORT='"${MASTER_PORT}"' \
  NPROC_PER_NODE=8 \
  llamafactory-cli train '"${CONFIG}"'
'
