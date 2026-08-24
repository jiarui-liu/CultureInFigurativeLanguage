#!/bin/bash
# Generic 4-node CPT launcher for the filtered-untagged controls (paper point 6).
#   sbatch cpt_untagged.slurm <config_basename>   e.g. qwen3p5_9b_cpt_ar_untagged.yaml
#SBATCH --job-name=qwen-cpt-untag
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/continued_pretraining
#SBATCH --output=logs/cpt-%j.out
#SBATCH --error=logs/cpt-%j.err
set -euo pipefail
mkdir -p logs
CFG_NAME=${1:?usage: sbatch cpt_untagged.slurm <config_basename>}
CPT_DIR="/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/continued_pretraining"
CONFIG="${CPT_DIR}/configs/${CFG_NAME}"
[[ -f "$CONFIG" ]] || { echo "no config $CONFIG"; exit 1; }

VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"
source "${CPT_DIR}/env.sh"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}"
export WANDB_MODE=offline
export WANDB_DIR="${CPT_DIR}/logs"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

srun bash -c '
  export PATH="'"${VENV}"'/bin:${PATH}"
  export HF_DATASETS_OFFLINE=1
  export HF_DATASETS_CACHE="/tmp/hf_cache_'"${SLURM_JOB_ID}"'"
  export WANDB_MODE=offline
  export WANDB_DIR="'"${CPT_DIR}"'/logs"
  export TRITON_CACHE_DIR="/tmp/triton_'"${SLURM_JOB_ID}"'"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  FORCE_TORCHRUN=1 \
  NNODES='"${SLURM_NNODES}"' NODE_RANK=${SLURM_NODEID} \
  MASTER_ADDR='"${MASTER_ADDR}"' MASTER_PORT='"${MASTER_PORT}"' \
  NPROC_PER_NODE=8 \
  llamafactory-cli train '"${CONFIG}"'
'
