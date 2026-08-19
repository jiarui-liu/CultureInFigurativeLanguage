#!/bin/bash
# Slurm launcher for the 4-node Qwen 3.5 9B CHINESE continued-pretraining run.
# Sibling of cpt.slurm (Hindi); runs as a separate, parallel job.
#   sbatch cpt_zh.slurm     (from anywhere; --chdir pins the working dir)
#SBATCH --job-name=qwen-zh-cpt-unf
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
# TODO: add --account=... if your cluster requires accounting.

set -euo pipefail
mkdir -p logs

CPT_DIR="/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/continued_pretraining"
CONFIG="${CPT_DIR}/configs/qwen3p5_9b_cpt_zh_unfiltered.yaml"

# Reuse the monitorability-pretraining venv: it already has a working
# llamafactory-cli + transformers 5.6 (Qwen3.5 support) + deepspeed.
VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"

source "${CPT_DIR}/env.sh"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"

# Avoid NFS lock issues with datasets/wandb
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}"
export WANDB_MODE=offline
export WANDB_DIR="${CPT_DIR}/logs"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

mkdir -p /lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-zh-cpt-unfiltered

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
