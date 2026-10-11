#!/bin/bash
# T2: one arm of the Arabic symbolism-tag experiment, 4 GPUs.
#   sbatch cpt_ar_t2.slurm <untagged|dict|sym>
#SBATCH --job-name=ar-t2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=700G
#SBATCH --time=36:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/continued_pretraining
#SBATCH --output=logs/ar-t2-%x-%j.out
#SBATCH --error=logs/ar-t2-%x-%j.err
set -euo pipefail
mkdir -p logs
ARM="${1:?usage: sbatch cpt_ar_t2.slurm <untagged|dict|sym>}"

CPT_DIR="/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/continued_pretraining"
CONFIG="${CPT_DIR}/configs/qwen3p5_9b_cpt_ar_t2_${ARM}.yaml"
[ -f "$CONFIG" ] || { echo "no config $CONFIG"; exit 1; }

VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 2000))   # distinct port per job: three arms may share a node
source "${CPT_DIR}/env.sh"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}"
export WANDB_MODE=offline
export WANDB_DIR="${CPT_DIR}/logs"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NODE=4 \
MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
llamafactory-cli train "${CONFIG}"
