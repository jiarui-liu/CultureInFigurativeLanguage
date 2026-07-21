#!/usr/bin/env bash
# Shared distributed-launch environment. Source on every node.

export GPUS_PER_NODE=8
export MASTER_PORT="${MASTER_PORT:-29500}"

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
