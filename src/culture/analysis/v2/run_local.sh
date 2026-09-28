#!/bin/bash
# Runs every v2 analysis step on the local GPU (A6000 48GB), one model per process.
set -x
source /data/group_data/r3lit_culture_pretrain/envs/bidir/bin/activate
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface PYTHONUNBUFFERED=1
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
export PYTHONPATH=$REPO/src:$REPO/src/culture/analysis/v2
cd $REPO/src/culture/analysis/v2
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.80} VLLM_MAX_SEQS=64
python run_primary_all.py ${PRIMARY_STEPS:-typology en_zh multi judge}   # Qwen3.5-27B-FP8 (+ Qwen3-Embedding-0.6B)
python entity_typology_validate.py --only Qwen3.5-9B --skip_openrouter
python entity_typology_validate.py --only aya --skip_openrouter
python entity_typology_validate.py --only nemotron             # OpenRouter free tier, best effort
python meaning_clusters.py --judges both                       # primary cached; runs Qwen3.5-9B then aya in one process
