#!/bin/bash
#SBATCH --job-name=ceiling
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --exclude=babel-o5-28,babel-q5-32,babel-x9-16
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/ceiling-%j.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src VLLM_TP=1
Q=$(python $REPO/src/culture/bidirectional/stage_hf.py Qwen/Qwen3.5-27B-FP8 | tail -1)
python -m culture.bidirectional.probe_ceiling --model $Q --name qwen27b --version "${VERSION:-}"
G=$(python $REPO/src/culture/bidirectional/stage_hf.py google/gemma-4-26B-A4B-it | tail -1)
LLM_QUANT=fp8 python -m culture.bidirectional.probe_ceiling --model $G --name gemma --version "${VERSION:-}"
