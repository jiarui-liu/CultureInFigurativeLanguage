#!/bin/bash
# LLM judge for the free-generation test (TASK=free_gen) or paraphrase verification (TASK=paraphrase_verified).
#   sbatch --export=ALL,TASK=free_gen,JUDGE=gemma|qwen27b judge_x.slurm      (one GPU, <1 h)
#SBATCH --job-name=judgex
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-o5-28,babel-q5-32,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:59:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/judgex-%j.out
set -euo pipefail
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
source $REPO/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface PYTHONPATH=$REPO/src
if [ "$JUDGE" = qwen27b ]; then M=$(python $REPO/src/culture/bidirectional/stage_hf.py Qwen/Qwen3.5-27B-FP8 | tail -1); Q=""
else M=$(python $REPO/src/culture/bidirectional/stage_hf.py google/gemma-4-26B-A4B-it | tail -1); Q="--quant fp8"; fi
cd $REPO && python -m culture.analysis.rebuttal.$TASK judge --name $JUDGE --model "$M" $Q
