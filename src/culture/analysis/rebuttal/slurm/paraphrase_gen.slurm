#!/bin/bash
# Paraphrase every IdiomAtlas-MC option of one language with Qwen3.5-27B-FP8 on one GPU (<1 h).
#   sbatch --export=ALL,LANG_=ar paraphrase_gen.slurm
#SBATCH --job-name=paragen
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-o5-28,babel-q5-32,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:59:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/paragen-%j.out
set -euo pipefail
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
source $REPO/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface PYTHONPATH=$REPO/src
M=$(python $REPO/src/culture/bidirectional/stage_hf.py Qwen/Qwen3.5-27B-FP8 | tail -1)
cd $REPO && python -m culture.analysis.rebuttal.paraphrase_mc generate --model "$M" --langs $LANG_
