#!/bin/bash
# Free-generation meaning test: one 9B arm writes English explanations for one or more languages (one GPU, <30 min).
#   sbatch --export=ALL,RUN=cpt,LANGS="ar",MODEL=<hf spec> free_gen.slurm
#SBATCH --job-name=freegen
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-o5-28,babel-q5-32,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/freegen-%j.out
set -euo pipefail
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
source $REPO/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface PYTHONPATH=$REPO/src
LOCAL=$(python $REPO/src/culture/bidirectional/stage_hf.py "$MODEL" | tail -1)
cd $REPO && python -m culture.analysis.rebuttal.free_gen generate --model "$LOCAL" --run $RUN --langs $LANGS
