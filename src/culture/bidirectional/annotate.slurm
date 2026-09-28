#!/bin/bash
# sbatch --export=ALL,LANG_=ar,N=12000,POOL=<dir>,OUT=<file> annotate.slurm
#SBATCH --job-name=annot
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/annot-%j.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
MODEL=$($REPO/src/culture/bidirectional/stage_model.sh ${ANNOT_MODEL:-/data/group_data/r3lit_culture_pretrain/models/google/gemma-4-26B-A4B-it})
python -m culture.bidirectional.annotate_culture --lang $LANG_ --pool_dir $POOL --n $N \
  --model $MODEL \
  --tp 2 --out $OUT
