#!/bin/bash
# Cultural notes for the culture-ranked documents of one language, one part per array task.
#   sbatch --array=0-3 --export=ALL,LANG_=ar,NPARTS=4 notes.slurm
#SBATCH --job-name=notes
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:2
#SBATCH --exclude=babel-o5-28,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/notes-%A_%a.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
MODEL=$(python $REPO/src/culture/bidirectional/stage_hf.py ${NOTES_MODEL:-google/gemma-4-26B-A4B-it} | tail -1)
python -m culture.bidirectional.culture_notes --lang $LANG_ --input $B/arms/$LANG_/culture_docs.jsonl.gz \
  --model $MODEL --tp 2 --part ${SLURM_ARRAY_TASK_ID:-0} --nparts ${NPARTS:-4} --out_dir $B/arms/$LANG_/notes
