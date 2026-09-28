#!/bin/bash
# Train the culture classifier for a language (part 0 only) and score a slice of its pool.
#   sbatch --array=0-3 --export=ALL,LANG_=ar,NPARTS=4 classify.slurm
#SBATCH --job-name=clf
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/clf-%A_%a.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
export HF_HOME=/scratch/jiaruil5/hf_home
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
PART=${SLURM_ARRAY_TASK_ID:-0}
if [ ! -f $B/clf/$LANG_.pkl ]; then
  if [ "$PART" = 0 ]; then
    python -m culture.bidirectional.culture_classifier train --annot $B/annot/$LANG_.jsonl --model_out $B/clf/$LANG_.pkl
  else
    while [ ! -f $B/clf/$LANG_.pkl ]; do sleep 30; done
  fi
fi
python -m culture.bidirectional.culture_classifier score --model_in $B/clf/$LANG_.pkl \
  --pool_dir $B/pool/$LANG_ --out_dir $B/scores/$LANG_ --part $PART --nparts ${NPARTS:-4}
