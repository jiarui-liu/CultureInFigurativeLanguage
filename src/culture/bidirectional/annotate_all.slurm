#!/bin/bash
# Culture-specificity annotation for several languages in one job (model staged once).
#   sbatch [--nodelist=<node with staged model>] --export=ALL,LANGS="ar hi zh",N=10000 annotate_all.slurm
#SBATCH --job-name=annot-all
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --exclude=babel-o5-28,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/annotall-%j.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
MODEL=$(python $REPO/src/culture/bidirectional/stage_hf.py ${ANNOT_MODEL:-google/gemma-4-26B-A4B-it} | tail -1)
for l in ${LANGS:-ar hi zh}; do
  [ -s $B/annot/$l.jsonl ] && continue
  python -m culture.bidirectional.annotate_culture --lang $l --pool_dir $B/pool/$l --n ${N:-10000} \
    --model $MODEL --tp 2 --out $B/annot/$l.jsonl
done
