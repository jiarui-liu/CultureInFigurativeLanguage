#!/bin/bash
# One judge x one language of the full-KB audit on ONE GPU for under an hour; resubmits itself until done.
#   squeue -u $USER -h -n $NAME -t PD | grep -q . || sbatch --job-name=$NAME --gres=gpu:1 --export=ALL,JUDGE=qwen27b|gemma,LANG_=zh,DEPTH=0 full_kb_audit.slurm
#SBATCH --job-name=kbaudit
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-o5-28,babel-q5-32,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:59:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/kbaudit-%j.out
set -euo pipefail
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
source $REPO/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface PYTHONPATH=$REPO/src
A=/data/group_data/r3lit_culture_pretrain/culture/bidir/audit/full
NAME=kb_${JUDGE}_${LANG_}
scontrol update jobid=$SLURM_JOB_ID name=$NAME || true
[ -f $A/DONE_${JUDGE}_${LANG_} ] && exit 0
if squeue -u $USER -h -n $NAME -t R -o %i | grep -v "^$SLURM_JOB_ID$" | grep -q .; then echo "another $NAME is running"; exit 0; fi
T0=$(date +%s)
if [ "$JUDGE" = qwen27b ]; then M=$(python $REPO/src/culture/bidirectional/stage_hf.py Qwen/Qwen3.5-27B-FP8 | tail -1); Q=""
else M=$(python $REPO/src/culture/bidirectional/stage_hf.py google/gemma-4-26B-A4B-it | tail -1); Q="--quant fp8"; fi
USED=$(( ($(date +%s) - T0) / 60 ))
python -m culture.analysis.rebuttal.full_kb_audit_judge judge --out $A --name $JUDGE --model "$M" --lang $LANG_ $Q \
  --budget_min $(( 52 - USED )) --chunk ${CHUNK:-1500}
if [ ! -f $A/DONE_${JUDGE}_${LANG_} ] && [ ${DEPTH:-0} -lt 8 ]; then
  squeue -u $USER -h -n $NAME -t PD | grep -q . || sbatch --job-name=$NAME --gres=gpu:1 --export=ALL,JUDGE=$JUDGE,LANG_=$LANG_,DEPTH=$(( ${DEPTH:-0} + 1 )) $REPO/src/culture/analysis/rebuttal/slurm/full_kb_audit.slurm
fi
