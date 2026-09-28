#!/bin/bash
# Score raw web files for the full-scale (9B) Culture corpus; one part per array task.
#   sbatch --array=0-7 --export=ALL,LANG_=ar,REPO_ID=HuggingFaceFW/fineweb-2,PATTERN='data/arb_Arab/train/000_0000[0-3].parquet',NPARTS=8 stream_select.slurm
#SBATCH --job-name=stream
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/stream-%A_%a.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
export HF_HOME=/scratch/jiaruil5/hf_home
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
python -m culture.bidirectional.stream_select_culture score --lang $LANG_ --repo $REPO_ID --pattern "$PATTERN" \
  --clf $B/clf/$LANG_.pkl --min_score ${MIN_SCORE:-1.5} --out_dir $B/full/$LANG_ \
  --part ${SLURM_ARRAY_TASK_ID:-0} --nparts ${NPARTS:-8} --workers 11 --rg_split ${RG_SPLIT:-1}
