#!/bin/bash
# Arabic idiom filter+tag on the locally downloaded FineWeb-2 arb_Arab files (array 0-7).
#SBATCH --job-name=ar-filter
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/arfilter-%A_%a.out
set -euo pipefail
source /data/group_data/r3lit_culture_pretrain/envs/bidir/bin/activate
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
cd $REPO; export PYTHONPATH=$REPO/src
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
P=$B/raw/fineweb2_ar/data/arb_Arab/train/000_0000${SLURM_ARRAY_TASK_ID}.parquet
python src/culture/training/mC4/filter_and_tag_ar.py filter \
  --idioms culture/data/idioms/ar/idioms_merged_llm_formatted.jsonl \
  --data_files "$P" --shard_prefix "p${SLURM_ARRAY_TASK_ID}_" \
  --out $B/corpora/ar/idiom_tagged --max_docs_per_idiom 400
