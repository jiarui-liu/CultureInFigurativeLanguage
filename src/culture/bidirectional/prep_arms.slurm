#!/bin/bash
# Build + pack arms for one language.
#   sbatch --export=ALL,LANG_=ar,STAGE=base prep_arms.slurm      (random, idiom_tagged, idiom_untagged)
#   sbatch --export=ALL,LANG_=ar,STAGE=culture prep_arms.slurm   (culture_docs -> culture)
#   sbatch --export=ALL,LANG_=ar,STAGE=notes_arm prep_arms.slurm (culture_notes)
#SBATCH --job-name=prep-arms
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/prep-%x-%j.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
TOK=/data/group_data/r3lit_culture_pretrain/models/Qwen/Qwen3.5-2B-Base
BUDGET=${BUDGET:-300000000}
A=$B/arms/$LANG_; P=$B/packed/$LANG_
python -m culture.bidirectional.build_arms --lang $LANG_ --cmd $STAGE --budget $BUDGET --out_dir $A
case $STAGE in
  base) ARMS="random idiom_tagged idiom_untagged";;
  culture) ARMS="culture";;
  notes_arm) ARMS="culture_notes";;
esac
for arm in $ARMS; do
  src=$A/$arm.jsonl.gz; [ $arm = culture ] && src=$A/culture_docs.jsonl.gz
  [ -f $P/$arm.json ] && continue
  python -m culture.bidirectional.tokenize_pack --inputs $src --tokenizer $TOK --out $P/$arm \
    --seq_len 4096 --max_tokens $BUDGET --workers 14
  n=$(python -c "import json; print(json.load(open('$P/$arm.json'))['n_tokens'])")
  if [ "$n" -lt $((BUDGET - 4096)) ]; then echo "WARNING: $arm has only $n tokens (< budget)"; fi
done
