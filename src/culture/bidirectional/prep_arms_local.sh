#!/bin/bash
# Same as prep_arms.slurm but runs on the login/vscode node (no preemption).
#   bash prep_arms_local.sh <lang> <stage> [workers]
set -euo pipefail
LANG_=$1; STAGE=$2; W=${3:-2}
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
TOK=/scratch/jiaruil5/hf/Qwen__Qwen3.5-2B-Base
[ -f $TOK/tokenizer.json ] || python $REPO/src/culture/bidirectional/stage_hf.py Qwen/Qwen3.5-2B-Base >/dev/null
BUDGET=${BUDGET:-300000000}
A=$B/arms/$LANG_; P=$B/packed/$LANG_
if [ ! -f $A/.${STAGE}_done ]; then
  python -m culture.bidirectional.build_arms --lang $LANG_ --cmd $STAGE --budget $BUDGET --out_dir $A && touch $A/.${STAGE}_done
fi
case $STAGE in base) ARMS="random idiom_tagged idiom_untagged";; culture) ARMS="culture";; notes_arm) ARMS="culture_notes";; esac
for arm in $ARMS; do
  src=$A/$arm.jsonl.gz; [ $arm = culture ] && src=$A/culture_docs.jsonl.gz
  [ -f $P/$arm.json ] && continue
  python -m culture.bidirectional.tokenize_pack --inputs $src --tokenizer $TOK --out $P/$arm --seq_len 4096 --max_tokens $BUDGET --workers $W
  n=$(python -c "import json; print(json.load(open('$P/$arm.json'))['n_tokens'])")
  [ "$n" -lt $((BUDGET - 4096)) ] && echo "WARNING: $LANG_/$arm has only $n tokens (< budget)"
done
echo "PREP_DONE $LANG_ $STAGE"
