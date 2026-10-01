#!/bin/bash
# Every 20 min, mirror the newest local checkpoint of each running training job to shared
# storage (covers jobs started before train_cpt.py mirrored its own checkpoints).
#   nohup bash mirror_loop.sh > $B/logs/mirror_loop.log 2>&1 &
HERE=$(cd "$(dirname "$0")" && pwd)
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
while true; do
  squeue -u $USER -h -t R -o "%j %N" | awk '$1 ~ /^t-(ar|hi|zh)-/' | while read -r name node; do
    run=${name#t-}; lang=${run%%-*}; run=${run#*-}
    echo "$(date '+%m-%d %H:%M') $lang/$run on $node: $(timeout 7000 ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no $node "bash $HERE/ckpt_fetch.sh $lang/$run" 2>&1 | tail -1)"
  done
  sleep 1200
done
