#!/bin/bash
# Mirror the newest node-local checkpoint of one run to shared storage (idempotent).
#   bash ckpt_fetch.sh <lang>/<run>
# Run on the node that holds the checkpoint: via `sbatch --nodelist=<node> ckpt_fetch.sh ...`
# for a finished/preempted job, or via ssh for a running one.
#SBATCH --job-name=ckpt-fetch
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/fetch-%j.out
set -euo pipefail
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
OUT=$B/ckpt/$1
CK=/scratch/jiaruil5/ckpt/$(echo -n "$OUT" | md5sum | cut -c1-8)
R=$B/ckpt_resume/$1
name=$(ls -d $CK/checkpoint-* 2>/dev/null | xargs -r -n1 basename | sort -t- -k2 -n | tail -1)
[ -n "$name" ] || { echo "no local checkpoint in $CK"; exit 0; }
[ -f $CK/$name/trainer_state.json ] || { echo "$name incomplete"; exit 0; }
cur=$(cat $R/COMPLETE 2>/dev/null || true)
if [ -n "$cur" ] && [ "${cur#checkpoint-}" -ge "${name#checkpoint-}" ]; then echo "mirror up to date ($cur)"; exit 0; fi
[ -f $R/.lock ] && [ $(( $(date +%s) - $(stat -c %Y $R/.lock) )) -lt 7200 ] && { echo "copy in progress"; exit 0; }
free=$(df --output=avail -BG $B | tail -1 | tr -dc 0-9)
[ "$free" -ge 55 ] || { echo "skip: only ${free}G free"; exit 0; }
mkdir -p $R; touch $R/.lock
echo "copying $CK/$name -> $R"
rm -rf $R/.tmp; cp -r $CK/$name $R/.tmp
find $R -maxdepth 1 -name 'checkpoint-*' -exec rm -rf {} +
mv $R/.tmp $R/$name; echo $name > $R/COMPLETE.tmp; mv $R/COMPLETE.tmp $R/COMPLETE; rm -f $R/.lock
echo "done $name"
