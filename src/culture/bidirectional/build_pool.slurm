#!/bin/bash
# Builds all pool shards listed in $B/pool_tasks.txt ("lang source shard args..."), 24 in parallel.
# Finished shards are skipped, so the job can be requeued safely.
#SBATCH --job-name=pool
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/pool-%j.out
set -uo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
B=/data/group_data/r3lit_culture_pretrain/culture/bidir
cat $B/pool_tasks.txt | xargs -P 24 -L 1 bash -c 'python -m culture.bidirectional.build_pool --lang $0 --source $1 --shard $2 --out_dir '$B'/pool/$0 "${@:3}"'
