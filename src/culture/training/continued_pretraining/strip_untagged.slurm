#!/bin/bash
# Build a filtered-untagged corpus (paper point 6) for one language by stripping
# the appended knowledge block from the augmented tagged shards.
#   sbatch strip_untagged.slurm <lang> <src_dir> <out_dir>
#SBATCH --job-name=strip-untag
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --partition=cpu,all
#SBATCH --output=logs/ar/strip-%x-%j.out
#SBATCH --error=logs/ar/strip-%x-%j.out
set -uo pipefail
LANG_=${1:?lang}; SRC=${2:?src_dir}; OUT=${3:?out_dir}
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
GLOB_ARG=""
[[ "$LANG_" == "ar" ]] && GLOB_ARG="--glob tagged_p*_*.json.gz"   # ar shards are tagged_pN_NNNNN
python src/culture/training/continued_pretraining/strip_tags.py \
    --lang "$LANG_" --src_dir "$SRC" --out_dir "$OUT" $GLOB_ARG
echo "STRIP_DONE lang=$LANG_ out=$OUT rc=$?"
