#!/bin/bash
#SBATCH --job-name=mc4_en
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --output=mc4_en.out
#SBATCH --error=mc4_en.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G


source ~/.bashrc
conda activate culture

# Full English C4: ~365M docs. At ~50K docs/min, one 48h run covers ~144M docs.
# Use --skip-docs to resume across multiple submissions:
#   Run 1: --skip-docs 0   (docs 0 - 143,999,999)
#   Run 2: --skip-docs 144000000 (docs 144M - 287,999,999)
#   Run 3: --skip-docs 288000000 (docs 288M - end)
# Output files are suffixed by skip offset so they don't overwrite each other.
# After all runs, merge: cat doc_indices_en*.json | jq -s '.[0].doc_indices + .[1].doc_indices + ...'

SKIP=${1:-0}

python3 /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/training/mC4/download_and_filter_mc4.py \
    --lang en \
    --indices-only \
    --batch-size 100000 \
    --skip-docs $SKIP
