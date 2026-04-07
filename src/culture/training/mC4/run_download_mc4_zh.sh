#!/bin/bash
#SBATCH --job-name=mc4_zh
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --output=mc4_zh.out
#SBATCH --error=mc4_zh.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G


source ~/.bashrc
conda activate culture

# Chinese C4 (multilingual subset) is much smaller than English.
# Should complete in a single 48h run.

python3 /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/training/mC4/download_and_filter_mc4.py \
    --lang zh \
    --indices-only \
    --batch-size 100000
