#!/bin/bash
# Build the Chinese SFT mixture (Infinity-Instruct-Chinese-Only + SmolTalk2 anchor).
# CPU job, no network (reads local parquet). sbatch build_sft_zh.slurm
#SBATCH --job-name=sft-build-zh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --partition=cpu,all
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning
#SBATCH --output=logs/build-zh-%j.out
#SBATCH --error=logs/build-zh-%j.err
set -euo pipefail
mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:${PATH}"
DATA=/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data
python3 build_sft_mixture.py --lang zh --out_dir "${DATA}/train_sft_zh" \
  --target_total 180000 --english_total 120000 --seed 42
echo "=== DONE ==="; ls "${DATA}/train_sft_zh"/*.jsonl | wc -l; cat "${DATA}/train_sft_zh"/*.jsonl | wc -l
touch logs/BUILD_ZH_DONE
