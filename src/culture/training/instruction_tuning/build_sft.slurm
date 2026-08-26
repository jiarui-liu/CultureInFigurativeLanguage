#!/bin/bash
# Build both SFT mixtures (hi + ar) as a CPU job. No network (reads local parquet).
#   sbatch build_sft.slurm
#SBATCH --job-name=sft-build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --partition=cpu,all
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err

set -euo pipefail
mkdir -p logs
VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"
DATA=/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data

echo "=== build HINDI mixture ==="
python3 build_sft_mixture.py --lang hi --out_dir "${DATA}/train_sft_hi" \
  --target_total 180000 --english_total 120000 --seed 42

echo "=== build ARABIC mixture ==="
python3 build_sft_mixture.py --lang ar --out_dir "${DATA}/train_sft_ar" \
  --target_total 180000 --english_total 120000 --seed 42

echo "=== DONE. shard counts / line counts ==="
for L in hi ar; do
  echo "train_sft_${L}:"; ls "${DATA}/train_sft_${L}"/*.jsonl | wc -l
  cat "${DATA}/train_sft_${L}"/*.jsonl | wc -l
done
touch logs/BUILD_DONE
