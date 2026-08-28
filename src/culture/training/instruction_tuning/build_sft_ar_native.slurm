#!/bin/bash
# Build the native-Arabic SFT mixture (runbook §2): Quora-Arabic-GPT4 (native prompts)
# + CIDAR (human-reviewed cultural), 1x each, NO upsampling; SmolKalam dropped.
# English anchor downsampled to hold 60/40 (~35,367). -> train_sft_ar_native.
#   sbatch build_sft_ar_native.slurm
#SBATCH --job-name=sft-build-ar-native
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --partition=cpu,all
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning
#SBATCH --output=logs/build-ar-native-%j.out
#SBATCH --error=logs/build-ar-native-%j.err
set -euo pipefail
mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:${PATH}"
DATA=/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data
for d in quora-ar-gpt4 cidar; do
  [ -d "$DATA/$d" ] || { echo "FATAL: $DATA/$d not downloaded"; exit 1; }
done
echo "=== build NATIVE ARABIC mixture (Quora + CIDAR + English anchor) ==="
# target_total is the Quora cap (keeps all ~43K that pass filters); CIDAR (~10K) added on top.
# english_total 35,367 holds the 60/40 ratio against the ~53K native target (runbook D0.3b/D0.4).
python3 build_sft_mixture.py --lang ar --ar_recipe native \
  --out_dir "${DATA}/train_sft_ar_native" \
  --target_total 50000 --english_total 35367 --seed 42
echo "=== line count ==="; cat "${DATA}/train_sft_ar_native"/*.jsonl | wc -l
touch logs/BUILD_AR_NATIVE_DONE
