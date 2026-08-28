#!/bin/bash
# Re-run ONLY the amthal in-domain PPL/BPB probe for one AR run (fixes the filename bug).
#   sbatch eval_ar_amthal_only.slurm <run>
#SBATCH --job-name=ar-amthal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ar/ar-amthal-%x-%j.out
#SBATCH --error=logs/ar/ar-amthal-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?run}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  ar-cpt-native-sft)  M="$CK/qwen3p5-9b-ar-cpt-native-sft" ;;
  ar-base-native-sft) M="$CK/qwen3p5-9b-ar-base-native-sft" ;;
  *) echo bad; exit 1;;
esac
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/ar
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar/$RUN
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --data_path "$DATA/ar_amthal_heldout.jsonl" --text_field text \
  --output_dir "$OUT/ppl_ar_amthal"
echo "AMTHAL_DONE $RUN"
