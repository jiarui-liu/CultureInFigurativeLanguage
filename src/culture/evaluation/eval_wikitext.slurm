#!/bin/bash
#SBATCH --job-name=hi-eval-wt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-wt-%j.out
#SBATCH --error=logs/eval-wt-%j.err
#SBATCH --partition=h100
set -uo pipefail
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" TOKENIZERS_PARALLELISM=false
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi
BASE=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
CPT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt
for pair in "base $BASE" "cpt $CPT"; do
  set -- $pair; name=$1; model=$2
  echo "### WikiText-103 $name"
  python -m culture.evaluation.perplexity --model_path "$model" --run_name "$name" \
    --hf_dataset Salesforce/wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
    --output_dir "$OUT/$name/ppl_wikitext" || echo "[warn] $name wikitext failed"
done
echo "DONE"
