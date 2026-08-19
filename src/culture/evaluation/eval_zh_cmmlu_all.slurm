#!/bin/bash
#SBATCH --job-name=zh-cmmlu-all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/zh-cmmlu-all-%x-%j.out
#SBATCH --error=logs/zh-cmmlu-all-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_zh_cmmlu_all.slurm <base|cpt|unfiltered>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/tr_${SLURM_JOB_ID}"
BASE=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
CPT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-zh-cpt
UNFM=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-zh-cpt-unfiltered
case "$RUN" in base) M="$BASE";; cpt) M="$CPT";; unfiltered) M="$UNFM";; *) echo bad; exit 1;; esac
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/zh
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/$RUN
python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
  --tasks cmmlu --cmmlu_dir "$DATA/cmmlu" --cmmlu_subjects all \
  --cmmlu_num_fewshot 5 --batch_size 8 --output_dir "$OUT/_cmmlu_all"
# patch full-CMMLU into the main results
python - "$OUT" <<'PY'
import json,sys,shutil,os
out=sys.argv[1]
full=json.load(open(f"{out}/_cmmlu_all/cmmlu.json"))
shutil.copy(f"{out}/_cmmlu_all/cmmlu.json", f"{out}/cmmlu.json")
s=json.load(open(f"{out}/summary.json")); s["tasks"]["cmmlu"]=full["metrics"]; s["tasks"]["cmmlu"]["subjects"]="all67"
json.dump(s, open(f"{out}/summary.json","w"), ensure_ascii=False, indent=2)
print("patched cmmlu(all67):", full["metrics"].get("acc"), "n=", full["metrics"].get("n"))
PY
echo "DONE $RUN"
