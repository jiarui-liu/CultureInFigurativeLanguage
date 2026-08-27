#!/usr/bin/env bash
# One-shot launcher for the Chinese SFT DAG: build mixture -> train (cpt+base) ->
# eval (Chinese tasks + English retention). Idempotent-ish; records IDs.
# Run AFTER downloading lhoestq/Infinity-Instruct-Chinese-Only. Usage: bash launch_zh_sft.sh
set -euo pipefail
IT=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning
EV=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/evaluation
DATA=/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data
cd "$IT"

if [ -z "$(find "$DATA/infinity-instruct-zh" -name '*.parquet' -o -name '*.jsonl*' 2>/dev/null | head -1)" ]; then
  echo "ERROR: $DATA/infinity-instruct-zh has no parquet/jsonl files. Download first:"
  echo "  export HF_TOKEN=hf_xxx; hf download lhoestq/Infinity-Instruct-Chinese-Only --repo-type dataset --local-dir $DATA/infinity-instruct-zh"
  exit 1
fi

ACCT="--account=ar-ai-midpri"   # run on the midpri group; drop if not desired
sub(){ sbatch "$@" | grep -oE '[0-9]+'; }

B=$(sub $ACCT build_sft_zh.slurm)
TC=$(sub $ACCT --dependency=afterok:$B --job-name=sft-zh-cpt  sft.slurm configs/qwen3p5_9b_sft_zh-cpt-sft.yaml)
TB=$(sub $ACCT --dependency=afterok:$B --job-name=sft-zh-base sft.slurm configs/qwen3p5_9b_sft_zh-base-sft.yaml)
EC=$(sbatch $ACCT --dependency=afterok:$TC "$EV/eval_zh_sft.slurm"     zh-cpt-sft  | grep -oE '[0-9]+')
ECE=$(sbatch $ACCT --dependency=afterok:$TC "$EV/eval_english_zh.slurm" zh-cpt-sft  | grep -oE '[0-9]+')
EB=$(sbatch $ACCT --dependency=afterok:$TB "$EV/eval_zh_sft.slurm"     zh-base-sft | grep -oE '[0-9]+')
EBE=$(sbatch $ACCT --dependency=afterok:$TB "$EV/eval_english_zh.slurm" zh-base-sft | grep -oE '[0-9]+')

cat > logs/zh_sft_dag.env <<EOF
BUILD_ZH=$B
TRAIN_ZH_CPT=$TC
TRAIN_ZH_BASE=$TB
EVAL_ZH_CPT=$EC
EVAL_ZH_CPT_EN=$ECE
EVAL_ZH_BASE=$EB
EVAL_ZH_BASE_EN=$EBE
EOF
echo "=== zh SFT DAG submitted ==="; cat logs/zh_sft_dag.env
squeue -u jiaruiliu -o "%.10i %.16j %.2t %R" | grep -iE 'zh|JOBID'
