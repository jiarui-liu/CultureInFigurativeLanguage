#!/bin/bash
# One-shot DAG for the SECOND Arabic instruction-tuning run (NATIVE recipe:
# Quora-Arabic-GPT4 + CIDAR + English anchor). build -> train {cpt,base} -> full eval.
# Outputs to NEW folders (train_sft_ar_native; ckpts *-ar-{cpt,base}-native-sft;
# eval/ar/ar-{cpt,base}-native-sft) so nothing existing is overwritten.
#   bash launch_ar_native_sft.sh
set -euo pipefail
IT=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning
EVDIR=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/evaluation
DATA=/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data
cd "$IT"; mkdir -p logs
for d in quora-ar-gpt4 cidar; do
  [ -d "$DATA/$d" ] || { echo "FATAL: $DATA/$d not downloaded. Run the hf download commands first."; exit 1; }
done
ACCT="--account=ar-ai-midpri"
BUILD=$(sbatch $ACCT --parsable build_sft_ar_native.slurm)
TCPT=$(sbatch $ACCT --parsable --dependency=afterok:$BUILD sft.slurm configs/qwen3p5_9b_sft_ar-cpt-native-sft.yaml)
TBASE=$(sbatch $ACCT --parsable --dependency=afterok:$BUILD sft.slurm configs/qwen3p5_9b_sft_ar-base-native-sft.yaml)
ECPT=$(sbatch $ACCT --parsable --dependency=afterok:$TCPT "$EVDIR/eval_ar_sft_full.slurm" ar-cpt-native-sft)
EBASE=$(sbatch $ACCT --parsable --dependency=afterok:$TBASE "$EVDIR/eval_ar_sft_full.slurm" ar-base-native-sft)
{
  echo "BUILD_AR_NATIVE=$BUILD"
  echo "TRAIN_AR_CPT_NATIVE=$TCPT"
  echo "TRAIN_AR_BASE_NATIVE=$TBASE"
  echo "EVAL_AR_CPT_NATIVE=$ECPT"
  echo "EVAL_AR_BASE_NATIVE=$EBASE"
} > logs/ar_native_dag.env
echo "DAG submitted; IDs in $IT/logs/ar_native_dag.env"; cat logs/ar_native_dag.env
