#!/bin/bash
# Upload artifacts to the HF *dataset* repo. MUST run on a compute node:
# the login node's proxy blocks huggingface.co (403); compute nodes reach it
# once the proxy is unset (same trick as eval_zh*.slurm).
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" upload_hf.slurm eval     # eval/{hi,zh}/*   (~20M)
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" upload_hf.slurm data     # data/*           (~115G)
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" upload_hf.slurm models   # final model+tokenizer only, NO optimizer states (~76G)
#   sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" upload_hf.slurm all
#SBATCH --job-name=hf-upload
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --output=logs/hf-upload-%x-%j.out
#SBATCH --error=logs/hf-upload-%x-%j.err
#SBATCH --export=ALL
set -uo pipefail
WHAT="${1:-all}"
REPO=Jerry9999/CultureInFigurativeLanguage
HF=/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin/hf
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs

# compute nodes reach HF only with the cluster proxy removed (mirrors eval_zh*.slurm)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export no_proxy="*" NO_PROXY="*"
: "${HF_TOKEN:?HF_TOKEN not set -- submit with: sbatch --export=ALL,HF_TOKEN=\"\$HF_TOKEN\" ...}"

DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data
EVAL=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval
CKPTS=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
RUNS="qwen3p5-9b-zh-cpt qwen3p5-9b-zh-cpt-unfiltered qwen3p5-9b-hi-cpt qwen3p5-9b-hi-cpt-unfiltered"

up_eval(){ echo "### eval (hi+zh) -> $REPO"; "$HF" upload "$REPO" "$EVAL" eval --repo-type dataset; }
up_data(){ echo "### data -> $REPO"; "$HF" upload "$REPO" "$DATA" data --repo-type dataset; }
up_models(){
  for r in $RUNS; do
    [ -d "$CKPTS/$r" ] || { echo "[skip] $r missing"; continue; }
    echo "### model $r -> models/$r  (final weights+tokenizer only; excluding optimizer/checkpoint-*)"
    "$HF" upload "$REPO" "$CKPTS/$r" "models/$r" --repo-type dataset \
      --exclude "checkpoint-*/**"
  done
}
case "$WHAT" in
  eval)   up_eval;;
  data)   up_data;;
  models) up_models;;
  all)    up_eval; up_data; up_models;;
  *) echo "usage: sbatch upload_hf.slurm <eval|data|models|all>"; exit 1;;
esac
echo "DONE upload: $WHAT"
