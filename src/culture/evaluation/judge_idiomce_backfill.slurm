#!/bin/bash
# Backfill the IdiomCE LLM-judge scores for the runs whose generations exist but
# were never judged: untagged (paper Untagged col) + hi-cpt-sft (+IT col) + hi-base-sft.
# No GPU: this only grades the stored idiomce.json hypotheses with the Llama API
# judge (gpt-5-4-mini), which is reachable from compute nodes (not the sandbox).
#   sbatch src/culture/evaluation/judge_idiomce_backfill.slurm
#SBATCH --job-name=hi-idiomce-judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/idiomce-judge-%j.out
#SBATCH --error=logs/idiomce-judge-%j.err
#SBATCH --partition=cpu,all

set -uo pipefail

REPO_ROOT="/storage/home/jiaruiliu/local/git-repos/culture-pretraining"
CIFL="${REPO_ROOT}/CultureInFigurativeLanguage"
cd "${CIFL}"
mkdir -p logs

VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${CIFL}/.eval_deps:${CIFL}/src"

# Drop the submit-time sandbox proxy so the compute node reaches api.llama.com.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export no_proxy="*" NO_PROXY="*"

# Llama API judge creds: mirror interactive last-export-wins from ~/.bashrc.
eval "$(grep -E '^export LLAMA_API_KEY=' "$HOME/.bashrc" | tail -1)"
export OPENAI_API_KEY="${LLAMA_API_KEY:?LLAMA_API_KEY not found in ~/.bashrc}"
export OPENAI_API_BASE="https://api.llama.com/experimental/compat/openai/v1"
export JUDGE_MODEL="gpt-5-4-mini-genai-responses"

echo "Judging IdiomCE for: untagged, hi-cpt-sft, hi-base-sft"
python -m culture.evaluation.judge_idiomce \
  --runs untagged,hi-cpt-sft,hi-base-sft \
  --judge_model "${JUDGE_MODEL}" \
  --provider openai \
  --batch_size 20

echo "DONE idiomce judge backfill"
