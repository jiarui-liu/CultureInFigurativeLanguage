#!/bin/bash
# IdiomCE-only chat-template generation + Llama judge for ONE Hindi instruct ckpt.
# (Targeted re-run: the full eval_gen_chat had a wrong idiomce path; GSM8K/HumanEval
# already succeeded, so we redo only IdiomCE.)  Writes to eval/hi/<run>/idiomce_chat.
#   sbatch eval_idiomce_chat.slurm <hi-cpt-sft|hi-base-sft>
#SBATCH --job-name=idiomce-chat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/idiomce-chat-%x-%j.out
#SBATCH --error=logs/idiomce-chat-%x-%j.err
#SBATCH --partition=h100

set -uo pipefail
RUN="${1:?usage: sbatch eval_idiomce_chat.slurm <hi-cpt-sft|hi-base-sft>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/.eval_deps:$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  hi-cpt-sft)  M="$CK/qwen3p5-9b-hi-cpt-sft" ;;
  hi-base-sft) M="$CK/qwen3p5-9b-hi-base-sft" ;;
  *) echo "unknown RUN=$RUN"; exit 1 ;;
esac
EV=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi/$RUN
echo "RUN=$RUN MODEL=$M"

python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
  --tasks idiomce --idiomce_path "$REPO/data/eval/hi/idiomce_hi.jsonl" \
  --gen_batch_size 8 --no_judge --chat \
  --output_dir "$EV/idiomce_chat" || { echo "[err] idiomce gen failed"; exit 1; }

eval "$(grep -E '^export LLAMA_API_KEY=' "$HOME/.bashrc" | tail -1)"
export OPENAI_API_KEY="${LLAMA_API_KEY:?LLAMA_API_KEY missing}"
export OPENAI_API_BASE="https://api.llama.com/experimental/compat/openai/v1"
python -m culture.evaluation.judge_idiomce \
  --eval_root "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi" \
  --runs "$RUN/idiomce_chat" \
  --judge_model gpt-5-4-mini-genai-responses --provider openai --batch_size 20

echo "IDIOMCE_CHAT_DONE $RUN"
