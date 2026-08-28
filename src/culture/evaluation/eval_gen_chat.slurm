#!/bin/bash
# Chat-template (model-native) generative eval for ONE instruction-tuned checkpoint.
# Instruct models are prompted with their chat template; results go to NEW folders
# (en_gen_chat/, idiomce_chat/) so the raw base-protocol outputs are preserved.
#   sbatch eval_gen_chat.slurm <hi|ar|zh> <hi-cpt-sft|hi-base-sft|ar-cpt-sft|...>
# Runs GSM8K + HumanEval (all langs) and, for hi, IdiomCE generation + Llama judge.
#SBATCH --job-name=gen-chat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen-chat-%x-%j.out
#SBATCH --error=logs/gen-chat-%x-%j.err
#SBATCH --partition=h100

set -uo pipefail
LANG_="${1:?usage: sbatch eval_gen_chat.slurm <hi|ar|zh> <run>}"
RUN="${2:?usage: sbatch eval_gen_chat.slurm <hi|ar|zh> <run>}"

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/.eval_deps:$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  hi-cpt-sft)  M="$CK/qwen3p5-9b-hi-cpt-sft" ;;
  hi-base-sft) M="$CK/qwen3p5-9b-hi-base-sft" ;;
  ar-cpt-sft)  M="$CK/qwen3p5-9b-ar-cpt-sft" ;;
  ar-base-sft) M="$CK/qwen3p5-9b-ar-base-sft" ;;
  zh-cpt-sft)  M="$CK/qwen3p5-9b-zh-cpt-sft" ;;
  zh-base-sft) M="$CK/qwen3p5-9b-zh-base-sft" ;;
  *) echo "unknown RUN=$RUN"; exit 1 ;;
esac
EV=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/$LANG_/$RUN
echo "LANG=$LANG_ RUN=$RUN MODEL=$M"

# --- GSM8K + HumanEval, chat template, NEW folder en_gen_chat ---
echo "### [gen-chat] GSM8K + HumanEval"
python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
  --tasks gsm8k,humaneval --gen_batch_size 16 --chat \
  --output_dir "$EV/en_gen_chat" || echo "[warn] gsm8k/humaneval chat failed"

# --- IdiomCE (Hindi only): chat generation into NEW folder, then Llama judge ---
if [ "$LANG_" = "hi" ]; then
  echo "### [gen-chat] IdiomCE generation"
  python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
    --tasks idiomce --idiomce_path "/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/hi/idiomce_hi.jsonl" \
    --gen_batch_size 8 --no_judge --chat \
    --output_dir "$EV/idiomce_chat" || echo "[warn] idiomce chat gen failed"

  echo "### [gen-chat] IdiomCE judge"
  eval "$(grep -E '^export LLAMA_API_KEY=' "$HOME/.bashrc" | tail -1)"
  export OPENAI_API_KEY="${LLAMA_API_KEY:?LLAMA_API_KEY missing}"
  export OPENAI_API_BASE="https://api.llama.com/experimental/compat/openai/v1"
  python -m culture.evaluation.judge_idiomce \
    --eval_root "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi" \
    --runs "$RUN/idiomce_chat" \
    --judge_model gpt-5-4-mini-genai-responses --provider openai --batch_size 20 \
    || echo "[warn] idiomce judge failed"
fi

echo "GEN_CHAT_DONE $LANG_ $RUN"
