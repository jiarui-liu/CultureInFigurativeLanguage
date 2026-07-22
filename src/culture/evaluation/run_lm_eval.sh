#!/usr/bin/env bash
# Dimension 2 (general English, forgetting check) via EleutherAI lm-evaluation-harness,
# plus Dimension 1 English retention (WikiText BPB/PPL).
#
# Runs each task at its canonical shot count on BOTH checkpoints so the base->CPT
# delta is apples-to-apples. Expected result: Delta ~= 0 (no catastrophic forgetting).
#
# Requires: pip install lm-eval   (a.k.a. lm_eval; EleutherAI/lm-evaluation-harness)
# HumanEval executes model-generated code -> needs --confirm_run_unsafe_code.
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B}
CPT_MODEL=${CPT_MODEL:-/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt}
OUT_DIR=${OUT_DIR:-results/hi}
BATCH_SIZE=${BATCH_SIZE:-auto}

# task:num_fewshot  (MMLU 0-shot, BoolQ 0-shot, GSM8K 8-shot CoT, HumanEval 0-shot)
TASKS=(
  "mmlu:0"
  "boolq:0"
  "gsm8k_cot:8"
  "humaneval:0"
)
# WikiText for Dim 1 English retention (BPB/PPL); scored, no few-shot.
WIKITEXT_TASK=${WIKITEXT_TASK:-wikitext}

model_args() {  # $1 = checkpoint path
  echo "pretrained=$1,dtype=bfloat16,trust_remote_code=True,attn_implementation=sdpa"
}

run_task() {  # $1 model_path  $2 run_name  $3 task  $4 num_fewshot  $5 extra
  local model=$1 name=$2 task=$3 nfs=$4 extra=${5:-}
  echo "=== [$name] $task (${nfs}-shot) ==="
  # shellcheck disable=SC2086
  lm_eval --model hf \
    --model_args "$(model_args "$model")" \
    --tasks "$task" \
    --num_fewshot "$nfs" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUT_DIR/$name/lm_eval/$task" \
    $extra
}

run_all() {  # $1 model_path  $2 run_name
  local model=$1 name=$2
  for spec in "${TASKS[@]}"; do
    local task="${spec%%:*}" nfs="${spec##*:}" extra=""
    [[ "$task" == "humaneval" ]] && extra="--confirm_run_unsafe_code"
    run_task "$model" "$name" "$task" "$nfs" "$extra"
  done
  # Dim 1 English retention (WikiText reports word_perplexity + bits_per_byte).
  run_task "$model" "$name" "$WIKITEXT_TASK" 0 ""
}

run_all "$BASE_MODEL" base
run_all "$CPT_MODEL"  cpt

echo
echo "Done. Results under $OUT_DIR/{base,cpt}/lm_eval/<task>/."
echo "Compare acc/EM/pass@1 per task (Dim 2) and wikitext bits_per_byte/word_perplexity (Dim 1 retention)."
