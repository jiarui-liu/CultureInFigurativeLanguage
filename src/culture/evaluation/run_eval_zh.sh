#!/usr/bin/env bash
# Evaluate the Chinese base vs. continued-pretrained (chengyu CPT) checkpoints on
# the four Chinese idiom/culture benchmarks, then print the base->CPT delta.
#
# Tasks: chid, chengyu_bench, cmmlu, ccpm (all base-model multiple-choice / cloze,
# scored by log-likelihood — no instruction following, no OpenAI judge).
#
# Requires: transformers, torch, datasets (+ HF_TOKEN only if you hit HF rate
# limits; ChID/CMMLU are not gated). Download data first with download_zh.sh.
set -euo pipefail

# --- Checkpoints -----------------------------------------------------------
BASE_MODEL=${BASE_MODEL:-/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B}
CPT_MODEL=${CPT_MODEL:-/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-zh-cpt}

# --- Local eval data (see download_zh.sh + eval_benchmarks_download.md) -----
DATA_DIR=${DATA_DIR:-data/eval/zh}
# ChID: local jsonl if set, else pulled from HF thu-coai/chid (validation).
CHID_ARGS=()
[[ -n "${CHID_PATH:-}" ]] && CHID_ARGS+=(--chid_path "$CHID_PATH")
# Chengyu-Bench: cloned repo dir + which binary subtask.
CHENGYU_BENCH_DIR=${CHENGYU_BENCH_DIR:-$DATA_DIR/ChengyuBench}
CHENGYU_BENCH_SUBTASK=${CHENGYU_BENCH_SUBTASK:-connotation}
# CCPM: point at the cloned JSONL (confirm the exact file name after git clone).
CCPM_PATH=${CCPM_PATH:-$DATA_DIR/CCPM/data/test.jsonl}
# CMMLU: 16 China-specific subjects by default (loader default); override here.
CMMLU_SUBJECTS=${CMMLU_SUBJECTS:-}
CMMLU_ARGS=()
[[ -n "$CMMLU_SUBJECTS" ]] && CMMLU_ARGS+=(--cmmlu_subjects "$CMMLU_SUBJECTS")

OUT_DIR=${OUT_DIR:-results/zh}
TASKS=${TASKS:-chid,chengyu_bench,cmmlu,ccpm}

run() {
  local model=$1 name=$2
  echo "=== Evaluating $name ($model) ==="
  python -m culture.evaluation.run_eval \
    --model_path "$model" \
    --run_name "$name" \
    --tasks "$TASKS" \
    "${CHID_ARGS[@]}" \
    --chengyu_bench_dir "$CHENGYU_BENCH_DIR" \
    --chengyu_bench_subtask "$CHENGYU_BENCH_SUBTASK" \
    --ccpm_path "$CCPM_PATH" \
    "${CMMLU_ARGS[@]}" \
    --num_fewshot 0 \
    --cmmlu_num_fewshot 5 \
    --output_dir "$OUT_DIR/$name"
}

run "$BASE_MODEL" base
run "$CPT_MODEL" cpt

echo "=== base -> CPT delta ==="
python -m culture.evaluation.compare_results \
  --base "$OUT_DIR/base/summary.json" \
  --cpt  "$OUT_DIR/cpt/summary.json"
