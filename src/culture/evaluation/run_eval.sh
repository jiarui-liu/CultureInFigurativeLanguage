#!/usr/bin/env bash
# Evaluate the Hindi base vs. continued-pretrained checkpoints on the four
# Hindi language/culture/idiom benchmarks, then print the base->CPT delta.
#
# Requires: transformers, torch, datasets, openai, tenacity (+ HF_TOKEN for MILU,
# OPENAI_API_KEY for the IdiomCE judge). Edit the paths below for your cluster.
set -euo pipefail

# --- Checkpoints -----------------------------------------------------------
BASE_MODEL=${BASE_MODEL:-/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B}
CPT_MODEL=${CPT_MODEL:-/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt}

# --- Local eval data (see README for schemas & download) -------------------
DATA_DIR=${DATA_DIR:-data/eval/hi}
MABL_PATH=${MABL_PATH:-$DATA_DIR/mabl_hi.csv}
GLOBAL_PIQA_PATH=${GLOBAL_PIQA_PATH:-$DATA_DIR/global_piqa_hi.tsv}
IDIOMCE_PATH=${IDIOMCE_PATH:-$DATA_DIR/idiomce_hi.jsonl}
# MILU pulled from HF (ai4bharat/MILU, needs HF_TOKEN) unless MILU_PATH is set.
MILU_ARGS=()
[[ -n "${MILU_PATH:-}" ]] && MILU_ARGS+=(--milu_path "$MILU_PATH")
[[ -n "${MILU_FEWSHOT_PATH:-}" ]] && MILU_ARGS+=(--milu_fewshot_path "$MILU_FEWSHOT_PATH")

JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o}
OUT_DIR=${OUT_DIR:-results/hi}
TASKS=${TASKS:-mabl,milu,global_piqa,idiomce}

run() {
  local model=$1 name=$2
  echo "=== Evaluating $name ($model) ==="
  python -m culture.evaluation.run_eval \
    --model_path "$model" \
    --run_name "$name" \
    --tasks "$TASKS" \
    --mabl_path "$MABL_PATH" \
    --global_piqa_path "$GLOBAL_PIQA_PATH" \
    --idiomce_path "$IDIOMCE_PATH" \
    "${MILU_ARGS[@]}" \
    --num_fewshot 0 \
    --milu_num_fewshot 5 \
    --judge_model "$JUDGE_MODEL" \
    --output_dir "$OUT_DIR/$name"
}

run "$BASE_MODEL" base
run "$CPT_MODEL" cpt

echo "=== base -> CPT delta ==="
python -m culture.evaluation.compare_results \
  --base "$OUT_DIR/base/summary.json" \
  --cpt  "$OUT_DIR/cpt/summary.json"
