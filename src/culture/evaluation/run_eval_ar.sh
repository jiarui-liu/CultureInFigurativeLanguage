#!/usr/bin/env bash
# Evaluate the Arabic base vs. continued-pretrained (amthal CPT) checkpoints and
# print the base->CPT delta.
#
#   Dimension 2  Arabic language modeling      ar_fineweb2_heldout, ar_wiki_heldout (BPB)
#   Dimension 3  figurative understanding      ar_figurative                  (314, clean)
#   Dimension 4  cultural competence           arabculture, arabic_cultural_qa,
#                                              arabicmmlu, global_piqa_ar, alyah, dzirieval
#   Control      culture-agnostic reasoning    global_piqa_ar_parallel
#
# All tasks are base-model multiple-choice scored by log-likelihood — no
# instruction following, no OpenAI judge (unlike Hindi IdiomCE).
#
# READ acc_norm, NOT acc, for the continuation-scored tasks. Alyah's gold option
# is the longest one 57.5% of the time (chance 25%), so raw summed log-prob
# measures the length prior. run_eval.py already reports `primary` = acc_norm for
# continuation tasks and acc for letter tasks; use that field.
#
# Prereqs: bash download_ar.sh && python build_ar_probes.py (see below).
set -euo pipefail

# --- Checkpoints -----------------------------------------------------------
BASE_MODEL=${BASE_MODEL:-/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B}
CPT_MODEL=${CPT_MODEL:-/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-ar-cpt}

# --- Data ------------------------------------------------------------------
AR_DATA_DIR=${AR_DATA_DIR:-data/eval/ar/raw}     # written by download_ar.sh
PROBE_DIR=${PROBE_DIR:-data/eval/ar}             # written by build_ar_probes.py
# Optional belt-and-braces decontamination: drop eval items whose stem contains a
# KB idiom surface (measured: 2/1173 on Alyah, 0/314 on ar_figurative).
AR_KB_PATH=${AR_KB_PATH:-data/idioms/ar/idioms_merged_llm_formatted.jsonl}
KB_ARGS=(); [[ -f "$AR_KB_PATH" ]] && KB_ARGS=(--ar_kb_path "$AR_KB_PATH")

OUT_DIR=${OUT_DIR:-results/ar}
TASKS=${TASKS:-kinayat_cloze,kinayat_meaning,ar_figurative,arabculture,arabic_cultural_qa,arabicmmlu,global_piqa_ar,alyah,dzirieval,global_piqa_ar_parallel}
# 'all' -> the full cross-dialect axis (msa+5 dialects+english, 12,000 items).
ACQA_DIALECTS=${ACQA_DIALECTS:-msa}
BATCH_SIZE=${BATCH_SIZE:-8}

run_mc() {
  local model=$1 name=$2
  echo "=== [MC] Evaluating $name ($model) ==="
  python -m culture.evaluation.run_eval \
    --model_path "$model" \
    --run_name "$name" \
    --tasks "$TASKS" \
    --ar_data_dir "$AR_DATA_DIR" \
    --acqa_dialects "$ACQA_DIALECTS" \
    "${KB_ARGS[@]}" \
    --ar_num_fewshot 0 \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUT_DIR/$name"
}

run_ppl() {
  # perplexity.py scores ONE corpus per invocation and always writes
  # <output_dir>/perplexity.json, so give each probe its own output dir.
  local model=$1 name=$2
  for probe in ar_fineweb2_heldout ar_wiki_heldout; do
    echo "=== [PPL/BPB] $name / $probe ($model) ==="
    python -m culture.evaluation.perplexity \
      --model_path "$model" \
      --run_name "$name" \
      --data_path "$PROBE_DIR/$probe.jsonl" \
      --text_field text \
      --output_dir "$OUT_DIR/$name/ppl_${probe#ar_}"
  done
}

for pair in "$BASE_MODEL base" "$CPT_MODEL cpt"; do
  set -- $pair
  run_ppl "$1" "$2"
  run_mc  "$1" "$2"
done

echo "=== base -> CPT delta ==="
python -m culture.evaluation.compare_results \
  --base "$OUT_DIR/base/summary.json" \
  --cpt  "$OUT_DIR/cpt/summary.json"
