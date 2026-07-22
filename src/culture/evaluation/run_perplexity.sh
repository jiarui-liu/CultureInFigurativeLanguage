#!/usr/bin/env bash
# Dimension 1 (language modeling): PPL + bits-per-byte on both checkpoints, for
# the Hindi held-out corpus (adaptation) and WikiText-103 (English retention).
# Lower is better; expect CPT << base on the Hindi corpus, CPT ~= base on WikiText.
#
# CONTAMINATION: HELDOUT_PATH must be a slice EXCLUDED from CPT training (reserve
# it before training, or use an independent Hindi corpus). See perplexity.py.
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B}
CPT_MODEL=${CPT_MODEL:-/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt}
OUT_DIR=${OUT_DIR:-results/hi}
HELDOUT_PATH=${HELDOUT_PATH:-data/eval/hi/hi_proverbs_heldout.jsonl}   # {"text": ...} per line
TEXT_FIELD=${TEXT_FIELD:-text}

run() {  # $1 model  $2 name
  local model=$1 name=$2
  echo "=== [$name] Hindi held-out (adaptation) ==="
  python -m culture.evaluation.perplexity \
    --model_path "$model" --run_name "$name" \
    --data_path "$HELDOUT_PATH" --text_field "$TEXT_FIELD" \
    --output_dir "$OUT_DIR/$name/ppl_hi"

  echo "=== [$name] WikiText-103 (retention) ==="
  python -m culture.evaluation.perplexity \
    --model_path "$model" --run_name "$name" \
    --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
    --output_dir "$OUT_DIR/$name/ppl_wikitext"
}

run "$BASE_MODEL" base
run "$CPT_MODEL"  cpt

echo
echo "Done. PPL/BPB JSON under $OUT_DIR/{base,cpt}/ppl_{hi,wikitext}/perplexity.json"
