#!/usr/bin/env bash
# Download the four Chinese evaluation benchmarks into data/eval/zh/.
#
#   chid          -> HF thu-coai/chid            (also load_dataset("clue","chid"))
#   cmmlu         -> HF haonan-li/cmmlu          (67 subject configs, dev+test)
#   chengyu_bench -> GitHub sofyc/ChengyuBench   (git clone; JSON, raw-file load)
#   ccpm          -> GitHub THUNLP-AIPoet/CCPM   (git clone; JSONL, raw-file load)
#
# ChID + CMMLU also load straight from HuggingFace at eval time, so the HF steps
# below are only needed for offline / pre-download / inspection. Chengyu-Bench and
# CCPM are GitHub-only and MUST be cloned.
#
# HF_TOKEN note: neither ChID nor CMMLU is gated, so no token is required; CMMLU
# is a *script* dataset, so the loader passes trust_remote_code=True (and falls
# back to the parquet mirror lmlmcat/cmmlu on newer `datasets`). Export HF_TOKEN
# only if you hit HF rate limits.
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/eval/zh}
mkdir -p "$DATA_DIR"

echo "=== Prereqs ==="
pip install "huggingface_hub[cli]" datasets >/dev/null 2>&1 || \
  echo "  (install huggingface_hub[cli] + datasets manually if this failed)"

# --- 1. ChID (chengyu cloze) ----------------------------------------------
# IMPORTANT: the HF mirror thu-coai/chid ships NO gold answers in ANY split
# (competition format), so it is not scorable on its own. Use the ORIGINAL
# chujiezheng/ChID-Dataset, which ships passages + a SEPARATE answer file.
echo "=== 1. ChID (chujiezheng/ChID-Dataset — gold-bearing) ==="
if [[ ! -d "$DATA_DIR/ChID-Dataset" ]]; then
  git clone https://github.com/chujiezheng/ChID-Dataset.git "$DATA_DIR/ChID-Dataset" || \
    echo "  ChID clone failed (network?). Retry: git clone https://github.com/chujiezheng/ChID-Dataset $DATA_DIR/ChID-Dataset"
fi
echo "  After clone, locate the passages + answer file and pass BOTH:"
echo "    --chid_path <dev.json/jsonl>  --chid_answer_path <dev_answer.json/csv>"
echo "    (inspect: ls -R $DATA_DIR/ChID-Dataset ; names/subdirs may differ)"

# --- 2. CMMLU (China-specific subjects) -----------------------------------
# Loads live from HF haonan-li/cmmlu at eval time (trust_remote_code=True). To
# pre-download the whole dataset repo (dev + test for all 67 subjects):
echo "=== 2. CMMLU (haonan-li/cmmlu) ==="
huggingface-cli download haonan-li/cmmlu --repo-type dataset \
  --local-dir "$DATA_DIR/cmmlu" || \
  echo "  CMMLU repo download skipped (loads live at eval time anyway)."

# --- 3. Chengyu-Bench (connotation + appropriateness) ---------------------
# GitHub-only; confirm the JSON field/file names after cloning (the loader is
# defensive but the exact schema was unverified — GitHub was unreachable).
echo "=== 3. Chengyu-Bench (sofyc/ChengyuBench) ==="
if [[ ! -d "$DATA_DIR/ChengyuBench" ]]; then
  git clone https://github.com/sofyc/ChengyuBench.git "$DATA_DIR/ChengyuBench" || \
    echo "  Chengyu-Bench clone failed (network?). Retry: git clone https://github.com/sofyc/ChengyuBench $DATA_DIR/ChengyuBench"
fi

# --- 4. CCPM (Chinese Classical Poetry Matching) --------------------------
echo "=== 4. CCPM (THUNLP-AIPoet/CCPM) ==="
if [[ ! -d "$DATA_DIR/CCPM" ]]; then
  git clone https://github.com/THUNLP-AIPoet/CCPM.git "$DATA_DIR/CCPM" || \
    echo "  CCPM clone failed (network?). Retry: git clone https://github.com/THUNLP-AIPoet/CCPM $DATA_DIR/CCPM"
fi

cat <<EOF

=== Done. Expected layout ===
$DATA_DIR/
├── ChID-Dataset/                 # git clone chujiezheng/ChID-Dataset (gold-bearing)
│                                 #   -> --chid_path <dev.json> --chid_answer_path <dev_answer.*>
├── cmmlu/                        # hf download haonan-li/cmmlu -> --cmmlu_dir (local CSV mode)
│                                 #   (<dir>/test/<subject>.csv, <dir>/dev/<subject>.csv)
├── ChengyuBench/                 # git clone sofyc/ChengyuBench  -> --chengyu_bench_dir
└── CCPM/                         # git clone THUNLP-AIPoet/CCPM   -> --ccpm_path <VALID>.jsonl

Confirm exact file names, then point the loaders at them:
  ls -R $DATA_DIR/ChID-Dataset          # find dev.json + dev_answer.* (gold)
  ls    $DATA_DIR/cmmlu/test | head     # <subject>.csv files
  ls    $DATA_DIR/ChengyuBench
  ls    $DATA_DIR/CCPM                  # use valid.jsonl (has answers), NOT test_public.jsonl
EOF
