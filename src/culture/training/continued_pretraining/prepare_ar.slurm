#!/bin/bash
# DAG step 2: reshard the augmented tagged shards into train_ar, and measure the
# exact Qwen token budget (for the matched baseline). Depends (afterany) on the
# filter job -- afterany NOT afterok because filter_and_tag_ar.py may segfault at
# interpreter teardown after flushing outputs (runbook gotcha); we judge success
# by filter_report.json, not the exit code.
#   sbatch --dependency=afterany:<filter_jobid> prepare_ar.slurm
#SBATCH --job-name=ar-prepare
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/ar/prepare-%j.out
#SBATCH --error=logs/ar/prepare-%j.out
set -uo pipefail

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"

DATA_ROOT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data
SRC="$DATA_ROOT/ar-amthal-cpt/data/fineweb2_par"    # parallel filter output (per-task shards)
OUT="$DATA_ROOT/train_ar"
MODEL=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B

# --- validate the filter output before trusting it (not the exit code) --------
# Parallel filter writes one filter_report_pN.json per array task; sum written
# across all of them. (Segfault-at-teardown may zero a task's exit code but the
# report is flushed first.) Fall back to a single filter_report.json if present.
python - "$SRC" <<'PY'
import glob, json, os, sys
src = sys.argv[1]
reps = sorted(glob.glob(os.path.join(src, "filter_report_*.json"))) \
       or glob.glob(os.path.join(src, "filter_report.json"))
if not reps:
    print(f"FATAL: no filter_report*.json under {src}"); sys.exit(1)
tot_w = tot_s = tot_m = 0
for p in reps:
    r = json.load(open(p))
    tot_w += r.get("written", 0); tot_s += r.get("scanned", 0); tot_m += r.get("matched", 0)
print(f"{len(reps)} reports: scanned={tot_s} matched={tot_m} written={tot_w}")
sys.exit(0 if tot_w > 0 else 2)
PY
[[ $? -eq 0 ]] || { echo "FATAL: filter wrote 0 docs"; exit 1; }

# --- reshard tagged_*.json.gz -> train_*.jsonl --------------------------------
python src/culture/training/continued_pretraining/prepare_data.py \
    --src_dir "$SRC" --out_dir "$OUT"
python src/culture/training/continued_pretraining/prepare_data.py \
    --verify_only --out_dir "$OUT"

# --- measure the EXACT Qwen token budget over train_ar (same counting as
#     build_unfiltered_hi.py: add_special_tokens=False, summed per doc) ---------
python - "$OUT" "$MODEL" <<'PY'
import glob, json, os, sys
from transformers import AutoTokenizer
out_dir, model = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
total = ndocs = 0
batch = []
def flush():
    global total
    if not batch: return
    for ids in tok(batch, add_special_tokens=False)["input_ids"]:
        total += len(ids)
    batch.clear()
for shard in sorted(glob.glob(os.path.join(out_dir, "train_*.jsonl"))):
    with open(shard, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            batch.append(json.loads(line)["text"]); ndocs += 1
            if len(batch) >= 1000: flush()
flush()
# IMPORTANT: write the budget file OUTSIDE the dataset dir. A stray non-.jsonl
# file inside train_ar makes LLaMA-Factory's directory loader fail with
# "ValueError: File types should be identical." (mixed .jsonl/.txt extensions).
budget_path = out_dir.rstrip("/") + ".token_budget.txt"
with open(budget_path, "w") as f:
    f.write(str(total))
print(f"TOKEN_BUDGET docs={ndocs} tokens={total} ({total/1e9:.3f}B) -> {budget_path}")
PY
echo "PREPARE_DONE out=$OUT budget=$(cat "${OUT}.token_budget.txt" 2>/dev/null)"
