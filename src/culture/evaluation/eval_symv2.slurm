#!/bin/bash
# Evaluate ONE already-trained 9B checkpoint on the symbolism-probe v2 letter task only.
#
# Why this exists: the eval task lists in docs/plans/bidirectional_transfer_plan.md §3.1b
# switched from `symbolism_{L}` (v1, continuation-scored) to `symbolism_v2_{L}_letter`, but
# five checkpoints were evaluated before that switch and only have the v1 file. v2 is also
# ~2x larger (ar 98 / hi 155 / zh 206 items vs 46 / 67 / 111).
#
#   sbatch eval_symv2.slurm ar culture
#   sbatch eval_symv2.slurm zh untagged
#
# Writes into a scratch subdir first, then copies the task JSON up and MERGES the metric
# into the arm's existing summary.json. run_eval rewrites summary.json from scratch with
# only the tasks of that invocation, so pointing --output_dir straight at the arm dir
# would silently destroy the other 12 task entries.
#SBATCH --job-name=symv2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/symv2-%x-%j.out
#SBATCH --error=logs/symv2-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
L="${1:?usage: sbatch eval_symv2.slurm <ar|hi|zh> <run>}"
RUN="${2:?usage: sbatch eval_symv2.slurm <ar|hi|zh> <run>}"

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  base)         M=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B;;
  cpt)          M="$CK/qwen3p5-9b-$L-cpt";;
  unfiltered)   M="$CK/qwen3p5-9b-$L-cpt-unfiltered";;
  untagged)     M="$CK/qwen3p5-9b-$L-cpt-untagged";;
  culture)      M="$CK/qwen3p5-9b-$L-cpt-culture";;
  culturenotes) M="$CK/qwen3p5-9b-$L-cpt-culturenotes";;
  *) echo "bad RUN: $RUN"; exit 1;;
esac
[ -f "$M/config.json" ] || { echo "no checkpoint at $M"; exit 1; }

TASK="symbolism_v2_${L}_letter"
MC=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc
[ -f "$MC/$TASK.jsonl" ] || { echo "missing $MC/$TASK.jsonl"; exit 1; }
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/$L/$RUN
STAGE="$OUT/_symv2_stage"
mkdir -p "$STAGE"

python -m culture.evaluation.run_eval \
  --model_path "$M" --run_name "$RUN" \
  --tasks "jsonl:$TASK" \
  --mc_jsonl_dir "$MC" \
  --num_fewshot 0 \
  --batch_size 8 \
  --output_dir "$STAGE" || { echo "eval FAILED, arm dir untouched"; exit 1; }

cp "$STAGE/$TASK.json" "$OUT/$TASK.json"
python3 - "$OUT" "$TASK" <<'PY'
import json, os, sys
out, task = sys.argv[1], sys.argv[2]
sp = os.path.join(out, "summary.json")
summary = json.load(open(sp)) if os.path.exists(sp) else {"tasks": {}}
summary.setdefault("tasks", {})[task] = json.load(open(os.path.join(out, task + ".json")))["metrics"]
json.dump(summary, open(sp, "w"), ensure_ascii=False, indent=2)
print("summary.json now holds:", " ".join(sorted(summary["tasks"])))
PY
rm -rf "$STAGE"
echo "DONE $L/$RUN $TASK -> $OUT"
