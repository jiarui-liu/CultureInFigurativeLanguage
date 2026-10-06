#!/bin/bash
# Evaluate ONE 9B checkpoint on the NEW Chinese / Hindi culture benchmarks
# (built by culture.evaluation.build_culture_mc).
#
# Why: the paper measured Chinese culture with CCPM alone and Hindi culture with a
# 100-item Global-PIQA slice, while Arabic had five benchmarks. These tasks take zh
# from 1 to 3 and hi from 1 to 3, and add the first item-matched cross-language
# control (the parallel4 sets are the same 103 items in zh, hi and ar).
#
#   sbatch eval_newculture.slurm zh culture
#   sbatch eval_newculture.slurm hi unfiltered-matched
#
# Writes to a staging dir and merges into the arm's summary.json, because run_eval
# rewrites summary.json with only the tasks of the invocation.
#SBATCH --job-name=newcult
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/newcult-%x-%j.out
#SBATCH --error=logs/newcult-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
L="${1:?usage: sbatch eval_newculture.slurm <zh|hi> <run>}"
RUN="${2:?usage: sbatch eval_newculture.slurm <zh|hi> <run>}"

CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  base)               M=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B;;
  cpt)                M="$CK/qwen3p5-9b-$L-cpt";;
  unfiltered)         M="$CK/qwen3p5-9b-$L-cpt-unfiltered";;
  unfiltered-matched) M="$CK/qwen3p5-9b-$L-cpt-unfiltered-matched";;
  untagged)           M="$CK/qwen3p5-9b-$L-cpt-untagged";;
  culture)            M="$CK/qwen3p5-9b-$L-cpt-culture";;
  culturenotes)       M="$CK/qwen3p5-9b-$L-cpt-culturenotes";;
  *) echo "bad RUN: $RUN"; exit 1;;
esac
[ -f "$M/config.json" ] || { echo "no checkpoint at $M"; exit 1; }

case "$L" in
  zh) NAMES="global_piqa_zh global_piqa_zh_cultural global_piqa_zh_parallel4";;
  hi) NAMES="global_piqa_hi global_piqa_hi_cultural global_piqa_hi_parallel4 parambench_hi_culture parambench_hi_other";;
  *) echo "bad LANG: $L"; exit 1;;
esac

MC=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc
TASKS=""
for n in $NAMES; do
  [ -f "$MC/$n.jsonl" ] || { echo "missing $MC/$n.jsonl -- run build_culture_mc first"; exit 1; }
  TASKS="${TASKS:+$TASKS,}jsonl:$n"
done
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/$L/$RUN
STAGE="$OUT/_newcult_stage"
mkdir -p "$STAGE"

python -m culture.evaluation.run_eval \
  --model_path "$M" --run_name "$RUN" \
  --tasks "$TASKS" \
  --mc_jsonl_dir "$MC" \
  --num_fewshot 0 \
  --batch_size 8 \
  --output_dir "$STAGE" || { echo "eval FAILED, arm dir untouched"; exit 1; }

for n in $NAMES; do cp "$STAGE/$n.json" "$OUT/$n.json" || exit 1; done
python3 - "$OUT" $NAMES <<'PY'
import json, os, sys
out, names = sys.argv[1], sys.argv[2:]
sp = os.path.join(out, "summary.json")
summary = json.load(open(sp)) if os.path.exists(sp) else {"tasks": {}}
summary.setdefault("tasks", {})
for n in names:
    summary["tasks"][n] = json.load(open(os.path.join(out, n + ".json")))["metrics"]
json.dump(summary, open(sp, "w"), ensure_ascii=False, indent=2)
print("summary.json now holds:", " ".join(sorted(summary["tasks"])))
PY
rm -rf "$STAGE"
echo "DONE $L/$RUN -> $OUT"
