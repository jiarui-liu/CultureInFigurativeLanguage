#!/usr/bin/env bash
# SFT DAG monitor. Prints STATUS lines; when all 4 evals are terminal, writes
# FINAL_REPORT.md and prints ALL_DONE. Also gates on the build manifest.
# Safe to run repeatedly (idempotent).
set -uo pipefail
IT=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning
cd "$IT"
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
source logs/dag_jobs.env

st() { sacct -j "$1" -n -o State%25 2>/dev/null | head -1 | tr -d ' '; }
is_terminal() { case "$1" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE) return 0;; *) return 1;; esac; }

echo "=== $(date) ==="
# Build gate: once build terminal, sanity-check manifests
BST=$(st "$BUILD")
echo "build($BUILD)=$BST"
if is_terminal "$BST" && [[ ! -f logs/BUILD_CHECKED ]]; then
  ok=1
  for L in hi ar; do
    MP=/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_${L}.manifest.json
    if [[ -f "$MP" ]]; then
      tot=$(python3 -c "import json;print(json.load(open('$MP'))['total'])" 2>/dev/null || echo 0)
      echo "manifest $L total=$tot"
      [[ "${tot:-0}" -lt 1000 ]] && ok=0
    else echo "manifest $L MISSING"; ok=0; fi
  done
  if [[ "$BST" != "COMPLETED" || "$ok" -ne 1 ]]; then
    echo "BUILD_BAD: cancelling downstream"
    scancel $T_HI_CPT $T_AR_CPT $T_HI_BASE $T_AR_BASE $E_HI_CPT_C $E_HI_CPT_G $E_HI_BASE_C $E_HI_BASE_G $E_AR_CPT $E_AR_BASE 2>/dev/null
    echo "BUILD FAILED — see logs/build-${BUILD}.err" > FINAL_REPORT.md
    touch logs/BUILD_CHECKED logs/ALL_DONE; echo "ALL_DONE"; exit 0
  fi
  touch logs/BUILD_CHECKED
fi

# training states
for j in $T_HI_CPT $T_AR_CPT $T_HI_BASE $T_AR_BASE; do echo "train($j)=$(st $j)"; done

# eval states
alldone=1
IFS=':' read -ra EV <<< "$EVAL_ALL"
for j in "${EV[@]}"; do
  s=$(st "$j"); echo "eval($j)=$s"
  is_terminal "$s" || alldone=0
done

if [[ "$alldone" -eq 1 ]]; then
  echo "all evals terminal -> aggregating"
  python3 aggregate_sft.py || echo "[warn] aggregate failed"
  touch logs/ALL_DONE
  echo "ALL_DONE"
else
  echo "PENDING"
fi
