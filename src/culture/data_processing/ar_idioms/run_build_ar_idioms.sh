#!/usr/bin/env bash
# Build the merged Arabic idiom KB -> data/idioms/ar/idioms_merged_llm_formatted.jsonl
#
# Only human-authored sources are fetched (see EXCLUDED_SOURCES in the Python module
# for what is deliberately left out and why). Rows without any meaning are dropped.
# Requires: pip install datasets
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
OUT=${OUT:-"$REPO_ROOT/data/idioms/ar/idioms_merged_llm_formatted.jsonl"}
REPORT=${REPORT:-"$REPO_ROOT/data/idioms/ar/build_report.json"}

echo "=== Sanity: Arabic normalizer ==="
python3 "$REPO_ROOT/src/culture/data_processing/ar_idioms/normalize.py"

echo "=== Sanity: build-script self-test (offline) ==="
python3 "$REPO_ROOT/src/culture/data_processing/ar_idioms/build_ar_idioms.py" --self-test

echo "=== Building Arabic idiom KB ==="
python3 "$REPO_ROOT/src/culture/data_processing/ar_idioms/build_ar_idioms.py" \
  --out "$OUT" \
  --report "$REPORT" \
  "$@"

echo
echo "=== Done ==="
echo "KB:     $OUT"
echo "Report: $REPORT"
echo "Rows:   $(wc -l < "$OUT")"
