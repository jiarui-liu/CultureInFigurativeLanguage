#!/usr/bin/env bash
# Download the Arabic evaluation benchmarks into data/eval/ar/raw/.
#
# Dimension 3 (figurative)          Dimension 4 (cultural competence)
#   ar_figurative  <- alyah+dziri     arabculture, arabic_cultural_qa, arabicmmlu,
#                                     global_piqa_ar, alyah, dzirieval,
#                                     global_piqa_ar_parallel (control)
# Dimension 2 (PPL/BPB) probes are built separately by build_ar_probes.py.
#
# WHY curl AND NOT `hf download` / `load_dataset`:
# every LFS/Xet-backed file on the Hub stalls at 0 bytes behind this proxy (the
# xet protocol is blocked). Plain HTTPS against `resolve/main/<path>` works. If
# you insist on the python client, export HF_HUB_DISABLE_XET=1 first.
#
# None of these repos is gated, so no HF_TOKEN is required.
#
# NOT downloaded, on purpose — CONTAMINATED BY OUR OWN TRAINING KB:
#   menaattia/Kinayat              314/325 test items are in data/idioms/ar (96.6%)
#   UBC-NLP/Jawaher-benchmark      199/200 test items (99.5%)
#   Renad10/Absher-Benchmark       81/83 + 408/478 (also: gold leaked into prompt,
#                                  mixed Latin/Arabic answer letters, 58 missing golds,
#                                  91% position bias)
#   ahmed02mk/amthal-hassaniya     319/319 (100%)
# Kinayat is the only true ChID-style Arabic idiom cloze in existence; if you want
# it as a labelled *memorization ceiling* rather than a benchmark, fetch it by hand
# and report it separately. See plans/arabic_pipeline_plan.md §5.
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/eval/ar/raw}
mkdir -p "$DATA_DIR"/{arabculture,global_piqa,global_piqa_parallel}

hf_get() {  # hf_get <repo> <path-in-repo> <dest>
  local url="https://huggingface.co/datasets/$1/resolve/main/$2"
  if [[ -s "$3" ]]; then echo "  cached  $3"; return 0; fi
  if curl -sfL --max-time 900 "$url" -o "$3"; then
    echo "  ok      $3 ($(du -h "$3" | cut -f1))"
  else
    echo "  FAILED  $1/$2" >&2; return 1
  fi
}

echo "=== 1. MBZUAI/ArabCulture (3,482 rows, 13 countries; 11 flagged should_discard) ==="
for c in Algeria Egypt Jordan KSA Lebanon Libya Morocco Palestine Sudan Syria Tunisia UAE Yemen; do
  hf_get MBZUAI/ArabCulture "$c/test-00000-of-00001.parquet" "$DATA_DIR/arabculture/$c.parquet"
done

echo "=== 2. QCRI/ArabicCulturalQA (2,000 questions x 6 dialect variants = 12,000) ==="
hf_get QCRI/ArabicCulturalQA mcq/test.jsonl "$DATA_DIR/acqa_test.jsonl"

echo "=== 3. MBZUAI/ArabicMMLU (14,455 test rows; 10,529 in the 13 Arab-region subjects) ==="
hf_get MBZUAI/ArabicMMLU All/test.csv "$DATA_DIR/arabicmmlu_test.csv"

echo "=== 4. Global-PIQA, 13 Arabic varieties (non-parallel, culturally grounded) ==="
for v in acm_arab acq_arab aeb_arab afb_arab apc_arab_jord apc_arab_leba \
         apc_arab_pale apc_arab_syri arb_arab arq_arab ars_arab ary_arab arz_arab; do
  hf_get mrlbenchmarks/global-piqa-nonparallel "data/nonparallel_$v.tsv" \
         "$DATA_DIR/global_piqa/nonparallel_$v.tsv"
done

echo "=== 5. Global-PIQA parallel arb/ary/arz (CONTROL: culture-agnostic physics) ==="
for v in arb ary arz; do
  hf_get mrlbenchmarks/global-piqa-parallel "data/parallel_${v}_arab.tsv" \
         "$DATA_DIR/global_piqa_parallel/parallel_${v}_arab.tsv"
done

echo "=== 6. tiiuae/alyah-emirati-benchmark (1,173 items; 214 figurative) ==="
hf_get tiiuae/alyah-emirati-benchmark data/test-00000-of-00001.parquet "$DATA_DIR/alyah.parquet"

echo "=== 7. touati-kamel/DziriEval (1,000 rows -> 950 unique questions; 100 figurative) ==="
hf_get touati-kamel/DziriEval dzirieval.jsonl "$DATA_DIR/dzirieval.jsonl"

echo
echo "=== Done. Verify with: ==="
echo "  PYTHONPATH=src python -c \"from culture.evaluation.tasks_ar import LOADERS_AR as L;"
echo "  [print(k, len(f(ar_data_dir='$DATA_DIR').examples)) for k,f in L.items()]\""
echo "Expected: ar_figurative 314 | arabculture 3471 | arabic_cultural_qa 2000 |"
echo "          arabicmmlu 10529 | global_piqa_ar 1099 | global_piqa_ar_parallel 309 |"
echo "          alyah 1173 | dzirieval 950"
