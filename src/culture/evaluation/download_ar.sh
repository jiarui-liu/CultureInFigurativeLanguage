#!/usr/bin/env bash
# Download the Arabic evaluation benchmarks into data/eval/ar/raw/.
#
# Dimension 3 (idiom / figurative)   Dimension 4 (cultural competence)
#   kinayat_cloze  <- the Arabic ChID  arabculture, arabic_cultural_qa, arabicmmlu,
#   kinayat_meaning                    global_piqa_ar, alyah, dzirieval,
#   ar_figurative  <- alyah+dziri      global_piqa_ar_parallel (control)
# Dimension 2 (PPL/BPB) probes are built separately by build_ar_probes.py.
#
# WHY curl AND NOT `hf download` / `load_dataset`:
# every LFS/Xet-backed file on the Hub stalls at 0 bytes behind this proxy (the
# xet protocol is blocked). Plain HTTPS against `resolve/main/<path>` works. If
# you insist on the python client, export HF_HUB_DISABLE_XET=1 first.
#
# None of these repos is gated, so no HF_TOKEN is required.
#
# NOT downloaded, and NOT for overlap reasons — these fail on their own merits:
#   Renad10/Absher-Benchmark    the gold answer is concatenated into the prompt string
#                               in ~10% of rows, 58 missing golds, 91% position bias,
#                               and the answer column mixes A/B/C/D with أ/ب/ج/د
#   ahmed02mk/amthal-hassaniya  instruction/output generation, no options -> not
#                               log-likelihood scorable
#   UBC-NLP/Jawaher-benchmark   free-text explanation, no options; and its gold
#                               explanation is exactly what we inject into the corpus,
#                               so a judge would score memorised text
#
# Sharing idioms with the training KB is NOT a reason to drop a benchmark: that is
# the knowledge CPT is supposed to inject, the same way MMLU overlaps Wikipedia.
# What matters is whether the evaluation ITEM leaks. Measured: the tagger emits only
# meanings/entities/region, never `examples`, so 0 of 298 Kinayat stems appear in a
# real tagged shard. Kinayat is downloaded and used. See plans/... §5.
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

echo "=== 8. menaattia/Kinayat — the Arabic ChID (150 cloze + 325 meaning items) ==="
hf_get menaattia/Kinayat "Arabic_Idioms%20-%20Kinayat_test.csv" "$DATA_DIR/kinayat_test.csv"

echo
echo "=== Done. Verify with: ==="
echo "  PYTHONPATH=src python -c \"from culture.evaluation.tasks_ar import LOADERS_AR as L;"
echo "  [print(k, len(f(ar_data_dir='$DATA_DIR').examples)) for k,f in L.items()]\""
echo "Expected: kinayat_cloze 150 | kinayat_meaning 325 | ar_figurative 314 |"
echo "          arabculture 3471 | arabic_cultural_qa 2000 | arabicmmlu 10529 |"
echo "          global_piqa_ar 1099 | global_piqa_ar_parallel 309 | alyah 1173 | dzirieval 950"
