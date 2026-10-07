#!/usr/bin/env bash
# Upload the two benchmarks this paper releases to a Hugging Face dataset repo.
#
#   IdiomAtlas-MC   idiom -> figurative meaning, 4-way, Hindi/Chinese/Arabic,
#                   split by whether the idiom occurs in the training corpus
#   Symbolism probe what a shared entity symbolises in a culture, with an
#                   English-association distractor ("lure") to measure Anglocentric
#                   defaulting
#
# NOT uploaded: global_piqa_* and parambench_hi_* are third-party datasets that we
# only reformatted for our harness; redistributing them here would be wrong. Point
# users at the originals instead.
#
# IdiomCulture is not included: as of this writing it is design + code only
# (docs/plans/idiomculture_benchmark_plan.md), no items have been generated.
#
# Usage:
#   ./upload_benchmarks_hf.sh                      # dry run, builds and shows the tree
#   ./upload_benchmarks_hf.sh --push               # actually upload
#   REPO=you/your-repo ./upload_benchmarks_hf.sh --push
#
# Requires: `hf auth login` (or HF_TOKEN set) and network access to huggingface.co.
set -euo pipefail

REPO="${REPO:-Jerry9999/IdiomAtlas-benchmarks}"
SRC="${SRC:-/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc}"
STAGE="${STAGE:-/tmp/hf_benchmark_upload}"
PUSH=0
[ "${1:-}" = "--push" ] && PUSH=1

command -v hf >/dev/null 2>&1 || { echo "ERROR: the 'hf' CLI is not on PATH"; exit 1; }
[ -d "$SRC" ] || { echo "ERROR: source dir not found: $SRC"; exit 1; }

rm -rf "$STAGE"; mkdir -p "$STAGE/idiomatlas_mc" "$STAGE/symbolism_probe"

# ---- IdiomAtlas-MC -----------------------------------------------------------
# *_seen  : the idiom occurs in the idiom-annotated corpus
# *_unseen: it occurs in no document we scanned
# the un-suffixed file is the union
for L in ar hi zh; do
  for S in "" _seen _unseen; do
    f="$SRC/idiomatlas_mc_${L}${S}.jsonl"
    [ -f "$f" ] && cp "$f" "$STAGE/idiomatlas_mc/" || echo "  (skip missing $(basename "$f"))"
  done
done

# ---- Symbolism probe ---------------------------------------------------------
# v2 is the version the paper reports; v1 is kept for reproducibility of the
# earlier 9B runs. "_letter" presents options as A/B/C/D, which is the format the
# paper scores, because log-likelihood over the bare options tracks their own prior.
for L in ar hi zh; do
  for V in symbolism_${L} symbolism_${L}_letter symbolism_v2_${L} symbolism_v2_${L}_letter; do
    f="$SRC/${V}.jsonl"
    [ -f "$f" ] && cp "$f" "$STAGE/symbolism_probe/"
  done
done

# ---- dataset card ------------------------------------------------------------
cat > "$STAGE/README.md" <<'CARD'
---
license: cc-by-4.0
language: [hi, zh, ar]
task_categories: [multiple-choice]
tags: [idioms, figurative-language, culture, multilingual]
configs:
  - config_name: idiomatlas_mc
    data_files: idiomatlas_mc/*.jsonl
  - config_name: symbolism_probe
    data_files: symbolism_probe/*.jsonl
---

# IdiomAtlas benchmarks

Two multiple-choice benchmarks released with *Do Idioms Carry Culture?*

Every file is JSON Lines with the same shape:

| field | meaning |
|---|---|
| `qid` | stable item id |
| `context` | the question stem, in the target language |
| `options` | list of answer strings |
| `gold` | index into `options` |
| `meta` | per-benchmark extras (below) |

## IdiomAtlas-MC (`idiomatlas_mc/`)

Given an idiom, pick its figurative meaning from four candidates. Hindi, Chinese and
Arabic. The gold option is the first figurative meaning in the IdiomAtlas knowledge
base; distractors are meanings of *other* idioms of the same language, topically
related to the gold (embedding cosine 0.35–0.75) and of similar length. Items whose
gold or distractor shares surface form with the idiom are removed, which brings a
lexical-overlap baseline to chance (23–24%).

Files are split by corpus exposure:

- `*_seen.jsonl` — the idiom occurs in the idiom-annotated web corpus
- `*_unseen.jsonl` — it occurs in no document we scanned
- `idiomatlas_mc_<lang>.jsonl` — the union

Sizes: 600 seen and 600 unseen per language, except Chinese, where only 78 unseen
items survive because almost every *chengyu* occurs on the web.

**Read the split against an untrained baseline, not in absolute terms.** Seen idioms
are by construction the more frequent and more conventional ones, so an untrained
model already scores higher on them — in Chinese it accounts for most of the gap.

`meta`: `idiom`, `split`, `distractor_idioms`, `distractor_sims`.

## Symbolism probe (`symbolism_probe/`)

Given an entity that two cultures share, pick what it symbolises in the target
culture. One distractor is the **English** association for the same entity (the
"lure"), so the probe measures how often a model answers through an Anglophone lens.
Items were generated from the entity's evidence idioms by one model family and
blind-verified by another; only items both agreed on are kept.

- `symbolism_v2_*` is the version the paper reports; `symbolism_*` is the earlier
  version, kept so the 9B runs reproduce.
- `*_letter` presents options as A/B/C/D. **Use these.** Scoring the bare option
  strings by log-likelihood mostly tracks their own prior and sits near chance.

`meta`: `entity`, `entity_en`, `lure` (index of the English-association option),
and for v1 also `distractors`, `gold_evidence`, `lure_evidence`.

## Not included

`global_piqa_*` and `parambench_hi_*` appear in our evaluation harness but are
third-party datasets; please get them from their original authors.

## Caveats

Items are produced and checked by LLMs from two model families, not by native
speakers, so their errors correlate with what current models believe. Native-speaker
validation of a sample is still outstanding.
CARD

echo "=== staged tree ==="
find "$STAGE" -type f | sed "s#$STAGE#.#" | sort
echo
echo "=== item counts ==="
for f in "$STAGE"/*/*.jsonl; do printf "  %-34s %6d\n" "$(basename "$f")" "$(wc -l < "$f")"; done
echo
if [ "$PUSH" -eq 1 ]; then
  echo "=== creating repo (ok if it already exists) ==="
  hf repo create "$REPO" --repo-type dataset -y 2>/dev/null || true
  echo "=== uploading to $REPO ==="
  hf upload "$REPO" "$STAGE" . --repo-type dataset \
    --commit-message "Add IdiomAtlas-MC and the symbolism probe"
  echo "done: https://huggingface.co/datasets/$REPO"
else
  echo "DRY RUN — nothing uploaded. Re-run with --push to upload to: $REPO"
fi
