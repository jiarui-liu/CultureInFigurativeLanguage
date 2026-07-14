# Continue Pretraining Data — Progress Report

Last updated: 2026-07-14

## Overview

The continue pretraining pipeline has three stages per language:
1. **Document filtering** — scan C4/mC4 to find documents containing idioms (Aho-Corasick matching)
2. **Deduplication & frequency capping** — cap per-idiom doc count at 10K, remove near-duplicates
3. **Idiom metadata tagging** — append structured meaning annotations to each document

---

## Chinese (zh)

### Idiom dataset
- **31,155** Chinese chengyu with figurative/literal meanings
- Source: merged from multiple dictionaries, LLM-formatted
- File: `culture/data/idioms/zh/idioms_merged_llm_formatted.jsonl`

### Stage 1: Document filtering — COMPLETED
- **Completed**: 2026-03-17
- Corpus: mC4 Chinese subset (`allenai/c4`, `zh`)
- Total documents scanned: **54,542,308**
- Documents matched (contain >= 1 idiom): **17,715,432** (32.48%)
- Run mode: `--indices-only` (saved matching doc indices, not actual text)
- SLURM job: `run_download_mc4_zh.sh` (CPU partition, 64GB RAM, 4 CPUs, ~13h runtime)
- Output files:
  - `culture/data/mc4_filtered/doc_indices_zh.json` (166 MB) — list of 17.7M matching doc indices
  - `culture/data/mc4_filtered/idiom_doc_counts_zh.json` (754 KB) — per-idiom doc frequency
  - `culture/data/mc4_filtered/filtering_zh_summary.json`

### Stage 2: Deduplication & frequency capping — NOT STARTED
- Need to load `idiom_doc_counts_zh.json`, cap each idiom to 10K docs, select high-quality subset for over-represented idioms, deduplicate near-duplicate documents.

### Stage 3: Idiom metadata tagging — NOT STARTED
- Need to extract actual document text for the selected indices, then append idiom meaning annotations (figurative + literal + classical source) to each document.

### Notes
- 32.48% match rate is reasonable — Chinese chengyu are 4-character fixed phrases, so false positives are relatively low.
- No actual document text (`.json.gz` files) has been extracted yet; only index arrays exist.

---

## English (en)

### Idiom dataset
- **32,080** English idioms (raw, LLM-formatted)
- After trivial-idiom filtering (removing single common words, non-figurative phrases): **21,278** figurative-only idioms
- Files:
  - `culture/data/idioms/en/idioms_merged_llm_formatted.jsonl` (32,080 entries)
  - `culture/data/idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl` (21,278 entries)
- Trivial filtering script: `src/culture/training/mC4/filter_trivial_idioms_en.py` (uses LLM + heuristics)

### Stage 1: Document filtering — COMPLETED
- **Completed**: 2026-03-17
- Corpus: C4 English (`allenai/c4`, `en`)
- Total documents scanned: **364,868,892**
- Documents matched: **313,941,010** (86.04%)
- Run mode: `--indices-only`
- SLURM job: `run_download_mc4_en.sh` (CPU partition, 64GB RAM, 4 CPUs, ~26h runtime)
- Output files:
  - `culture/data/mc4_filtered/doc_indices_en.json` (3.2 GB) — list of 314M matching doc indices
  - `culture/data/mc4_filtered/idiom_doc_counts_en.json` (509 KB)
  - `culture/data/mc4_filtered/filtering_en_summary.json`

### Stage 2: Deduplication & frequency capping — NOT STARTED

### Stage 3: Idiom metadata tagging — NOT STARTED

### Known issue: very high English match rate (86%)
- 86% of C4 English docs match at least one idiom — this is suspiciously high.
- Likely cause: many English idioms contain common everyday words/phrases (e.g., "hand", "heart", "time", "break the ice"), which match documents that use these words literally rather than figuratively.
- The `filter_trivial_idioms_en.py` script already reduced the idiom list from 32K to 21K, but the remaining idioms still include many short/common phrases.
- **Action needed**: Investigate `idiom_doc_counts_en.json` to identify which idioms dominate the match count. Consider (a) more aggressive idiom filtering (longer phrases only), (b) context-aware matching (not just substring), or (c) relying entirely on the frequency capping in Stage 2 to limit the over-matched idioms to 10K docs each.

---

## Summary Table

| | Chinese (zh) | English (en) |
|---|---|---|
| Idiom count | 31,155 | 21,278 (figurative-only) |
| Corpus | mC4 zh (54.5M docs) | C4 en (364.9M docs) |
| Stage 1: Filtering | DONE (17.7M matched, 32.5%) | DONE (313.9M matched, 86.0%) |
| Stage 2: Dedup & cap | NOT STARTED | NOT STARTED |
| Stage 3: Metadata tag | NOT STARTED | NOT STARTED |
| Actual text extracted | No | No |

---

## Next Steps

1. **Investigate English match rate** — analyze `idiom_doc_counts_en.json` to understand the distribution and decide whether additional idiom filtering is needed before proceeding.
2. **Implement Stage 2** — write or adapt the dedup/cap script that takes doc indices + idiom doc counts, caps at 10K docs/idiom, and selects quality subset.
3. **Extract document text** — re-stream C4/mC4, pull only the selected indices, save as `.json.gz` chunks.
4. **Implement Stage 3** — tag extracted documents with idiom metadata (figurative meaning, literal meaning, entities, source citations for Chinese).
5. **Extend to Hindi and Arabic** — compile idiom datasets, then run the same pipeline on mC4 hi/ar subsets.
