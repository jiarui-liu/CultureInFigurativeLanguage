# Arabic Pipeline — Analyses, Corpora, Evaluation, CPT

**Owner:** automated run. **Started / last updated:** 2026-08-22.
**Status:** ⬜ not started · 🔄 in progress · ✅ done · ⚠️ done with caveats · ❌ dropped

Tracks every task, **every decision and why**, and ends with the exact commands to
download the corpus and run training on another server (§9).

---

## 0. Context / inputs ✅

| Item | Value |
|---|---|
| Arabic idiom KB | `data/idioms/ar/idioms_merged_llm_formatted.jsonl` — **10,386** entries |
| Schema | `idiom, entities, literal_meanings, figurative_meanings, figurative_meanings_en, examples, variety, variety_region, register` (missing → `"NAN"`) |
| Chat LLM | MetaGen Llama API via `meta-autoresearch/code/autoresearch/utils/llm.py`, `METAGEN_API_KEY`, model `openai-gpt-5-4-responses` ✅ verified |
| Embeddings | same module, **`EMBEDDING_API_KEY=$APE_API_KEY`**, `text-embedding-3-small` @512d ✅ verified |
| Python env | `/home/jiaruiliu/.venv-verl/bin/python` (+ installed here: `scikit-learn matplotlib arabic-reshaper python-bidi pyahocorasick`) |
| Plot font | `/usr/share/fonts/google-droid-sans-fonts/DroidKufi-Regular.ttf` (DejaVu has **no** Arabic glyphs) |

**D0.1 — MetaGen API embeddings, not local Qwen3.** Multilingual, no install, verified on
Arabic. Consequence: Arabic vectors are in a *different space* from the stored zh/en
`figurative_embeddings.npz` (Qwen3), so cross-lingual work **re-embeds** the English side
rather than reusing those files. Local `Qwen3-Embedding-0.6B` remains the documented fallback.

---

## 1. Discovery — what the Chinese pipeline did ✅

| # | Analysis | zh/en script | Arabic counterpart |
|---|---|---|---|
| A1 | statistics | `idiom_statistics.py` | `analyze_ar_idioms.py stats` |
| A2 | entity frequency + plot | `entity_clustering.py` | `… entities` |
| A3 | entity embedding k-means + 2-D map | `cluster_entities_with_embeddings.py` | `… cluster` |
| A4 | intra-lingual semantic clusters | `intra_lingual_idiom_clusters.py` | `… semantic` |
| A5 | cross-lingual same-meaning/diff-entity | `cross_lingual_same_meaning_diff_entity.py` | `cross_lingual_ar_en.py pairs` |
| A6 | cross-lingual same-entity/diff-meaning | `cross_lingual_same_entity_diff_meaning.py` | `cross_lingual_ar_en.py entities` |
| A7 | — | *(none)* | `… variety` (Arabic-only) |

**Blocker found:** A2–A6 key off `entities`, A5/A6 also use `literal_meanings`; Arabic had
**0% of both**. Hence Phase 2.

---

## 2. Phase 2 — Enrich the KB ✅

`src/culture/data_processing/ar_idioms/enrich_ar_idioms.py`

**D2.1 — LLM-generate `entities` + `literal_meanings`, tag provenance.** No Arabic source
supplies them. This does not weaken inclusion criterion 1, which governs which *source
datasets* we ingest — idioms and their figurative meanings stay verbatim human lexicography.
Every enriched row records `meta.field_provenance`.

**D2.2 — validate every generated entity against the idiom.** An LLM invents entities freely.
Each is checked to actually occur in the idiom (after Arabic normalization, so clitics and
diacritics don't cause false rejections). **360 hallucinated entities were dropped.**

Result: 10,386 calls, **0 failures**; entities **87.8%**, literal **100.0%**.

---

## 3. Phase 3 — Analyses ✅ (A5/A6 🔄)

`src/culture/analysis/analyze_ar_idioms.py` · `cross_lingual_ar_en.py`
Outputs in `data/idioms/ar/analysis/`.

| Analysis | Result |
|---|---|
| A1 stats | 10,386 idioms · 4.73 tokens mean · register: classical 4,530 / colloquial 5,755 / MSA 101 |
| A2 entities | **7,358 unique**, 17,083 mentions. Top: الله، الناس، العين، الكلب، الجمل، الحمار، الدار، القلب |
| A3 cluster | 12 k-means clusters over 1,353 entities + 2-D PCA map |
| A4 semantic | **89** shared-meaning clusters, 114 idiom pairs (cosine ≥ 0.86) |
| A7 variety | 10 varieties contrasted; distinctive entities per dialect via lift |
| A5/A6 | 🔄 running (English side capped at 6,000 rows — see D3.2) |

**D3.1 — cross-lingual against English**, the pivot the zh pipeline already uses; the Arabic KB
also carries 796 human English glosses to anchor on.
**D3.2 — cap the English side at 6,000 rows.** Embedding all 32,080 English entries (~48k
vectors) would cost ~1.5 h of API time for a marginal gain. Documented cap, not a silent one.

---

## 4. Phase 4 — Pretraining corpora ✅

**Every candidate was sampled with ≥4,000 real documents** (byte-range parquet reads), then
scanned for punctuation, Arabic ratio, PDF presentation-forms, porn/gambling/SEO, boilerplate,
and **idiom density** — with every spam regex hit hand-verified (`ساسكس`=Sussex, `كسكس`=couscous,
`بوكر`=Booker all fire naively).

### ✅ Accepted (recommended mix — Strategy A)

| # | Corpus | Size | Share | Why |
|---|---|---|---:|---|
| 1 | `epfml/FineWeb2-HQ` `arb_Arab` | 100 GB | 35% | best prose register, 2× FineWeb-2 idiom density |
| 2 | `HuggingFaceFW/finepdfs` `arb_Arab` | 56.6 GB | 30% | **3.45% idiom-bearing docs — 7–20× every web corpus**; books, literature, theses |
| 3 | `HuggingFaceFW/fineweb-2` `arb_Arab` | 106 GB | 20% | volume + breadth; **0.00% verified porn** in 4,000 docs |
| 4 | `HPLT/HPLT2.0_cleaned` `ara_Arab_1..5` | ~250 GB | 10% | lowest SEO spam (0.07%), different crawl |
| 5 | `MohamedRashad/arabic-billion-words` | 8.1 GB | 5% | spotless MSA press; columnists quote أمثال constantly. ⚠️ license undeclared |

### ❌ Discarded — with measured evidence

| Corpus | Reason |
|---|---|
| **`ClusterlabAi/101_billion_arabic_words`** | **0 of 2,390 docs across two shards contain ANY punctuation** (all others 89–100%); normalization also splits words (`قرار ا`, `ي حد د`). Plus **~2.0% hardcore porn**, **10.4% crusher/Alibaba SEO**, **34.8% nav boilerplate**, carding-fraud pages. Unusable at any filtering budget. |
| **`allenai/c4` `ar`** | 12.7% docs <300 chars, 10.7% no punctuation, 2.67% machinery SEO, pipe-delimited keyword stuffing, vBulletin index dumps, mid-word-truncated tickers; ~40% redundant with sources we already take. |
| **`AdaMLLab/AraMix-HQ`** (as primary) | The "HQ" score selects fluency, not legitimacy: **2.60% machinery SEO**, 1.02% CJK leakage, verified porn. Prefer `AdaMLLab/AraMix` `data/consensus`. |
| **`lightonai/ArabicWeb24`** | **Gated, access denied** (403 for this account). Zero documents read → no verdict; not accepted on reputation. |

**Corrections to the corpus survey:** `AdaMLLab/mixminmatch` **does not exist** (404) — the real
IDs are `AdaMLLab/AraMix` (`data/consensus` = the 54.1B "Matched" release, 80.9 GB) and
`AdaMLLab/AraMix-HQ`. FineWeb2-HQ is **not** cleaner than FineWeb-2 in the spam sense (it has
*more* verified porn); its advantage is register. HPLT's `ara_Arab_1..5` are shards, **not**
quality tiers.

**D4.1 — build our own mix (Strategy A) rather than using AraMix.** AraMix is a superset built
from seven sources *including* the discarded 101B, so it inherits that contamination and
double-counts against FineWeb-2/HPLT/FinePDFs.
**D4.2 — FinePDFs is weighted far above its size.** It is the only source where amthal actually
live (3.45% vs 0.17–0.50%). Cost: 15.2% of its docs are in Arabic Presentation Forms (fix with
NFKC) and 21.8% show lām-alef ligature corruption (detect and drop).

### Mandatory filter stack (before the idiom matcher)
1. Drop docs with **no sentence punctuation** (kills 101B-style text and title-only stubs).
2. Gambling/porn blocklist — guard the FPs above.
3. Machinery/Alibaba SEO blocklist (`كسارة|مطحنة|المصنعين والموردين|مصنع من الصين`).
4. Forex/binary-options + transliteration garble.
5. FinePDFs only: `NFKC`, then drop ≥3 hits of `إال |اآل|األ`, and Latin ratio > 0.35.
6. Drop Arabic-char ratio < 0.5, docs < 300 chars, line-uniqueness < 0.6.

---

## 5. Phase 5 — Evaluation benchmarks 🔄

Agent running (first attempt died on an API error and was relaunched). Requirements: ≥5 real
examples per benchmark, and an explicit **contamination check** — Jawaher, Kinayat, Absher,
Tunisian proverbs and tahaalselwii are all **ingested into our training KB**, so they cannot be
used as evaluation without train/test leakage.

---

## 6. Phase 6 — Filtering, tagging, CPT ✅

`src/culture/training/mC4/filter_and_tag_ar.py` — emits `tagged_*.json.gz` in exactly the
format `continued_pretraining/prepare_data.py` already consumes.

**Measured on 4,000 streamed FineWeb-2 `arb_Arab` documents (10,386-entry inventory):**

| matching | docs | match rate |
|---|---:|---:|
| raw substring (what `download_and_filter_mc4.py` does) | **0** | 0.00% |
| **Tier 0 — normalized Aho-Corasick** | **142** | **3.55%** |
| Tier 0 + Tier 2 stem | 170 | 4.25% |

**D6.1 — Tier 2 stem matching is OFF by default.** It adds +19.7% documents, but inspecting the
stem-only hits showed they are **false positives**: `كل شيء بأوان` fired on an article about
sexual dysfunction, `حبلك على غاربك` on a Tunisian history page and a school-problems thread —
the idiom appears in none of them. Light stemming collapses frequent words to short stems that
coincide by chance in long documents. Opt in with `--use_stem` if you accept the precision cost.

**D6.2 — keep substring semantics, do not add word boundaries.** A proclitic (و/ف/ب/ال) on the
first word or an enclitic pronoun on the last only adds characters outside the match; enforcing
token boundaries was measured to destroy 21.4% of genuine hits.

**Tier-0 precision spot-check (sampled contexts, 5/5 genuine):** `القشة التي قصمت ظهر البعير`,
`اتق شر من أحسنت إليه` (introduced in-text as «ذلك المثل»), `القول ما قالت حذام`,
`إنما الأعمال بالنيات`, `في أمان الله`.

**Over-matching diagnostic** (idioms firing on the largest share of the corpus) is emitted by
`measure` so common fixed phrases can be pruned: top offender `توكل على الله` at 0.75%.

End-to-end tagging verified on real data — 62/2,000 docs tagged, knowledge block renders
figurative meaning, literal meaning, entities and dialect in Arabic.

**Training assets:** `configs/qwen3p5_9b_cpt_ar.yaml`, `cpt_ar.slurm`, and `ar_amthal`
registered in `configs/dataset_info.json`. Optimizer block is byte-identical to the hi/zh
configs so the three runs stay comparable.

---

## 7. Phase 7 — Publish & commit 🔄

- Upload `data/idioms/ar/**` to `Jerry9999/CultureInFigurativeLanguage` under `data/` (additive).
- `git commit` + `git push`.

> **⚠️ Rotate the HF token.** It was pasted in plaintext into the chat, so it is in the
> transcript. It is used here only via an environment variable and never written to a file, log
> or commit. Rotate at <https://huggingface.co/settings/tokens> after the upload.

---

## 8. Decision log

| ID | Decision | Rationale |
|---|---|---|
| D0.1 | MetaGen API embeddings over local Qwen3 | multilingual, verified, no install; forces re-embedding the English side |
| D2.1 | LLM-generate entities + literal meanings, tag provenance | no Arabic source has them; criterion 1 governs source ingestion |
| D2.2 | Validate entities against the idiom text | dropped 360 hallucinations |
| D3.1 | Cross-lingual vs English | the pivot the zh pipeline uses; 796 human EN glosses available |
| D3.2 | Cap English side at 6,000 rows | ~1.5 h of API time for marginal gain |
| D4.1 | Build our own corpus mix, not AraMix | AraMix inherits the discarded 101B and double-counts |
| D4.2 | Over-weight FinePDFs vs its size | 7–20× the idiom density of any web corpus |
| D6.1 | **Tier-2 stem matching off by default** | measured false positives on real text |
| D6.2 | Keep substring semantics (no word boundaries) | boundaries would destroy 21.4% of genuine hits |

---

## 9. RUNBOOK — corpus download & training on another server

Everything below is copy-paste. `$REPO` = this repository, `$DATA_ROOT` = a large scratch disk.

### 9.1 Environment

```bash
git clone <this-repo> && cd CultureInFigurativeLanguage
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets huggingface_hub pyahocorasick \
            scikit-learn matplotlib arabic-reshaper python-bidi numpy
pip install -e .                      # the `culture` package
export HF_TOKEN=<your token>          # FineWeb-2 / FinePDFs are ungated, but avoids rate limits
export DATA_ROOT=/lustre/.../culture-pretraining-data
```

### 9.2 Build the Arabic idiom KB (fast, ~2 min — or just use the published one)

```bash
# Option A: pull the finished KB from HuggingFace (recommended)
huggingface-cli download Jerry9999/CultureInFigurativeLanguage --repo-type dataset \
  --include "data/idioms/ar/*" --local-dir .

# Option B: rebuild from the upstream sources
python src/culture/data_processing/ar_idioms/build_ar_idioms.py \
    --out data/idioms/ar/idioms_merged_llm_formatted.jsonl \
    --report data/idioms/ar/build_report.json
# then enrich (needs METAGEN_API_KEY):
python src/culture/data_processing/ar_idioms/enrich_ar_idioms.py \
    --input data/idioms/ar/idioms_merged_llm_formatted.jsonl \
    --cache data/idioms/ar/enrich_cache.jsonl --workers 24
```

### 9.3 Filter + tag the pretraining corpus

The corpus is **streamed**, so there is no multi-TB download step. Run one job per source and
concatenate; each writes `tagged_*.json.gz`.

```bash
AR=src/culture/training/mC4/filter_and_tag_ar.py

# 1) FinePDFs — highest idiom density, run this one first
python $AR filter --dataset HuggingFaceFW/finepdfs --config arb_Arab \
    --out $DATA_ROOT/ar-amthal-cpt/data/finepdfs --max_docs_per_idiom 10000

# 2) FineWeb2-HQ
python $AR filter --dataset epfml/FineWeb2-HQ --config arb_Arab \
    --out $DATA_ROOT/ar-amthal-cpt/data/fineweb2hq --max_docs_per_idiom 10000

# 3) FineWeb-2
python $AR filter --dataset HuggingFaceFW/fineweb-2 --config arb_Arab \
    --out $DATA_ROOT/ar-amthal-cpt/data/fineweb2 --max_docs_per_idiom 10000

# 4) HPLT 2.0 (five shard-configs)
for c in ara_Arab_1 ara_Arab_2 ara_Arab_3 ara_Arab_4 ara_Arab_5; do
  python $AR filter --dataset HPLT/HPLT2.0_cleaned --config $c \
      --out $DATA_ROOT/ar-amthal-cpt/data/hplt_$c --max_docs_per_idiom 10000
done

# 5) arabic-billion-words (purity anchor)
python $AR filter --dataset MohamedRashad/arabic-billion-words --config default \
    --out $DATA_ROOT/ar-amthal-cpt/data/abw --max_docs_per_idiom 10000

# sanity: recall/precision + over-matching diagnostic on a sample
python $AR measure --limit 20000 --out /tmp/ar_measure
```

Each run writes `filter_report.json` (scanned / matched / written / inventory coverage /
top idioms). **Check the `over_matching_idioms` list** and prune any expression firing on
>0.5% of the corpus before the full build.

### 9.4 Reshard for LLaMA-Factory

```bash
mkdir -p $DATA_ROOT/ar-amthal-cpt/data_all
cp $DATA_ROOT/ar-amthal-cpt/data/*/tagged_*.json.gz $DATA_ROOT/ar-amthal-cpt/data_all/
python src/culture/training/continued_pretraining/prepare_data.py \
    --src_dir $DATA_ROOT/ar-amthal-cpt/data_all \
    --out_dir $DATA_ROOT/train_ar
python src/culture/training/continued_pretraining/prepare_data.py \
    --verify_only --out_dir $DATA_ROOT/train_ar        # prints the doc count
```

### 9.5 Point the config at your paths, then train

Edit `src/culture/training/continued_pretraining/configs/`:
* `dataset_info.json` → `ar_amthal.file_name` = `$DATA_ROOT/train_ar`
* `qwen3p5_9b_cpt_ar.yaml` → `model_name_or_path`, `deepspeed`, `dataset_dir`, `output_dir`

```bash
sbatch src/culture/training/continued_pretraining/cpt_ar.slurm     # 4 nodes x 8 GPUs
# or single node:
llamafactory-cli train src/culture/training/continued_pretraining/configs/qwen3p5_9b_cpt_ar.yaml
```

**Gotchas (learned on the hi/zh runs):** keep `flash_attn: sdpa` — the FA2 path crashes on
Qwen3.5's optional `s_aux`; `.json.gz` is not directly loadable by LLaMA-Factory, hence
`prepare_data.py`; `overwrite_output_dir: false` gives auto-resume on requeue.

### 9.6 Evaluate

```bash
BASE_MODEL=/path/Qwen3.5-9B CPT_MODEL=/path/qwen3p5-9b-ar-cpt \
  bash src/culture/evaluation/run_perplexity.sh    # Dim 2, Arabic LM
  bash src/culture/evaluation/run_lm_eval.sh       # Dim 1, English retention
# Dim 3/4 Arabic benchmarks: see Phase 5 once the benchmark set is finalised.
```

---

## 10. Progress log

| Date | Event |
|---|---|
| 2026-08-22 | Discovery done; MetaGen chat + APE embeddings verified; plan created |
| 2026-08-22 | Phase 2 enrichment: 10,386 calls, 0 failures, 360 hallucinated entities dropped |
| 2026-08-22 | Phase 3 A1–A4 + A7 complete; A5/A6 running |
| 2026-08-22 | Phase 4 complete: 5 corpora accepted, 4 discarded with measured evidence |
| 2026-08-22 | Phase 6 complete: matcher measured 0 → 142 docs/4k; Tier 2 disabled on precision evidence; CPT config + slurm added |
