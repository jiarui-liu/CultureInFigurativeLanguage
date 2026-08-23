# Arabic Pipeline — Analyses, Corpora, Evaluation, CPT

**Owner:** automated run. **Started / last updated:** 2026-08-22.
**Status:** ⬜ not started · 🔄 in progress · ✅ done · ⚠️ done with caveats · ❌ dropped

Tracks every task, **every decision and why**, and ends with the exact commands to
download the corpus and run training on another server (§9).

**One-line summary:** CPT Qwen3.5-9B on **FineWeb-2 `arb_Arab`** — quality-gated, filtered down
to the ~3.6 % of documents containing one of our 10,386 Arabic idioms, each tagged with an Arabic
knowledge block giving that idiom's figurative meaning, literal meaning, entities and dialect.

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

## 3. Phase 3 — Analyses ✅

`src/culture/analysis/analyze_ar_idioms.py` · `cross_lingual_ar_en.py`
Outputs in `data/idioms/ar/analysis/`.

| Analysis | Result |
|---|---|
| A1 stats | 10,386 idioms · 4.73 tokens mean · register: classical 4,530 / colloquial 5,755 / MSA 101 |
| A2 entities | **7,358 unique**, 17,083 mentions. Top: الله، الناس، العين، الكلب، الجمل، الحمار، الدار، القلب |
| A3 cluster | 12 k-means clusters over 1,353 entities + 2-D PCA map |
| A4 semantic | **89** shared-meaning clusters, 114 idiom pairs (cosine ≥ 0.86) |
| A7 variety | 10 varieties contrasted; distinctive entities per dialect via lift |
| A6 shared entities | ✅ **97 of the top 150 Arabic entities also head English idioms.** عين=eye (ar 108 / en 39), يد=hand (35/35), يوم=day (37/31), شيء=thing (32/86). → `shared_entities_ar_en.jsonl`, `shared_entity_stats_ar_en.json` |
| A5 pairs | ✅ **363 same-meaning/different-entity ar↔en pairs** (258 unique Arabic idioms, 251 English). 285 have entities on both sides and **all 285 have entirely non-overlapping entities** — which is the point of A5. → `cross_lingual_pairs_ar_en.jsonl` |

**A5 finding — the cross-lingual meaning pivot does not transfer from Chinese to Arabic.**
Running it the Chinese way (embed both sides, cosine ≥ 0.70) gave **21 pairs against zh-en's
37,045**. That gap is not a threshold choice or a data defect; measuring top-1 similarity against
the 4,538 English glosses shows why:

| Arabic side embedded as | n | mean | p99 | max | ≥ 0.70 |
|---|---:|---:|---:|---:|---:|
| Arabic text | 9,596 | 0.354 | 0.465 | **0.607** | **0** |
| human English gloss | 1,590 | 0.502 | 0.695 | 0.825 | 15 |

**Not one Arabic-language meaning reaches 0.70**, while Chinese clears it directly (max 0.879)
with the same `text-embedding-3-small` @512d. The content is not at fault — the Arabic entries
that happen to carry a human English gloss match normally — so this is Arabic↔English *sentence*
alignment in the embedding model. Every one of those 21 pairs came from the 796 English-glossed
entries; the other 9,590 contributed nothing.

Fix (**D3.3**): translate the Arabic meanings to short English paraphrases first, then match
English↔English — the same move that makes A6 work. **21 → 363 pairs, a 17× recovery.** 9,680 of
11,152 meanings were translated (mean 40 chars, matching the English glosses' 64; 0.1 % empty).
Six randomly sampled pairs were all valid equivalences with different entities — e.g.
`ضَرِطُ البَلْقاءِ` ↔ *chin music* (boastful empty talk), `تمحي في الزبدة وتحفي في اللبن` ↔ *bang
one's head against a brick wall*, `أشأَمُ مِنَ اْلأَخْيَلِ` ↔ *bad news bears*.

Still an order of magnitude below zh-en's 37,045, and the residual gap is honest signal rather
than a bug: the Arabic KB is a third the size on the meaning side (11,152 vs 47,085 meanings),
and its entries are dictionary exegesis of largely classical proverbs, which have fewer direct
English counterparts than modern Chinese chengyu glosses do.

A second, linguistic fix went in alongside (**D3.4**): Arabic proverb dictionaries write
etymology and anecdote first and the actual gloss last, after `يُضرب في/لمن…` ("it is said
of…"). `usage_gloss()` trims to that clause, cutting a mean 230-char entry to a 54-char median
gloss, and it fires on 38.2 % of meanings (49.3 % of entries over 150 chars). It is on by default
because it removes text that is genuinely not the meaning — but reported honestly: **on its own
it did not move the pair count at all**, because the ceiling was the embedding space, not length.

**D3.1 — cross-lingual against English**, the pivot the zh pipeline already uses; the Arabic KB
also carries 796 human English glosses to anchor on.
**D3.2 — cap the English side at 6,000 rows.** Embedding all 32,080 English entries (~48k
vectors) would cost ~1.5 h of API time for a marginal gain. Documented cap, not a silent one.

**Two operational failures worth recording** (both fixed in `embed_cached`):

1. The first A5 run died at 1,152/4,084 English vectors on a single transient `HTTP 504`.
   `autoresearch.utils.llm.embed_texts` re-raises 5xx, so one blip killed a 40-minute job.
   `_embed_with_retry` now backs off exponentially (5→160 s, 6 attempts). Everything already
   embedded is on disk, so a rerun resumes from the cache and costs nothing.
2. The relaunch then 401'd because embeddings hit the **APE** endpoint but
   `EmbeddingSettings.from_env()` reads `EMBEDDING_API_KEY` first and `METAGEN_API_KEY` only as a
   fallback — exporting `APE_API_KEY` alone silently authenticates with the *wrong* key. Run
   `export EMBEDDING_API_KEY=$APE_API_KEY`. Auth errors now fail fast with that message instead
   of burning five minutes of backoff. Related: `~/.bashrc` returns early in non-interactive
   shells, so `source ~/.bashrc` inside a `tmux new-session` command sets nothing.

---

## 4. Phase 4 — Pretraining corpora ✅

**Every candidate was sampled with ≥4,000 real documents** (byte-range parquet reads), then
scanned for punctuation, Arabic ratio, PDF presentation-forms, porn/gambling/SEO, boilerplate,
and **idiom density** — with every spam regex hit hand-verified (`ساسكس`=Sussex, `كسكس`=couscous,
`بوكر`=Booker all fire naively).

### ✅ SELECTED: `HuggingFaceFW/fineweb-2` `arb_Arab` — single corpus

**D4.3 (2026-08-23) — one corpus, not a five-way blend.** The hand-tuned mix below was
over-engineered and, once the candidates were measured end-to-end on the same footing, two of its
premises turned out to be false. Head-to-head, quality gates on, same inventory:

| Corpus | scanned | match rate | uniq idioms **per 1M kept chars** | median tokens/doc | % docs > `cutoff_len` |
|---|---:|---:|---:|---:|---:|
| **`fineweb-2`** | 4,000 | **3.58 %** | **14.7** | **4,198** | 28 % |
| `finepdfs` | 3,000 | 5.53 % | 4.3 | 95,664 | **93 %** |
| `FineWeb2-HQ` | 3,000 | **1.00 %** | 8.2 | 3,510 | 23 % |

Two corrections this forced:

- **FineWeb2-HQ was to be the largest share (35 %) on the survey's claim of "2× FineWeb-2 idiom
  density". It is 3.6× WORSE (1.00 % vs 3.58 %).** The HQ classifier selects formal/encyclopedic
  register — precisely where colloquial proverbs do not appear. Optimising for generic "quality"
  optimises against our signal.
- **FinePDFs' density advantage is mostly unusable.** Its median document is 95,664 Qwen tokens
  (5.8 training sequences) and 93 % exceed `cutoff_len`. The knowledge block is appended at the
  END, so for 93 % of its documents the idiom and its gloss land in *different training
  sequences* and the tagging does nothing. Normalised per character kept, it yields 4.3 unique
  idioms/1M vs FineWeb-2's 14.7 — a 3.4× deficit.

FineWeb-2 wins on the metric that matters (idiom yield per character actually trained on), fits
the context window, is the largest single source at 106 GB, has the lowest gate-rejection rate
(1.1 %), and showed **0.00 % verified porn** in 4,000 sampled docs. One source also makes the
recipe reproducible without defending five arbitrary percentages.

<details><summary>Superseded five-way mix (kept for the record)</summary>

| # | Corpus | Size | Share | Original rationale |
|---|---|---|---:|---|
| 1 | `epfml/FineWeb2-HQ` | 100 GB | 35% | best prose register, "2× density" — **disproved** |
| 2 | `HuggingFaceFW/finepdfs` | 56.6 GB | 30% | 3.45% idiom-bearing — **true but unusable at 16K ctx** |
| 3 | `HuggingFaceFW/fineweb-2` | 106 GB | 20% | volume + breadth — **now the sole source** |
| 4 | `HPLT/HPLT2.0_cleaned` `ara_Arab_1..5` | ~250 GB | 10% | lowest SEO spam (0.07%), different crawl |
| 5 | `MohamedRashad/arabic-billion-words` | 8.1 GB | 5% | spotless MSA press. ⚠️ license undeclared |

Reach for these only if FineWeb-2 alone does not yield enough tokens. HPLT is the natural second
(different crawl, low spam); FinePDFs becomes viable now that `--max_doc_chars` windows long
documents, and is the best source of *literary* register if register diversity is wanted.
</details>

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

### Mandatory filter stack (before the idiom matcher) — ✅ IMPLEMENTED

`src/culture/training/mC4/quality_ar.py`, called from `filter_and_tag_ar.py` before the matcher.
Every drop is attributed by gate in `filter_report.json → rejected_by_gate`. Disable for
ablations with `--no_quality_filter`.

1. Drop docs with **no sentence punctuation** (kills 101B-style text and title-only stubs).
2. Gambling/porn blocklist — guard the FPs above.
3. Machinery/Alibaba SEO blocklist (`كسارة|مطحنة|المصنعين والموردين|مصنع من الصين`).
4. Forex/binary-options + transliteration garble.
5. FinePDFs only: `NFKC`, then drop ≥3 hits of `إال |اآل|األ`, and Latin ratio > 0.35.
6. Drop Arabic-char ratio < 0.5, docs < 300 chars, line-uniqueness < 0.6.

**Measured on live traffic:**

| Corpus | scanned | rejected | top gates | match rate after |
|---|---:|---:|---|---:|
| FineWeb-2 `arb_Arab` | 4,000 | **46 (1.2 %)** | not_arabic_enough 24 · too_short 21 · spam_adult 1 | 3.58 % |
| FinePDFs `arb_Arab` | 3,000 | **1,245 (41.5 %)** | **pdf_ligature_corrupt 836 (27.9 %)** · not_arabic_enough 226 · too_short 111 · latin_heavy 50 · repetitive_lines 22 | 5.53 % |

The PDF gate is the one that earns its keep: **27.9 % ligature corruption measured, above the
21.8 % the survey estimated.** FinePDFs is 30 % of the mix, so without this gate roughly **8 % of
the entire training corpus would be lām-alef-corrupted text**. And FinePDFs still out-densities
every web source *after* discarding 41.5 % of it (5.53 % vs 3.58 % idiom-bearing), which
re-confirms D4.2.

False positives were the design constraint, not an afterthought: the blocklists match multi-word
collocations, never bare substrings, because `كسارة` (crusher) collides with `كسكس` (couscous) and
`ساسكس` (Sussex), and prose *about* the ethics of `قمار` (gambling) is legitimate. Cross-checked
against the KB: **0 of 1,515 long idiom meanings are falsely blocked.**

> **Gotcha:** the streaming reader segfaults at interpreter teardown
> (`PyGILState_Release`) on some `datasets`/`pyarrow` builds. It fires *after*
> `filter_report.json` and the shards are written and flushed — outputs verified intact. Ignore
> it, or check the report file rather than the exit code.

---

## 5. Phase 5 — Evaluation benchmarks ✅

~45 Hub searches; **26 candidates screened, ≥5 verbatim examples read from every one**, each
checked for genuine item leakage (see the three-way distinction below) against the 10,375 unique normalised surfaces in
`data/idioms/ar/idioms_merged_llm_formatted.jsonl`. Implemented in
`src/culture/evaluation/tasks_ar.py` (registered in `run_eval.py`), downloaded by
`download_ar.sh`, probes by `build_ar_probes.py`, launched by `run_eval_ar.sh` / `eval_ar.slurm`.

### ✅ Accepted

| Task | Source | n | Scoring | Role |
|---|---|---:|---|---|
| `kinayat_cloze` | `menaattia/Kinayat`, `sentence`+`correct`/`incorrect` | **150** | `acc_norm` | **Dim 3 primary — the Arabic ChID.** Fill a `ـــــ` blank with the right idiom |
| `kinayat_meaning` | `menaattia/Kinayat`, `Ar_`/`Incorrect_Explanation` | **325** | `acc_norm` | Dim 3 — meaning selection. Measures *knowledge injection*; see the caveat below |
| `ar_figurative` | Alyah figurative (214) + DziriEval figurative (100) | **314** | `acc_norm` | Dim 3 — figurative comprehension on idioms outside the KB |
| `arabculture` | `MBZUAI/ArabCulture`, 13 countries | 3,471 | `acc_norm` | Dim 4 primary; native cultural completion |
| `arabic_cultural_qa` | `QCRI/ArabicCulturalQA` `mcq/test.jsonl` | 2,000 (×6 variants) | `acc` letter | Dim 4 primary; cross-dialect axis via `--acqa_dialects all` |
| `arabicmmlu` | `MBZUAI/ArabicMMLU`, 13 Arab-region subjects | 10,529 | `acc` letter | Dim 4 primary; the CMMLU-China-specific analogue |
| `global_piqa_ar` | Global-PIQA non-parallel, 13 Arabic varieties | 1,099 | `acc_norm` | Dialect-level cultural commonsense |
| `alyah` | `tiiuae/alyah-emirati-benchmark` (full) | 1,173 | `acc_norm` | Gulf/Emirati depth |
| `dzirieval` | `touati-kamel/DziriEval` | 950 | `acc_norm` | Maghreb; **secondary** (no license, no paper) |
| `global_piqa_ar_parallel` | Global-PIQA parallel arb/ary/arz | 309 | `acc_norm` | **Control** — culture-agnostic physics; regression detector |
| `ar_fineweb2_heldout` | FineWeb-2 `arb_Arab` **official test split** | 2,000 docs | BPB | Dim 2 in-domain |
| `ar_wiki_heldout` | Arabic Wikipedia, title-decontaminated | 2,000 docs | BPB | Dim 2 out-of-domain |

Counts are what the loaders actually return, verified against the downloaded files. Two differ
from the raw row counts on purpose: ArabCulture 3,482 → **3,471** (11 rows carry the annotators'
`should_discard=Yes`), DziriEval 1,000 → **950** (see D5.3).

### Contamination — three distinct things, only one disqualifying

**Correction (2026-08-23).** An earlier version of this plan discarded every Arabic idiom
benchmark on the grounds that its idioms appear in our training KB. That criterion was wrong and
it cost us the best benchmark in the survey. Sharing *knowledge* with the training corpus is what
CPT is for — by that standard MMLU would be void because Wikipedia is in every pretraining mix.
The three cases must be separated:

| # | Case | Verdict | Here |
|---|---|---|---|
| 1 | **Knowledge overlap** — the idiom and its meaning are in training | Not contamination; it is the intervention | `kinayat_cloze`, and every Dim-4 task |
| 2 | **Answer-string familiarity** — the gold answer *string* is injected verbatim, the distractor is not | Usable, must be labelled | `kinayat_meaning`, Jawaher |
| 3 | **Item leakage** — the eval instance (stem + options + answer) is in training | Fatal | **nothing** |

**Measured, not argued.** `filter_and_tag_ar.py::knowledge_block` emits only
`figurative_meanings`, `literal_meanings`, `entities` and `region` — it never emits the
`examples` field, so Kinayat's usage sentences never leave the KB jsonl. Checked against a real
tagged shard: **0 of 298 Kinayat stems appear in the tagged corpus** (3 of 317 gold *meanings*
do, which is case 1 working as designed).

`kinayat_cloze` is therefore clean, and **it is the Arabic ChID we earlier said did not exist**.
Solving it needs to know which of two idioms fits a context — unanswerable by recalling a seen
string. Exposure across the options is near-symmetric (94 golds vs 84 distractors are KB idioms);
`--kinayat_symmetric` restricts to the 116 items where both options have identical KB status.
No length bias (gold longer in 69/150). Gold position is randomised at load time because the
source file always lists the correct idiom first.

`kinayat_meaning` is case 2: the gold explanation is exactly the string we inject, the distractor
is not, so a model can win by recognition. Report it as **knowledge injection**, and read
`kinayat_cloze` for **comprehension**. Note the distractors are LLM-written and fluent while the
golds are terse dictionary prose — Qwen3-0.6B scores 0.09, well *below* chance, so a base model
has no free ride here.

### ❌ Discarded — on their own merits, not for overlap

| Benchmark | Reason |
|---|---|
| `Renad10/Absher-Benchmark` | The gold answer is concatenated into the prompt string in ~10 % of rows; 58 missing golds; 91 % position bias (majority-class ≈ 54 %); answer column mixes `A/B/C/D` with `أ/ب/ج/د`. Broken regardless of overlap |
| `ahmed02mk/amthal-hassaniya` | Instruction/output generation, no options → not log-likelihood scorable |
| `UBC-NLP/Jawaher-benchmark` | Free-text explanation, no options → needs the LLM judge; and it is case 2 at its most extreme (the gold explanation is verbatim training data), so the judge would be scoring memorised text. Left out, not declared invalid — worth revisiting with a generation-mode eval |
| `tahaalselwii/*`, `HabibaAbderrahim` Tunisian | Lexicons, not benchmarks; already in `EXCLUDED_SOURCES` |

**Procedural defect still worth fixing:** `build_ar_idioms.py` merges Jawaher *train* and *test*
under one source label `"jawaher"` with **no split provenance in `meta`**. That is not a
contamination problem, but it does mean the KB cannot tell you which upstream split an entry came
from. Record `split` before the next rebuild.

### ❌ Discarded — quality / wrong construct

`OALL/ACVA` (GPT-3.5 templated; 125/195 exact duplicates in `Arabic_Food`, 390/575 share a 4-word
prefix) · `CohereForAI/Global-MMLU ar` (MMLU translated into Arabic; its "culturally sensitive"
rows are *Western*-culture — Polaris from the USA, US/European history) · `nayeon212/BLEnD`
(0 Arabic-script prompts; 304 base questions inflated to 20,364 rows) · `QCRI/AraDiCE-Culture`
(no answer field at all) · `QCRI/AraDiCE` root (an 8-row index table) · `HYU-NLP/MIDAS AR_Idioms`
(8,051 idioms but `"Sentence": []` on **every** row and 24.2 % OCR-corrupt: `هللا`/`اال`
ligature damage) · `nassimjp/multilingual-proverb-reasoning` (Japanese proverbs glossed in
Pashto) · `asas-ai/Arabic_WSD_Benchmark` · `Raniahossam33/Arabic_Culture_Dataset` (LLM DPO pairs)
· `kellycyy/CulturalBench` (English-only) · `neulab/CulturalGround` (synthetic training data) ·
`ashabrawy/dia_figqa` + `cmu-lti/multi-figqa` (**no Arabic subset** — MABL, which we use for
Hindi, has no Arabic counterpart) · `ArSyra/*`, `IdiomX`, `IdiomTranslate30` (already excluded as
synthetic/translation corpora).

### The Dim-3 suite, and how to read it

| Task | n | What a gain means |
|---|---:|---|
| `kinayat_cloze` | 150 | **Comprehension** on KB idioms: picks the right idiom for an unseen context |
| `kinayat_meaning` | 325 | **Knowledge injection** on KB idioms (answer-string familiarity applies) |
| `ar_figurative` | 314 | **Generalisation**: Alyah/DziriEval idioms are ~99 % outside the KB |

Together 789 items across three constructs, and the *pattern* across them is the finding: a rise
in `kinayat_meaning` alone is memorisation; a rise in `kinayat_cloze` is comprehension of injected
idioms; a rise in `ar_figurative` is transfer to idioms we never trained on.

Still worth building later: **a larger ChID-Arabic.** MIDAS AR contributes **7,973 idioms absent
from our KB** (overlap 21/7,994 = 0.3 %), and FineWeb-2 `arb_Arab/test` is a 121 M-char held-out
corpus to mine natural usages from — blank the idiom, draw distractors from same-length MIDAS
entries. That would give a held-out-idiom cloze to sit alongside `kinayat_cloze`'s in-KB cloze,
which is the cleanest possible pairing. Requires cleaning MIDAS's 24 % OCR damage first.

### Wikipedia is inside FineWeb-2 — measured, and handled

95 / 42,080 docs (0.23 %) in the FineWeb-2 test shard come from `*.wikipedia.org`, 71 from
`ar.wikipedia.org`; 0.32 % of bytes. Extrapolated over ~55 M `arb_Arab` docs that is ~43 k–93 k
articles, **4–8 % of the Arabic dump**. So `build_ar_probes.py --exclude_urls '<train shards>'`
collects every `ar.wikipedia.org/wiki/<title>` in the shards that will actually be trained on and
excludes those titles from the OOD probe. For the *filtered* arm residual overlap is ≈0.01 %; for
an *unfiltered* arm the flag is mandatory.

### Verified end-to-end

All 8 MC tasks scored with a real model (`Qwen/Qwen3-0.6B`, `--limit 60`). The length-bias
correction is doing real work: on `ar_figurative` raw `acc` = 0.117 vs `acc_norm` = 0.317 —
un-normalised log-prob collapses onto the shortest option. **Read `primary`** (= `acc_norm` for
continuation tasks, `acc` for letter tasks), never raw `acc` on Alyah / `ar_figurative`.

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

## 7. Phase 7 — Publish & commit ✅ (push blocked)

**HuggingFace — done.** 22 files uploaded to
[`Jerry9999/CultureInFigurativeLanguage`](https://huggingface.co/datasets/Jerry9999/CultureInFigurativeLanguage/tree/main/data/idioms/ar)
under `data/idioms/ar/`. Purely additive: the repo had 13,485 files and **no `ar` files**
beforehand. Includes the KB (11 MB), `build_report.json`, `audit_report.json`,
`enrich_cache.jsonl`, the six audit `review/` shards, nine `analysis/` outputs, two plots, and a
`README.md` dataset card documenting the schema, `field_provenance`, per-source counts and the
evaluation-contamination warning. **Excluded:** `analysis/cache/` (191 MB of embedding vectors,
regenerable) — `data/` is also in `.gitignore`, so the Hub is the distribution channel.

**git commit — done.** `e6581f4` (pipeline) and `c4fc35e` (evaluation suite).

**⚠️ `git push` must be run by you.** It fails from this agent:
`fwdproxy ... 403 ... github.com has not been allowlisted in filter {"agent_id":"agent:claude_code"}`.
Run `git push origin main` yourself from an interactive shell.

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
| D5.1 | ~~Discard every Arabic *idiom* benchmark~~ **REVERSED 2026-08-23** | The original criterion (idioms overlap the training KB) conflated knowledge overlap with item leakage. Only item leakage disqualifies, and there is none: the tagger never emits `examples`, so 0/298 Kinayat stems reach the corpus. `kinayat_cloze` (150) restored as the Dim-3 primary — it is the Arabic ChID |
| D5.1b | Keep `kinayat_meaning` but label it | Its gold string IS injected verbatim while the distractor is not — an answer-string familiarity shortcut. Valid for measuring knowledge injection, not comprehension |
| D5.1c | Randomise gold position in both Kinayat tasks | The source CSV always lists the correct option first |
| D5.2 | In-domain BPB probe = FineWeb-2's **official test split**, not a reserved train shard | genuinely held out, stable across runs, and strictly better than the zh in-domain probe, which is contaminated by design |
| D5.3 | Dedup DziriEval on **question text**, not `id` | `id` is not a unique key: only 850 distinct ids for 950 distinct questions, because 50 ids are reused for different questions (49 with different golds). An id-keyed dedup silently deletes 100 real items |
| D5.4 | `acc_norm` primary for all continuation tasks | Alyah's gold is the longest option 57.5% of the time (chance 25%); measured raw `acc` 0.117 vs `acc_norm` 0.317 on `ar_figurative` |
| D5.5 | `--ar_kb_path` decontamination is opt-in, not forced | measured impact is 2/1173 (Alyah) and 0/314 (`ar_figurative`); those are incidental pan-Arab proverb overlaps, not item leakage, so dropping them is a choice not a fix |
| D4.3 | **One corpus (`fineweb-2 arb_Arab`), not a 5-way blend** | measured head-to-head: FineWeb2-HQ is 3.6× *worse* on idiom density than the survey claimed it was better, and 93% of FinePDFs docs exceed `cutoff_len` so their knowledge block never co-occurs with the idiom. FineWeb-2 leads on idioms per character actually trained on (14.7/1M vs 4.3) |
| D4.4 | Window over-long docs around the match (`--max_doc_chars 25000`) | 36% of tagged FineWeb-2 docs exceeded `cutoff_len`; budget uses the worst measured chars/token (1.56), not the median (2.52) |
| D4.5 | Trim knowledge-block meanings to the يُضرب clause, cap 300 chars | one classical entry ran to 21,140 chars of etymology — wrong content to inject, and it alone blew the window |
| D5.6 | Download benchmarks with `curl`, not `hf_hub_download` | every Xet-backed file stalls at 0 bytes behind this proxy; `resolve/main` over plain HTTPS works |
| D3.3 | Translate Arabic meanings to English before matching (A5) | measured: no Arabic-language meaning exceeds 0.607 cosine to any English gloss, vs a 0.70 threshold Chinese clears directly at 0.879 |
| D3.4 | Trim Arabic meanings to the `يُضرب` usage clause | Arabic dictionaries put etymology first; cuts 230→54 chars on 38.2% of meanings. Kept for correctness, but it did **not** change the pair count |
| D3.5 | A5 threshold 0.70, matching the Chinese run | comparability; 0.80 was my earlier mismatch and yielded 3 pairs |

---

## 9. RUNBOOK — corpus download & training on another server

Everything below is copy-paste. `$REPO` = this repository, `$DATA_ROOT` = a large scratch disk.

### What this pipeline actually does

> **Pretraining corpus: `HuggingFaceFW/fineweb-2`, config `arb_Arab`, split `train` — one
> source, nothing else** (decision **D4.3**, measured head-to-head against FinePDFs and
> FineWeb2-HQ in §4).
>
> We do **not** train on it raw. The corpus is streamed and each document goes through:
>
> 1. **Quality gates** (`quality_ar.py`) — punctuation, Arabic ratio, line uniqueness, length,
>    and adult/SEO/forex blocklists. Drops ~1.1 % of FineWeb-2; every drop is attributed by gate
>    in `filter_report.json`.
> 2. **Idiom filtering** (`filter_and_tag_ar.py`) — morphology-aware matching of the
>    10,386-entry Arabic KB against the normalized document. **Keeps only the ~3.6 % of documents
>    that actually contain an idiom**; raw substring matching would keep 0 %.
> 3. **Windowing** — documents too long for `cutoff_len` are trimmed to a window around the
>    matched idiom, so the idiom and its gloss stay in the same training sequence.
> 4. **Metadata tagging** — an Arabic knowledge block is appended to each kept document:
>    figurative meaning, literal meaning, entities and dialect for every idiom it contains.
>
> Output is `tagged_*.json.gz` → resharded by `prepare_data.py` → LLaMA-Factory `pt` stage.
> So the training text is **`<original FineWeb-2 document>\n\n<Arabic knowledge block>`**.

A tagged document ends like this:

```
المعاني الاصطلاحية للتعابير الواردة في النص:
- يِقْتِلِ الْقَتِيلْ وِيِمْشِي فِي جَنَازْتُهْ
  المعنى المجازي: يُضرَب لمن بلغ في الدهاء مبلغًا عظيمًا.
  المعنى الحرفي: يقتل القتيل ثم يمشي في جنازته.
  العناصر: الْقَتِيلْ، جَنَازْتُهْ
  اللهجة: Egyptian، Libyan
```

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
# Option A: pull the finished KB from HuggingFace (RECOMMENDED — it is already
# enriched, repaired and audited, and skips ~10,386 LLM calls).
export HF_HUB_DISABLE_XET=1        # Xet-backed files stall at 0 B behind some proxies
hf download Jerry9999/CultureInFigurativeLanguage --repo-type dataset \
  --include "data/idioms/ar/*" --local-dir .

# Option B: rebuild from the upstream sources — ALL FOUR STEPS, in order.
python src/culture/data_processing/ar_idioms/build_ar_idioms.py \
    --out data/idioms/ar/idioms_merged_llm_formatted.jsonl \
    --report data/idioms/ar/build_report.json
# enrich: adds `entities` + `literal_meanings` (needs METAGEN_API_KEY). ~40 min.
python -m culture.data_processing.ar_idioms.enrich_ar_idioms \
    --input data/idioms/ar/idioms_merged_llm_formatted.jsonl \
    --cache data/idioms/ar/enrich_cache.jsonl --workers 24
# repair: strips editing notation and cloze blanks the builder cannot see.
# Idempotent, and asserts no entry loses its idiom or all of its meanings.
python -m culture.data_processing.ar_idioms.repair_ar_kb \
    --input data/idioms/ar/idioms_merged_llm_formatted.jsonl
# audit: 16 deterministic checks. Expect ~99.1% clean; investigate if lower.
python -m culture.data_processing.ar_idioms.audit_idioms \
    --input data/idioms/ar/idioms_merged_llm_formatted.jsonl --lang ar \
    --report data/idioms/ar/audit_report.json
```

Expected end state: **10,386 entries, 99.09 % audit-clean**, `figurative_meanings` 100 %
(human), `literal_meanings` 100 % and `entities` 87.8 % (LLM, flagged in
`meta.field_provenance`), `variety_region`/`register` 100 %, ISO `variety` code 97.8 %.

### 9.3 Filter + tag the pretraining corpus

One source (`fineweb-2 arb_Arab`, see D4.3), streamed — no multi-TB download step. Writes
`tagged_*.json.gz`.

```bash
AR=src/culture/training/mC4/filter_and_tag_ar.py
# --idioms defaults to data/idioms/ar/idioms_merged_llm_formatted.jsonl, so run
# 9.2 first: a stale/unrepaired KB silently loses recall.
# The corpus is STREAMED - there is no multi-TB download step.

python $AR filter \
    --dataset HuggingFaceFW/fineweb-2 --config arb_Arab --split train \
    --out $DATA_ROOT/ar-amthal-cpt/data/fineweb2 \
    --max_docs_per_idiom 10000

# sanity: recall/precision + over-matching diagnostic on a sample
python $AR measure --limit 20000 --out /tmp/ar_measure
```

Defaults worth knowing (all measured, see §4): the §4 quality gates run before the matcher
(`--no_quality_filter` to disable), `--max_doc_chars 25000` windows over-long documents so the
knowledge block stays inside `cutoff_len`, and `--min_doc_chars 300`.

**If you need more tokens than FineWeb-2 alone provides**, add sources one at a time and
concatenate the output dirs — `HPLT/HPLT2.0_cleaned` (`ara_Arab_1`…`_5`) first, then
`HuggingFaceFW/finepdfs` (`--is_pdf` auto-enables; expect ~41.5% of its docs to be dropped, 27.9%
for lam-alef corruption alone). Do **not** add `epfml/FineWeb2-HQ`: measured 1.00% idiom density,
3.6× worse than FineWeb-2.


Each run writes `filter_report.json` (scanned / matched / written / inventory coverage /
top idioms / **`rejected_by_gate`**). Two things to check before committing to a full build:
**`over_matching_idioms`** — prune any expression firing on >0.5% of the corpus — and
**`rejected_by_gate`**, which should look like the table in §4; a wildly different profile means
the source is not what the survey measured.

The streaming reader may segfault at interpreter exit (`PyGILState_Release`) on some
`datasets`/`pyarrow` builds. It happens after the shards and report are flushed, so judge success
by `filter_report.json`, not the exit code.

### 9.4 Reshard for LLaMA-Factory

```bash
# Single source, so point prepare_data.py straight at 9.3's output dir.
# (If you added fallback sources, cp their tagged_*.json.gz into one dir first.)
python src/culture/training/continued_pretraining/prepare_data.py \
    --src_dir $DATA_ROOT/ar-amthal-cpt/data/fineweb2 \
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
# --- one-time: fetch the benchmarks (~14 MB) and build the BPB probes ---------
bash src/culture/evaluation/download_ar.sh            # -> data/eval/ar/raw/
# NOTE the path: the TAGGED shards, not train_ar/. prepare_data.py writes
# train_*.jsonl (so a *.json.gz glob matches nothing) AND projects onto
# KEEP_FIELDS = (text, source, matched_idioms), i.e. it drops `url` — so there
# would be no Wikipedia titles to exclude. build_ar_probes.py now hard-errors on
# both mistakes rather than silently writing an undecontaminated probe.
python src/culture/evaluation/build_ar_probes.py \
  --out_dir data/eval/ar \
  --exclude_urls "$DATA_ROOT/ar-amthal-cpt/data/*/tagged_*.json.gz"

# verify the loaders before burning GPU time (expected counts printed by download_ar.sh)
PYTHONPATH=src python -c "from culture.evaluation.tasks_ar import LOADERS_AR as L; \
  [print(k, len(f(ar_data_dir='data/eval/ar/raw').examples)) for k,f in L.items()]"

# --- run it -------------------------------------------------------------------
BASE_MODEL=/path/Qwen3.5-9B CPT_MODEL=/path/qwen3p5-9b-ar-cpt \
  bash src/culture/evaluation/run_eval_ar.sh          # Dim 2 BPB + Dim 3/4 MC + delta
# or on slurm, one checkpoint per job:
sbatch src/culture/evaluation/eval_ar.slurm base
sbatch src/culture/evaluation/eval_ar.slurm cpt

# English-retention check (Dim 1) is language-agnostic, reuse the existing script:
bash src/culture/evaluation/run_lm_eval.sh
```

Read the `primary` field in `summary.json`, not `acc` — see D5.4. `eval_ar.slurm` sets
`HF_HUB_OFFLINE=1`, so both one-time commands above must have been run beforehand on a
networked node.

### 9.7 Order of operations, and what each step needs

| # | Step | Needs network | Needs GPU | Needs API key | Rough time |
|---|---|---|---|---|---|
| 9.1 | env | ✅ | — | — | 10 min |
| 9.2 | KB — Option A download | ✅ | — | — | 1 min |
| 9.2 | KB — Option B rebuild | ✅ | — | `METAGEN_API_KEY` | ~45 min |
| 9.3 | filter + tag corpus | ✅ (streams) | — | — | hours, per source |
| 9.4 | reshard | — | — | — | minutes |
| 9.5 | train | — | ✅ 4×8 | — | days |
| 9.6 | benchmarks + probes (one-time) | ✅ | — | — | 5 min |
| 9.6 | evaluate | — | ✅ 1 | — | hours |

Two ordering constraints that are easy to get wrong:
- **9.2 before 9.3.** The matcher reads the KB; an unrepaired KB silently loses recall.
- **9.3 before 9.6's probe build.** The Wikipedia probe excludes titles found in the tagged
  shards, so those shards must already exist.

---

## 10. Progress log

| Date | Event |
|---|---|
| 2026-08-22 | Discovery done; MetaGen chat + APE embeddings verified; plan created |
| 2026-08-22 | Phase 2 enrichment: 10,386 calls, 0 failures, 360 hallucinated entities dropped |
| 2026-08-22 | Phase 3 A1–A4 + A7 complete; A5/A6 running |
| 2026-08-22 | Phase 4 complete: 5 corpora accepted, 4 discarded with measured evidence |
| 2026-08-22 | Phase 6 complete: matcher measured 0 → 142 docs/4k; Tier 2 disabled on precision evidence; CPT config + slurm added |
| 2026-08-22 | Phase 5 complete: 26 benchmarks screened (≥5 examples each), 8 tasks + 2 BPB probes implemented in `tasks_ar.py`; every Arabic *idiom* benchmark discarded as KB-contaminated; verified end-to-end with Qwen3-0.6B |
| 2026-08-22 | Phase 7: 22 Arabic artifacts published to HuggingFace (additive); commits `e6581f4` + `c4fc35e`. **`git push` blocked by fwdproxy — user must run it.** |
| 2026-08-22 | A6 complete (97/150 top Arabic entities shared with English). A5 re-running after a 504 and a wrong-key 401; retry + fail-fast added to `embed_cached` |
| 2026-08-22 | **Phase 3 complete.** A5 finished: raw ar↔en embedding matching yields 21 pairs (no Arabic meaning exceeds 0.607 similarity); translating the Arabic side first gives **363 pairs**, sample-validated. All analyses done |
