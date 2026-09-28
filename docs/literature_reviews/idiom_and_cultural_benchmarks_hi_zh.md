# Idiom & Cultural-Competence Benchmarks for Hindi and Chinese — A Survey

**Drafted:** 2026-08-28. **Purpose:** expand evaluation coverage for the hi/zh arms of the CPT +
instruction-tuning study, which are currently much thinner than the Arabic arm.

**Scope:** two categories per language — **(1) idiom / figurative-language benchmarks** and
**(2) cultural-competence benchmarks**. Multilingual benchmarks are covered where hi or zh is one
of the languages.

**Method:** web + arXiv search (2026-08-28), plus a programmatic HuggingFace sweep (114 candidate
repos across 30 queries). **Every row count marked ✓ was read from the HF dataset-viewer `/size`
or `/splits` endpoint directly**; counts marked *(paper)* come from the paper and were not verified
against a release. Repos that are gated or have no HF release are flagged in §7.

---

## 0. What we already evaluate

| Lang | Idiom / figurative | Cultural competence |
|---|---|---|
| **Hindi** | MABL (hi), IdiomCE (idiomatic MT, judged) | MILU, Global-PIQA (hi) |
| **Chinese** | ChID (chengyu cloze), Chengyu-Bench (connotation + appropriateness) | CMMLU (16 China-specific subjects), CCPM (classical poetry) |
| Arabic *(for contrast)* | kinayat_meaning, kinayat_cloze, ar_figurative | arabculture, arabic_cultural_qa, arabicmmlu, dzirieval, global_piqa_ar, alyah |

Arabic has 3 idiom + 6 cultural tasks; Hindi has 2 + 2 and Chinese 2 + 2. This survey closes that gap.

---

## 1. Hindi — idiom / figurative-language benchmarks

**The headline finding: Hindi is badly served here.** No large, dedicated Hindi idiom benchmark
exists. Every recent multilingual idiom benchmark either skips Hindi entirely or gives it a small
slice. This is a real result for the paper, not a search failure — see §7.

### 1.1 MABL — Metaphors Across Borders and Languages *(already in use)*
- Hindi subset of a multilingual figurative-inference set; 2-choice "sentence → what it means".
- Loaded locally in `src/culture/evaluation/tasks.py` (`MABL_TEMPLATE`); no clean HF release.
- **Keep as the primary Hindi figurative task.**

### 1.2 IdiomCE *(already in use)*
- English→Hindi **idiomatic translation**, generative, LLM-judged (`judge_idiomce.py`).
- Built locally via `build_idiomce_eval.py` / `build_idiomce_from_magpie.py`.

### 1.3 `sam749/hindi-proverbs` — **2,043 rows** ✓
- Columns: `proverb`, `meaning`. A raw **proverb→meaning lexicon**, not a benchmark: no distractors,
  no contexts, no splits.
- **Usable, but you must build the task**: it is the closest Hindi analogue to our Arabic
  `kinayat_meaning`. Convert to N-way multiple choice by sampling distractor meanings from other
  proverbs, exactly as we did for Arabic. **Check overlap against the Hindi idiom KB used for CPT
  filtering before using it** — if the CPT corpus was filtered on these proverbs, this becomes a
  memorization probe, not a comprehension test. That distinction is already RQ3 in the paper plan.

### 1.4 HiSlang-4.9k *(paper)*
- ~4,900 items for **Hindi slang detection/identification**. Adjacent to figurative language rather
  than idiom comprehension. Low priority; no verified release found.

### 1.5 What does NOT cover Hindi (checked, negative results worth recording)
- **MIDI** (arXiv:2606.02147, ACL 2026) — 18 languages/dialects of idioms in sentence *and dialogue*
  context, with paired figurative/literal usage. Covers **Chinese** but its Indic languages are
  **Kannada, Telugu, Tamil — not Hindi.**
- **IdiomX** (arXiv:2606.02584) — 190K+ contextualized examples over 12K+ idioms, 4 tasks. Only
  **English, Arabic, French**. No Hindi, no Chinese.
- **SiNFluD** (arXiv:2605.01323) — Sindhi. **FFE-Hallu** (arXiv:2601.20105) — Persian.
  **BRoverbs** (arXiv:2509.08960) — Portuguese. **VIVID** (arXiv:2608.03095) — Vietnamese.
  **ProverbEval** — Ethiopian languages + English. **Fig-QA**, **FLUTE**, **MAGPIE** — English.
- **MAPS** — proverbs-in-conversation, binary interpretation choice; multilingual but Hindi coverage
  unconfirmed — **verify before planning around it**.

---

## 2. Hindi — cultural-competence benchmarks

Much better supplied than idioms. Four strong candidates, all verified on HF.

### 2.1 SANSKRITI — **21,853 rows** ✓ (`13ari/Sanskriti`)
- arXiv:2506.15355, Findings of ACL 2025. QA pairs spanning **28 states + 8 union territories**;
  the largest Indian-cultural-knowledge set.
- Schema ✓: `state`, `attribute`, `question`, `option1..4`, `answer`, `question_type`, plus a
  source link per row. **4-way MCQ — drops straight into our existing `MCTask` loader.**
- **Top recommendation for Hindi cultural competence.** Caveat: questions appear to be
  English-language about Indian culture; confirm language before calling it a *Hindi* benchmark —
  it may be measuring cultural knowledge rather than Hindi-language cultural competence. That is
  still useful, but it tests a different axis than MILU.

### 2.2 DRISHTIKON — **64,290 rows** ✓ (`13ari/DRISHTIKON`)
- arXiv:2509.19274. Multimodal **and** multilingual (15 Indic languages), same 4-way MCQ schema as
  SANSKRITI plus `language` and `image` columns.
- **Filter to `language == Hindi` and drop the image column** for a text-only Hindi cultural task.
  The `language` field makes this the only benchmark here that is natively Hindi-language *and* large.
- **Second recommendation**, and arguably first if the Hindi filter yields enough rows.

### 2.3 ParamBench — **17,275 rows** ✓ (`bharatgenai/ParamBench`)
- arXiv:2508.16185. Graduate-level Indic-subject exam questions (`subject`, `exam_name`,
  `question_text`, `option_a..d`, `correct_answer`). Test split only.
- Closer to MILU (exam knowledge) than to culture. **Use as a MILU-style companion**, not a
  replacement — it measures academic Indic knowledge, not cultural competence.

### 2.4 DIWALI *(paper)* — ~8K cultural concepts, 17 facets, 36 sub-regions
- arXiv:2509.17399. A **cultural-specific-items (CSI) dataset for cultural text adaptation**, not an
  MCQ benchmark — the task is adapting text to an Indian cultural context, judged generatively.
- **Different task shape**; would need a judge harness like IdiomCE's. Medium priority.

### 2.5 Others noted
- **IndicQuest** (`l3cube-pune/IndicQuest`, and `-v2`) ✓ exists — `Question`/`Answer`/`Domain`,
  factual Indic knowledge, generative. Small.
- **VIRAASAT** (arXiv:2602.18429) — Indian cultural *reasoning*; no verified release found.
- **Pragyaan** (2025), **IndicMMLU-Pro** (arXiv:2501.15747), **BhashaSutra** survey (arXiv:2604.18423)
  — background, not directly adoptable.
- **Benchmarking Hindi LLMs** (arXiv:2508.19831) — IFEval-Hi, MT-Bench-Hi, GSM8K-Hi, ChatRAG-Hi,
  BFCL-Hi. **General capability, not cultural** — but IFEval-Hi is the natural *instruction-following*
  eval for the SFT stage, which we currently lack for Hindi.

---

## 3. Chinese — idiom benchmarks

The best-served of the four quadrants.

### 3.1 ChID *(already in use)* — chengyu cloze
### 3.2 Chengyu-Bench *(already in use)* — arXiv:2506.18105, EMNLP 2025 main
- **2,937 human-verified examples over 1,765 idioms**, three tasks: Evaluative Connotation,
  Appropriateness, **Open Cloze** (no options).
- **We currently use only connotation + appropriateness.** The paper reports frontier LLMs at >95%
  on connotation, ~85% on appropriateness, but only **~40% top-1 on Open Cloze** — i.e. the task with
  the most headroom is the one we are not running. **Adding Open Cloze is the single cheapest
  eval win available**: same dataset, already downloaded, and it is generative rather than
  answer-string-scored, which strengthens the RQ3 knowledge-injection-vs-comprehension split.
- Error analysis: **57.3% of model errors are meaning misinterpretation**, then domain-adaptation
  errors — directly the capability our meaning-tagged CPT targets.

### 3.3 Xiehouyu (歇后语) — `oHenri/chinese_xiehouyu` — **14,032 rows** ✓
- Schema ✓: `riddle`, `answer`. Two-part allegorical sayings — a setup and its punchline/meaning.
- **This is a genuinely new idiom *type* we do not test at all.** Chengyu are 4-character classical
  compounds; xiehouyu are colloquial two-part riddles. A model can be good at one and bad at the other.
- Raw lexicon, not a benchmark: build N-way MCQ by sampling distractor answers, same recipe as §1.3.
- **Top new-idiom recommendation for Chinese.**

### 3.4 MIDI — Chinese subset *(paper)*
- arXiv:2606.02147, ACL 2026. Idioms in **sentence and dialogue context**, with **paired figurative
  vs. literal** usage and MCQ options; native-speaker collected and annotated.
- The figurative/literal pairing is a control we do not currently have anywhere in the eval suite —
  it isolates whether the model actually disambiguates usage rather than pattern-matching the idiom.
- **High value; verify the release and extract the `zh` slice.**

### 3.5 Lexicons (not benchmarks, but usable to build tasks)
`jamesqijingsong/chengyu` ✓, `KSmart/chinese_traditional_chengyu` ✓, `mmdjiji/bert-chinese-idioms` ✓.
Overlap-check any of these against the CPT idiom KB before use (same memorization caveat as §1.3).

---

## 4. Chinese — cultural-competence benchmarks

### 4.1 CMMLU (16 China-specific subjects) *(already in use)* · CCPM *(already in use)*
`PoetryMTEB/CCPM` ✓ **108,872 rows** — sentence-pair/label schema.

### 4.2 CHARM — arXiv:2403.14112, ACL 2024
- The first benchmark built specifically for **Chinese-specific commonsense**, and deliberately split
  into **reasoning (1,800 questions) and memorization (759 questions)** tasks.
- 7 reasoning tasks: Anachronisms Judgment, Time Understanding, Sequence Understanding, Movie & Music
  Recommendation, Sport Understanding, NLI, Reading Comprehension.
- **This is the most methodologically aligned benchmark in this document.** Its reasoning-vs-memorization
  split is *the same decomposition* as the paper's RQ3 (answer-string-injected vs. clean comprehension),
  but pre-built and peer-reviewed. Adopting it lets us cite an established split instead of defending
  our own.
- **Top recommendation for Chinese cultural competence.**

### 4.3 C-Eval — `ceval/ceval-exam` ✓, **52 subject configs**, 497 rows in the sampled config
- Chinese academic/professional exams. The standard companion to CMMLU; overlaps in spirit.
- **Medium priority** — adds breadth but not a new capability axis. Its `chinese_language_and_literature`
  and `art_studies` configs are the culturally-loaded ones.

### 4.4 WenMind — `SCUT-DLVCLab/WenMind` ✓ **4,875 rows**, NeurIPS 2024 D&B
- **Classical Chinese** (文言文) comprehension. Schema ✓ is unusually rich: `domain`, `capability`,
  `question_format`, coarse/fine-grained task labels in both zh and en, and a `difficulty` field.
- Classical Chinese is where chengyu *come from*, so this is the natural upstream-competence test for
  a model trained on chengyu meaning tags. The per-row `capability` and `difficulty` fields support
  slicing without extra annotation.
- **Second recommendation for Chinese.**

### 4.5 Multilingual sets with a Chinese slice
- **CulturalBench** — `kellycyy/CulturalBench` ✓ (`CulturalBench-Easy` / `-Hard`, **4,908 rows** in
  Hard), built by human-AI red-teaming, with a `country` column to filter. Also
  `shreyahavaldar/CulturalBench-MC` ✓ (1,141) and `Lossfunk/Multilingual_CulturalBench` ✓.
  **Filter `country == China`** for a zh slice; **the same repo gives an India slice for Hindi**, which
  makes it the one benchmark that puts hi and zh on an identical scale. Valuable for cross-language
  comparison in the paper.
- **BLEnD** — everyday cultural knowledge across cultures; includes Chinese. Verify release.
- **Global-PIQA** ✓ — `mrlbenchmarks/global-piqa-parallel`, **131 language configs**, ~103 rows each.
  We already use the hi and ar slices; **a zh slice exists and we are not using it.** Cheap add,
  and its *parallel* design means the same items across hi/zh/ar — a genuine controlled comparison.

---

## 5. Recommended additions, ranked

| # | Benchmark | Lang | Category | Why | Effort |
|---|---|---|---|---|---|
| 1 | **Chengyu-Bench Open Cloze** | zh | idiom | Already downloaded; the ~40% task where headroom lives | **Trivial** — new task fn on existing data |
| 2 | **CHARM** | zh | cultural | Pre-built reasoning-vs-memorization split == our RQ3 | Low |
| 3 | **Global-PIQA zh slice** | zh | cultural | Already wired for hi/ar; parallel items across languages | **Trivial** |
| 4 | **SANSKRITI** | hi | cultural | 21,853 ✓ 4-way MCQ, fits `MCTask` as-is | Low |
| 5 | **DRISHTIKON (hi filter)** | hi | cultural | 64,290 ✓ with a `language` column → natively Hindi | Low |
| 6 | **Xiehouyu MCQ** | zh | idiom | 14,032 ✓ — an idiom *type* we test nowhere | Medium — build distractors |
| 7 | **CulturalBench (China + India slices)** | both | cultural | Identical scale for hi and zh — enables cross-language claims | Low |
| 8 | **WenMind** | zh | cultural | Classical Chinese = upstream of chengyu; rich slicing fields | Low |
| 9 | **`sam749/hindi-proverbs` → MCQ** | hi | idiom | The only Hindi proverb resource; mirrors `kinayat_meaning` | Medium — build distractors |
| 10 | **MIDI zh slice** | zh | idiom | Figurative/literal paired control we lack entirely | Medium — verify release |

**Two cross-cutting cautions:**
1. **Decontaminate every one of these against the SFT mixtures** built in
   `plans/instruction_tuning_runbook.md` §3.2 step 6. `QCRI/ArabicCulturalQA` is already flagged there
   for Arabic; CulturalBench and C-Eval are the analogous risks for hi/zh, and Chinese SFT pools
   frequently ingest exam data.
2. **Overlap-check the lexicon-derived tasks (#6, #9) against the CPT idiom KB.** If a proverb was used
   to filter or tag the CPT corpus, scoring it back is memorization, not comprehension — the paper
   already distinguishes these (RQ3), so the tasks must be labelled accordingly.

---

## 6. Directly relevant related work

**"Figurative and Cultural Knowledge in LLMs: Investigating Cross-Domain Transfer through Fine-Tuning"**
(arXiv:2608.18361) — Arabic-only, but it runs *our experiment*: does fine-tuning on cultural data
improve figurative understanding, and vice versa?

- Figurative benchmarks: **Jawaher** (800 train / 198 test, proverbs across 20 Arabic varieties),
  **Kinayat** (150, Egyptian idiom-explanation), **FannOrFlop** (6,984, poem-explanation).
- Cultural: **AraDiCE-Culture** (180), **ArabCulture** (3,482), **Palm** (15,500 / 1,930).
- Control: **ArabicMMLU** (980).
- **Finding:** fine-tuning on *poetry* improved idiom comprehension (**+2.33%, p<0.05**) and the
  ArabicMMLU control did not reproduce the gain — so it came from figurative content, not generic
  Arabic adaptation. But **cross-domain transfer between culture and figurative language was
  "limited, inconsistent, and highly model-dependent"**, and cultural fine-tuning *degraded* proverb
  performance in Arabic-centric models.

**Why this matters for us:** it is simultaneously the closest prior work and a warning. It uses a
control benchmark to separate figurative gains from language adaptation — the same logic as our
unfiltered / filtered-untagged controls — which is a citation worth having. And its negative transfer
result predicts that our cultural and idiom benchmarks may move in *different directions*; if that
happens, it is a replicated finding, not a broken run.

Also useful: **"A Survey of Idiom Datasets for Psycholinguistic and Computational Research"**
(arXiv:2508.11828) — the reference list to check before claiming any language has no idiom benchmark.

---

## 7. Gaps and access limitations

- **Hindi has no dedicated idiom benchmark.** Confirmed by exclusion: MIDI covers Kannada/Telugu/Tamil
  but not Hindi; IdiomX is en/ar/fr; the recent figurative-language releases are Sindhi, Persian,
  Portuguese, Vietnamese, Ethiopian. Hindi idiom evaluation rests on MABL + IdiomCE + whatever we
  build from `sam749/hindi-proverbs`. **This is a publishable observation, and a reason our Hindi
  idiom results are harder to situate than the Arabic ones.**
- **Not verified against a release** (paper-only or unconfirmed): DIWALI, VIRAASAT, HiSlang-4.9k,
  BLEnD, MAPS, MIDI, CHARM's HF form (GitHub/OpenDataLab exists at `opendatalab.github.io/CHARM/`).
- **`l3cube-pune/IndicQuest`** ✓ resolves but `/size` returned no row count — check locally.
- **SANSKRITI language ambiguity** (§2.1) must be resolved before it is described as a Hindi benchmark.
- Frequently-cited Chinese "traditional culture" models/benchmarks in Chinese-language sources
  (SuperCLUE, the GB/T 45288.2—2025 national standard's history-and-humanities section) are
  **leaderboards or standards, not downloadable datasets** — not adoptable.
