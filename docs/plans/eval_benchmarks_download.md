# Downloading the Hindi evaluation benchmarks

Exact download instructions for the four Hindi language/culture/idiom benchmarks
used by `src/culture/evaluation/`. Each section gives the verified command, the
expected schema, and where the eval loader expects the file.

Legend: **[VERIFIED]** confirmed against the live source; **[INSPECT]** confirm
by looking at the file/features after download (gated or column-order caveat).

Default local layout (matches `run_eval.sh`, `DATA_DIR=data/eval/hi`):

```
data/eval/hi/
├── mabl_hi.csv          # MABL Hindi
├── global_piqa_hi.tsv   # Global PIQA Hindi (Devanagari)
└── idiomce_hi.jsonl     # IdiomCE (you assemble this — see §4)
# MILU is pulled from HuggingFace at run time (gated), no local file needed.
```

Prereqs:
```bash
pip install "huggingface_hub[cli]" datasets
huggingface-cli login          # needed for MILU (gated)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # run_eval reads HF_TOKEN
```

---

## 1. MABL (figurative-meaning inference) — `mabl`

**Source:** `github.com/simran-khanuja/Multilingual-Fig-QA` (Kabra et al., ACL
Findings 2023). **No HuggingFace mirror of the Hindi split** — the `SEACrowd/mabl`
HF dataset contains only Indonesian/Javanese/Sundanese.

**Hindi file:** `langdata/hi.csv` (1,000 items, test-only). **[VERIFIED]** path
via the SEACrowd loader (`_URL = ".../main/langdata/" + "{iso}.csv"`, Hindi ISO = `hi`).

```bash
# just the Hindi file:
curl -L -o data/eval/hi/mabl_hi.csv \
  https://raw.githubusercontent.com/simran-khanuja/Multilingual-Fig-QA/main/langdata/hi.csv
# or clone the repo: git clone https://github.com/simran-khanuja/Multilingual-Fig-QA.git
#   -> Multilingual-Fig-QA/langdata/hi.csv
```

**Schema:** logical fields `startphrase`, `ending1`, `ending2`, `labels` (0/1 =
index of the correct ending).

> **[INSPECT] Column order varies by language** in this repo (e.g. Indonesian is
> `ending1,ending2,labels,startphrase`). Check the header of `hi.csv` before use:
> ```bash
> head -1 data/eval/hi/mabl_hi.csv
> ```
> The loader (`tasks.load_mabl`) reads by **column name** (with aliases), so any
> order works as long as the header names match; if `hi.csv` has no header or odd
> names, rename columns to `startphrase,ending1,ending2,labels`.

---

## 2. MILU (cultural-knowledge exam QA) — `milu`

**Source:** `ai4bharat/MILU` on HuggingFace. **Gated** (accept terms on the page +
HF token). License CC BY 4.0. **[VERIFIED]** from the dataset card.

- **Config for Hindi:** `"Hindi"` (full English name, **not** `"hi"`), passed as `data_dir`.
- **Splits:** `test` (evaluate) and `validation` (few-shot pool). No train split.

The eval loader pulls it automatically at run time (no local file needed) — just
authenticate. To pre-download / inspect:

```python
from datasets import load_dataset
test = load_dataset("ai4bharat/MILU", data_dir="Hindi", split="test", token=True)
val  = load_dataset("ai4bharat/MILU", data_dir="Hindi", split="validation", token=True)
print(test.features)          # <-- confirm field names (see INSPECT below)
```
```bash
huggingface-cli download ai4bharat/MILU --repo-type dataset --include "Hindi/*"
```

> **[INSPECT] Exact field names are behind the gate** — the card shows no example
> row and the schema endpoint requires auth. After loading, run `test.features`
> and confirm they match what `tasks.load_milu` expects: a question field
> (`question`), options (a list under `options`/`choices`, or `option1..option4`),
> and an answer (`answer`/`target`/`label` as letter, index, or option text). The
> loader is defensive across these, but verify once and adjust the template/aliases
> in `tasks.py` if MILU uses different names.

To use MILU's validation split for leak-free few-shot, dump it to a local file and
pass `--milu_fewshot_path`:
```python
val.to_json("data/eval/hi/milu_hi_val.jsonl", force_ascii=False, lines=True)
```

---

## 3. Global PIQA (cultural physical commonsense) — `global_piqa`

**Source:** `mrlbenchmarks/global-piqa-nonparallel` on HuggingFace. Openly
downloadable (CC BY-SA 4.0). **[VERIFIED]** — headers + rows read directly.

- **Hindi Devanagari subset:** config `hin_deva`, file `data/nonparallel_hin_deva.tsv`
  (100 rows). (`hin_latn` = romanized.)
- **Columns:** `prompt`, `solution0`, `solution1`, `label` (0/1), plus
  `language, eng_translated0/1, approx_cultural_score, llm_used, example_id, supplement`.

```python
from datasets import load_dataset
ds = load_dataset("mrlbenchmarks/global-piqa-nonparallel", "hin_deva")["test"]
```
```bash
# as a local TSV for the loader:
huggingface-cli download mrlbenchmarks/global-piqa-nonparallel \
  --repo-type dataset --include "data/nonparallel_hin_deva.tsv" \
  --local-dir data/eval/hi/_gpiqa
cp data/eval/hi/_gpiqa/data/nonparallel_hin_deva.tsv data/eval/hi/global_piqa_hi.tsv
# raw URL: https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel/resolve/main/data/nonparallel_hin_deva.tsv
```

> **License note:** the card restricts use to **LLM evaluation only — no training**.
> Fine for this eval; do not fold it into the CPT corpus.
>
> A 4-option **parallel** split also exists (`mrlbenchmarks/global-piqa-parallel`,
> `data/parallel_hin_deva.tsv`) with columns `solution0..solution3`. The loader
> handles 2 options; if you switch to the parallel split, it reads `solution0/1`
> only — use the non-parallel file above unless you extend the loader.

---

## 4. IdiomCE (En→Hi idiomatic translation) — `idiomce`

**No public data or code release. [VERIFIED]** (arXiv 2505.21937 has no repo link;
nothing on GitHub/HF). You must **assemble the eval JSONL yourself** from source
idiom resources. Target format (one line per item):

```json
{"source": "Don't count your chickens before they hatch.", "idiom_en": "count your chickens before they hatch", "reference": "<optional Hindi>"}
```
Only `source` is required; `reference`/`idiom_en` make the OpenAI judge
reference-guided (see the eval README).

**Where to get English idiom sentences (+ Hindi references):**

| Resource | Status | Get it |
|---|---|---|
| **Samanantar** (Indic parallel corpus) | **[VERIFIED] public** | `load_dataset("ai4bharat/samanantar", "hi")` — filter en side for idioms |
| **MAGPIE** (English potentially-idiomatic, 56K) | public | `github.com/hslh/magpie-corpus` (via ACL `2020.lrec-1.35`); 400 used by IdiomCE |
| Agrawal et al. 2018 (LREC `L18-1048`) en↔7 Indian-lang idioms | **[INSPECT]** no confirmed download | read PDF `aclanthology.org/L18-1048.pdf` for availability, or contact IIIT-H LTRC |
| Thakre et al. 2018 (Hi→En idiomatic sentences) | **[INSPECT]** no public dataset found | small journal paper; no repo located |

### Construction procedure (what we actually build)

Since IdiomCE's own eval set is unreleased, we build a **functionally-equivalent**
En→Hi idiomatic-translation eval from the project's own English idiom KB. The task
is unchanged: *an English sentence containing an idiom → model translates to Hindi
→ OpenAI judge rates idiomatic adequacy* (reference-less, like the IdiomCE paper).

Implemented by **`src/culture/evaluation/build_idiomce_eval.py`**. Steps:

1. **Load** the English idiom KB
   (`culture/data/idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl`),
   fields `idiom`, `figurative_meanings`, `literal_meanings`.
2. **Filter** to idioms with a non-empty figurative meaning and ≥2 words (drops
   trivial single-word entries), and de-duplicate by idiom text.
3. **Deterministically sample** `--num_samples` idioms (fixed `--seed`).
4. **Generate one English sentence per idiom** with an LLM (`ChatModel`): a natural,
   realistic sentence that *uses* the idiom figuratively **without explaining it**.
5. **Validate** that the generated sentence actually contains the idiom (lenient,
   inflection-tolerant content-word match) and meets a min-length check; failures
   are dropped (or kept with a flag) and counted.
6. **(optional) `--add_reference`** — generate an idiomatic Hindi reference with the
   LLM so the judge can run reference-guided. This reference is *LLM-generated, not
   gold* — it anchors the judge but is not ground truth; omit it to stay strictly
   reference-less like the paper.
7. **Write** `data/eval/hi/idiomce_hi.jsonl` in the schema above (`source`,
   `idiom_en`, `figurative_meaning`, optional `reference`), plus a run report
   (requested / generated / validation-failure counts).

```bash
# ~500 items via GPT-4o, reference-less (paper-faithful):
python -m culture.evaluation.build_idiomce_eval \
    --idiom_path culture/data/idioms/en/idioms_merged_llm_formatted_figurative_only.jsonl \
    --output_path data/eval/hi/idiomce_hi.jsonl \
    --num_samples 500 --model gpt-4o --provider openai

# add an LLM-generated Hindi reference for reference-guided judging:
python -m culture.evaluation.build_idiomce_eval --num_samples 500 --add_reference
```

**Alternative source (no generation):** instead of generating sentences, use
*real* idiom-bearing English sentences from **MAGPIE**. This is implemented by
`build_idiomce_from_magpie.py` — purely local, no LLM/network. It keeps
idiomatic-usage instances (label `i`, high confidence), extracts the PIE sentence
(`context[2]`, validated via `offsets`), lightly detokenizes it, dedupes to one
sentence per idiom type, and writes the loader schema **reference-less** (paper-
faithful). Download MAGPIE first (`git clone https://github.com/hslh/magpie-corpus`).

```bash
python -m culture.evaluation.build_idiomce_from_magpie \
    --magpie_path data/eval/hi/_magpie/MAGPIE_filtered_split_random.jsonl \
    --output_path data/eval/hi/idiomce_hi.jsonl --num_samples 400
```

The generator (`build_idiomce_eval.py`) is the other path; it reuses the repo's
English idiom KB but needs that KB mounted + an OpenAI key. **Samanantar** can't
practically supply references here — MAGPIE's BNC sentences won't appear verbatim
in it — so the reference-less MAGPIE build above is the recommended offline path.

> **Do not use** `github.com/amazon-science/idiom-mt` — it is a German–English
> idiom set (Fadaee et al. 2018), unrelated to IdiomCE.

---

# Downloading the Chinese evaluation benchmarks

Exact download instructions for the four Chinese idiom/culture benchmarks used by
`src/culture/evaluation/tasks_zh.py` (loaders `chid`, `chengyu_bench`, `cmmlu`,
`ccpm`) — the Chinese chengyu CPT analogue of the Hindi Dim-3 tasks. Same
legend: **[VERIFIED]** confirmed against the live source; **[INSPECT]** confirm
after download (schema unverified — GitHub was network-blocked at authoring time).

Default local layout (matches `run_eval_zh.sh`, `DATA_DIR=data/eval/zh`), created
by `src/culture/evaluation/download_zh.sh`:

```
data/eval/zh/
├── chid_valid.jsonl     # ChID (optional local copy; else loads from HF)
├── cmmlu/               # CMMLU repo (optional; else loads from HF)
├── ChengyuBench/        # git clone sofyc/ChengyuBench   -> --chengyu_bench_dir
└── CCPM/                # git clone THUNLP-AIPoet/CCPM    -> --ccpm_path <the .jsonl>
```

All four tasks are base-model MULTIPLE-CHOICE / CLOZE, scored by log-likelihood
(no OpenAI judge). Neither HF source is gated; **no `HF_TOKEN` required** (export
one only if you hit HF rate limits). CMMLU is a *script* dataset, so the loader
passes `trust_remote_code=True` and falls back to the parquet mirror
`lmlmcat/cmmlu` on newer `datasets`.

One-shot download:
```bash
DATA_DIR=data/eval/zh bash src/culture/evaluation/download_zh.sh
```

---

## 5. ChID (chengyu cloze) — `chid`

**Source:** HF `thu-coai/chid` (Zheng et al., ACL 2019, arXiv 1906.01265); also
`load_dataset("clue", "chid")`. Openly downloadable (Apache-2.0). **[VERIFIED]**
from the dataset card.

- **Splits:** `train` / `validation` / `test`; use **`validation`** — it carries
  gold answers. (The loader default is `hf_split="validation"`.)
- **Schema [INSPECT]:** the HF viewer shows a single `text` column whose value is a
  **JSON object** with `candidates` (list of 10 candidate chengyu) and `content`
  (list of passage strings). Blanks are inline markers `#idiomNNNNNN#` (e.g.
  `#idiom000000#`); a passage may contain several. The gold field is not shown in
  the viewer — the original ChID release uses `groundTruth` (list of idiom strings,
  aligned to the blanks in order); CLUE-chid uses `answers`/`answer`. The loader
  (`tasks_zh.load_chid`) is defensive across all of these and also unwraps the
  `text`-JSON column automatically.

> Note: the task brief said "7-way", but the live `thu-coai/chid` has **10
> candidates** per blank. The loader uses `len(candidates)`, so either works.

```python
from datasets import load_dataset
ds = load_dataset("thu-coai/chid", split="validation")
print(ds.features)                    # confirm content/candidates/groundTruth
ds.to_json("data/eval/zh/chid_valid.jsonl", force_ascii=False, lines=True)
```

**Scoring:** for each blank, split the passage into LEFT / RIGHT (other blanks in
the same passage → `____`); context = LEFT, continuation = `candidate + RIGHT`
(no leading space; Chinese is unspaced). All candidates are 4-char chengyu with an
identical RIGHT, so **`acc` (raw summed log-prob) is the primary metric** (RIGHT
cancels in the arg-max). `score_mode="continuation"` — both `acc` and `acc_norm`
are emitted; read `acc`.

---

## 6. Chengyu-Bench (connotation + appropriateness) — `chengyu_bench`

**Source:** GitHub `sofyc/ChengyuBench` (arXiv 2506.18105). `git clone`-only
(JSON files). **[INSPECT] — schema UNVERIFIED:** GitHub was network-blocked at
authoring time, so the exact JSON field/file names could not be confirmed. The
loader (`tasks_zh.load_chengyu_bench`) is written defensively from the paper's
task descriptions; **confirm the real names after cloning** and adjust the field
aliases / label maps in `tasks_zh.py` (`_CHENGYU_BENCH_ALIASES`,
`_connotation_gold`, `_appropriateness_gold`) if they differ.

```bash
git clone https://github.com/sofyc/ChengyuBench.git data/eval/zh/ChengyuBench
ls data/eval/zh/ChengyuBench        # <-- confirm file names + open one to see fields
```

Two binary subtasks (pick one via `--chengyu_bench_subtask`):

- **`connotation`** — an item has an idiom + a positive/negative label. Template
  `成语「{idiom}」的感情色彩是：`, options `[" 褒义", " 贬义"]` (positive / negative),
  gold from the label. Idiom aliases: `idiom/chengyu/word/成语/query`; label
  aliases: `label/connotation/sentiment/polarity/感情色彩/answer`.
- **`appropriateness`** — a passage with the target idiom marked (e.g. `##idiom##`)
  + a correct/wrong label. Template `{passage}\n上文中成语的使用是否恰当？答：`,
  options `[" 恰当", " 不恰当"]`. Passage aliases: `passage/content/sentence/text/context`;
  label aliases: `label/appropriate/correct/恰当/answer`.

The loader finds the subtask file by name (any json/jsonl whose name contains a
subtask keyword) under `--chengyu_bench_dir`; or pass a specific file as
`data_path`. `score_mode="continuation"`.

---

## 7. CMMLU (China-specific cultural subjects) — `cmmlu`

**Source:** HF `haonan-li/cmmlu` (Li et al., 2023). Not gated. **[VERIFIED]** —
67 subject configs, `dev` + `test` splits; **script dataset → `trust_remote_code`**.

- **Columns:** `Question, A, B, C, D, Answer` (Answer is a letter). MMLU-style.
- **Default subjects (16 China-specific):** `ancient_chinese, chinese_history,
  chinese_literature, chinese_civil_service_exam, chinese_driving_rule,
  chinese_food_culture, chinese_foreign_policy, chinese_teacher_qualification,
  construction_project_management, elementary_chinese, elementary_commonsense,
  ethnology, high_school_politics, modern_chinese, traditional_chinese_medicine,
  marxist_theory` — override with `--cmmlu_subjects a,b,c`.
- **Few-shot** exemplars come from each subject's own `dev` split (5-shot default,
  `--cmmlu_num_fewshot`), so there is no test leakage.

Loads live at eval time; to pre-download the whole repo:
```bash
huggingface-cli download haonan-li/cmmlu --repo-type dataset \
  --local-dir data/eval/zh/cmmlu
```
```python
from datasets import load_dataset
ds = load_dataset("haonan-li/cmmlu", "ancient_chinese", split="test",
                  trust_remote_code=True)
print(ds.features)                    # Question, A, B, C, D, Answer
```

> If the script loader breaks on a newer `datasets`, the loader auto-falls back to
> the parquet mirror `lmlmcat/cmmlu`. Scored by log-likelihood on the answer
> **letter** (`score_mode="letter"`), so `acc == acc_norm`.

---

## 8. CCPM (Chinese Classical Poetry Matching) — `ccpm`

**Source:** GitHub `THUNLP-AIPoet/CCPM` (Li et al., 2021, arXiv 2106.01979).
`git clone`-only (JSONL). **[INSPECT]** — GitHub was network-blocked at authoring
time; the schema below is per the paper and could not be verified live.

- **Schema (per line):** `{"translation": <modern paraphrase>, "choices": [4
  classical lines], "answer": <index>}`.

```bash
git clone https://github.com/THUNLP-AIPoet/CCPM.git data/eval/zh/CCPM
ls data/eval/zh/CCPM                 # <-- confirm the JSONL path (e.g. data/test.jsonl)
```

Point the loader at the JSONL with `--ccpm_path data/eval/zh/CCPM/<...>.jsonl`.
Template `现代文：{translation}\n对应的诗句是：`, options `[" " + line]`, gold =
`answer` index. Options differ in length → **`acc_norm`** is the fair metric
(`score_mode="continuation"`).

---

## Environment caveats (for reproducing this doc's verification)

Verified against reachable sources (HuggingFace card/files, arXiv, the SEACrowd
loader script). GitHub, `raw.githubusercontent`, and ACL Anthology were network-
blocked at verification time, so MABL's exact column order, MILU's gated field
names, and the Agrawal/Thakre download links are marked **[INSPECT]** rather than
asserted — confirm those after you can reach the sources / accept the gate.
