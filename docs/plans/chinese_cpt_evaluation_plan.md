# Evaluation plan: Chinese chengyu continued-pretrained checkpoint vs. untrained baseline

## Objective

Measure whether continued pretraining (CPT) on the Chinese **chengyu (成语)**
corpus improves **Chinese cultural / figurative-language understanding** — the
goal of the project's pipeline — **without catastrophic forgetting** of general
capability. This mirrors the Hindi case study for the Chinese branch.

The evaluation is a paired A/B: every metric is run on **both** checkpoints at
**identical settings** (same prompts, shot counts, scoring), and we report the
delta (Δ = CPT − base). Both are **base (non-instruct)** checkpoints, so
multiple-choice tasks are scored by **base-model log-likelihood**, not
instruction following. The three headline claims we want the numbers to support:

1. **Retention** — general English benchmarks stay flat (Δ ≈ 0).
2. **Adaptation** — Chinese language-modeling loss ↓ on the CPT model.
3. **Targeted gain** — large Δ on the Chinese idiom + culture benchmarks.

This plan follows the **four-category** Evaluation Protocol defined in the paper
(`OverleafCultureInFigurativeLanguage/colm2026_conference.tex`, §"Evaluation
Protocol" of the Hindi case study), reusing that exact category structure — **not**
the older Dim1/Dim2/Dim3 numbering — with the Chinese benchmarks substituted for
the Hindi ones.

## Models under test

| Role | Checkpoint | Notes |
|---|---|---|
| Baseline | `Qwen3.5-9B` (base) | `/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B` |
| CPT | `qwen3p5-9b-zh-cpt` | full-param `pt` stage |

CPT corpus: `jiviteshjn/fineweb-edu-zh-chengyu-cpt` (~3.74M docs, ~7.8B tokens) —
FineWeb-Edu Chinese documents filtered/augmented for chengyu, tag-augmented with
idiom metadata as in the Hindi pipeline. (Training recipe:
`src/culture/training/continued_pretraining/README.md`.)

---

## Evaluation categories (mirrors the paper's four-category protocol)

| # | Category | Purpose | Primary metric | Status |
|---|---|---|---|---|
| 1 | General English (retention) | Forgetting check | acc / EM / pass@1, PPL/BPB | **implemented** (shared) |
| 2 | Chinese language modeling (adaptation) | Adaptation | BPB, PPL | **implemented** (`perplexity.py`) |
| 3 | Chinese idiom understanding | **Target** | `acc` / `acc_norm` | **implemented** (`tasks_zh.py`) |
| 4 | Chinese general cultural competence | **Target** | `acc` / `acc_norm` | **implemented** (`tasks_zh.py`) |

All four are implemented under `src/culture/evaluation/`. Categories 1–2 reuse the
existing (language-agnostic) tooling unchanged; categories 3–4 are the new Chinese
task loaders (`tasks_zh.py`, wired into `run_eval.py`).

---

### (1) General English (retention)

Because the CPT is Chinese-only, English ability should be preserved (Δ ≈ 0).
Shared with the Hindi case study; no Chinese-specific code.

| Benchmark | Capability | Setup | Metric | Tool |
|---|---|---|---|---|
| **WikiText-103** | English language modeling | official test split | PPL, BPB | `perplexity.py` |
| **MMLU** | Knowledge (57 subj.) | 0-shot | acc | `run_eval.py` (`mmlu`) or lm-eval |
| **BoolQ** | Reading comp (yes/no) | 0-shot | acc | `run_eval.py` (`boolq`) or lm-eval |
| **GSM8K** | Grade-school math | 8-shot CoT | EM | `run_lm_eval.sh` |
| **HumanEval** | Code generation | 0-shot | pass@1 | `run_lm_eval.sh` |

- WikiText-103 via `perplexity.py --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test`.
- MMLU / BoolQ can run inside the eval module (`--tasks mmlu,boolq`) or via
  `lm-evaluation-harness`; GSM8K-CoT + HumanEval via `run_lm_eval.sh`.
- Expected: Δ ≈ 0 (a Chinese-only CPT should retain English).

### (2) Chinese language modeling (adaptation)

Direct signal that CPT changed the distribution the model fits — the Chinese
analogue of the Hindi in-domain + Samanantar probes.

| Corpus | Role | Reads |
|---|---|---|
| **In-domain slice** of `jiviteshjn/fineweb-edu-zh-chengyu-cpt` | Contaminated upper bound | PPL/BPB should drop sharply on CPT |
| **Chinese Wikipedia** (`wikimedia/wikipedia`, config `20231101.zh`) — the Samanantar-analogue | Disjoint, in-distribution probe | PPL/BPB should drop on CPT |

- Report **bits-per-byte (BPB)** as primary (tokenizer-agnostic) plus PPL.
- The in-domain slice overlaps the training corpus (a **contaminated** upper
  bound — reserve a held-out slice *before* training for a clean number). Chinese
  Wikipedia is disjoint from the CPT corpus but shares the web-Chinese
  distribution, so it is an **in-distribution adaptation probe** (Samanantar
  analogue), not a strict held-out test — a non-chengyu FineWeb-2-zh slice works
  equally well as the disjoint probe.
- Both via `perplexity.py`; same held-out documents, fixed context/stride, for
  base and CPT. Lower is better; expect a large drop on CPT.

### (3) Chinese idiom understanding — **target capability**

Two native benchmarks targeting figurative/idiomatic (chengyu) competence.
Implemented in `tasks_zh.py`; download via `docs/plans/eval_benchmarks_download.md`
§§5–6 + `download_zh.sh`.

| Task | Measures | Format | Scoring | Primary metric |
|---|---|---|---|---|
| **ChID** | Chengyu cloze (fill-the-blank) | N-way cloze | log-likelihood | `acc` (raw sum) |
| **Chengyu-Bench** | Connotation (褒/贬) + appropriateness (恰当/不恰当) | 2 binary subtasks | log-likelihood | `acc` |

- **ChID** splits each passage at the blank; context = LEFT, continuation =
  `candidate + RIGHT`. All candidates are 4-char chengyu sharing an identical
  RIGHT → **`acc` (raw summed log-prob) is the headline** (`score_mode="continuation"`,
  both acc/acc_norm emitted).
- **Chengyu-Bench** exposes two binary subtasks via `--chengyu_bench_subtask`.
  Schema is **[INSPECT]** (GitHub unreachable at authoring; loader is defensive —
  confirm field names after `git clone`).

### (4) Chinese general cultural competence — **target capability**

Two native benchmarks for broader, non-idiom cultural knowledge + Chinese
classical culture. Implemented in `tasks_zh.py`; download via §§7–8.

| Task | Measures | Format | Scoring | Primary metric |
|---|---|---|---|---|
| **CMMLU** (China-specific) | Chinese cultural-knowledge exam QA | 4-choice MMLU-style, 5-shot | log-likelihood (letter) | `acc` |
| **CCPM** | Chinese Classical Poetry Matching | 4-choice MC | log-likelihood | `acc_norm` |

- **CMMLU** uses the 16 China-specific subject configs (ancient_chinese,
  chinese_history, …, marxist_theory) concatenated; few-shot exemplars from each
  subject's `dev` split (no leakage). Answer-letter scoring → `acc == acc_norm`.
- **CCPM** options differ in length → **`acc_norm`** is the fair metric. Schema is
  **[INSPECT]** (GitHub-only).

---

## Methodology

- **Paired, identical settings.** Same few-shot count, prompt template, and
  scoring method for base and CPT. The `tasks_zh` templates are explicit constants
  at the top of the module — keep them fixed across both runs.
- **Report deltas.** `compare_results.py` prints the base→CPT Δ per task.
- **Base-model scoring.** All four Chinese tasks are multiple-choice / cloze
  scored by log-likelihood — no instruction following, no OpenAI judge (unlike
  Hindi IdiomCE).
- **Forgetting is multi-dimensional.** Besides accuracy, watch for output-format
  drift on the CPT model; expect the largest retention risk on code (HumanEval),
  the skill most distant from the Chinese-only training distribution.

## Tooling

| Category | Tool |
|---|---|
| 1 (English) | `perplexity.py` (WikiText) + `run_eval.py`/`run_lm_eval.sh` (MMLU/BoolQ/GSM8K/HumanEval) |
| 2 (Chinese LM) | `perplexity.py` (in-domain slice + Chinese Wikipedia) |
| 3–4 (idiom + culture) | `culture.evaluation` module — `tasks_zh.py` + `run_eval.py` + `run_eval_zh.sh` + `compare_results.py` |

## How to run

Set the two checkpoint paths once; each launcher runs base + CPT.

```bash
export BASE_MODEL=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
export CPT_MODEL=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-zh-cpt

# Download the category-3/4 data (ChID/CMMLU from HF; Chengyu-Bench/CCPM git clone)
DATA_DIR=data/eval/zh bash src/culture/evaluation/download_zh.sh

# (1) General English retention — WikiText PPL/BPB + MMLU/BoolQ (+ GSM8K/HumanEval)
python -m culture.evaluation.perplexity --model_path "$CPT_MODEL" --run_name cpt \
    --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
    --output_dir results/zh/cpt/ppl_wikitext
bash src/culture/evaluation/run_lm_eval.sh          # GSM8K-CoT + HumanEval (+ MMLU/BoolQ)

# (2) Chinese language modeling — in-domain slice (contaminated) + Chinese Wikipedia
python -m culture.evaluation.perplexity --model_path "$CPT_MODEL" --run_name cpt \
    --data_path data/eval/zh/chengyu_cpt_heldout.jsonl \
    --output_dir results/zh/cpt/ppl_zh_indomain
python -m culture.evaluation.perplexity --model_path "$CPT_MODEL" --run_name cpt \
    --hf_dataset wikimedia/wikipedia --hf_config 20231101.zh --hf_split train --limit 1000 \
    --output_dir results/zh/cpt/ppl_zh_wiki

# (3)+(4) Chinese idiom + cultural competence (ChID, Chengyu-Bench, CMMLU, CCPM)
DATA_DIR=data/eval/zh bash src/culture/evaluation/run_eval_zh.sh
```

Outputs land under `results/zh/{base,cpt}/`. `run_eval_zh.sh` prints the base→CPT
delta for categories 3–4; for categories 1–2 compare the per-task PPL/BPB JSON
(lower is better). Data download: `docs/plans/eval_benchmarks_download.md`.

> **Category-2 contamination note:** the in-domain slice is drawn from the CPT
> corpus and is therefore *contaminated* — it upper-bounds the adaptation effect.
> Reserve a held-out `fineweb-edu-zh-chengyu-cpt` slice **before/outside training**
> for a clean number; Chinese Wikipedia (disjoint, in-distribution) is the
> Samanantar-analogue adaptation probe.

## Success criteria

The CPT run is a success if, vs. the base model:

- **(3)+(4) target:** clear positive Δ on ChID + Chengyu-Bench (idiom) and on
  CMMLU + CCPM (culture) — the primary result.
- **(2) adaptation:** Chinese PPL/BPB ↓ (large on the in-domain slice, meaningful
  on Chinese Wikipedia).
- **(1) no forgetting:** Δ ≈ 0 (within noise) on English benchmarks; watch
  HumanEval, the likeliest regression.

## Results template (fill after runs)

| Category | Benchmark | Metric | Base | CPT | Δ |
|---|---|---|---|---|---|
| 1 | WikiText-103 | PPL | | | |
| 1 | WikiText-103 | BPB | | | |
| 1 | MMLU (0-shot) | acc | | | |
| 1 | BoolQ (0-shot) | acc | | | |
| 1 | GSM8K (8-shot CoT) | EM | | | |
| 1 | HumanEval (0-shot) | pass@1 | | | |
| 2 | chengyu-cpt in-domain | BPB | | | |
| 2 | Chinese Wikipedia | PPL | | | |
| 2 | Chinese Wikipedia | BPB | | | |
| 3 | ChID | acc | | | |
| 3 | Chengyu-Bench (connotation) | acc | | | |
| 3 | Chengyu-Bench (appropriateness) | acc | | | |
| 4 | CMMLU (China-specific) | acc | | | |
| 4 | CCPM | acc_norm | | | |

## Status

All four categories are implemented in `src/culture/evaluation/`:

- **(1) General English** — `perplexity.py` (WikiText) + `run_eval.py`
  (`mmlu`/`boolq`) / `run_lm_eval.sh` (GSM8K-CoT/HumanEval). Shared, unchanged.
- **(2) Chinese LM** — `perplexity.py` (in-domain slice + Chinese Wikipedia).
- **(3) Chinese idiom** — `tasks_zh.py` loaders `chid`, `chengyu_bench` +
  `run_eval_zh.sh`.
- **(4) Chinese culture** — `tasks_zh.py` loaders `cmmlu`, `ccpm` + `run_eval_zh.sh`.

Remaining prerequisites (not code):

- Download the category-3/4 data (`docs/plans/eval_benchmarks_download.md` §§5–8 /
  `download_zh.sh`), and **confirm the Chengyu-Bench + CCPM JSON/JSONL schemas
  after `git clone`** — those loaders are defensive but their exact field names
  were unverified (GitHub was network-blocked at authoring time).
- **Reserve a held-out `fineweb-edu-zh-chengyu-cpt` slice before/outside training**
  for a clean category-2 adaptation number.
