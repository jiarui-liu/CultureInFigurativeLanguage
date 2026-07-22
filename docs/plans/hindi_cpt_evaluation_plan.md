# Evaluation plan: Hindi continued-pretrained checkpoint vs. untrained baseline

## Objective

Measure whether continued pretraining (CPT) on the Hindi cultural corpus improves
**Hindi cultural / figurative-language understanding** — the goal of the project's
pipeline — **without catastrophic forgetting** of general capability.

The evaluation is a paired A/B: every metric is run on **both** checkpoints at
**identical settings**, and we report the delta (Δ = CPT − base). The three
headline claims we want the numbers to support:

1. **Adaptation** — Hindi language-modeling loss ↓ on the CPT model.
2. **No forgetting** — general English benchmarks stay flat (Δ ≈ 0) or improve.
3. **Targeted gain** — large Δ on the Hindi culture/idiom benchmarks (especially
   the figurative tasks, MABL and IdiomCE).

## Models under test

| Role | Checkpoint | Notes |
|---|---|---|
| Baseline | `Qwen3.5-9B` (base) | `/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B` |
| CPT | `qwen3p5-9b-hi-cpt` | full-param `pt` stage, 3 epochs |

CPT corpus: `jiviteshjn/hi-proverbs-cpt` (~338K docs, ~1.4B tokens). Both are
**base (non-instruct)** models, so evaluation prioritizes **log-likelihood /
cloze scoring** and few-shot completion — no instruction following assumed.
(Training recipe: see `src/culture/training/continued_pretraining/README.md`.)

---

## Evaluation dimensions

| Dim | What | Purpose | Primary metric | Status |
|---|---|---|---|---|
| 1 | Language modeling | Adaptation + retention | BPB, PPL | **implemented** |
| 2 | General English | Forgetting check | acc / EM / pass@1 | **implemented** |
| 3 | Hindi culture / idiom | **Target capability** | acc_norm / idiom_score | **implemented** |

All three dimensions are implemented under `src/culture/evaluation/` (Dim 1–2 via
the perplexity module + an `lm-evaluation-harness` launcher; Dim 3 via the custom
task module).

---

### Dimension 1 — Language modeling (perplexity)

Direct signal that CPT changed the distribution the model fits.

| Metric | Corpus | Reads |
|---|---|---|
| **BPB** (primary), PPL | Held-out slice of the Hindi CPT corpus (`hi-proverbs-cpt`) | **Adaptation** — should drop on CPT |
| **BPB**, PPL | English held-out (WikiText-103) | **Retention** — should stay flat |

- Report **bits-per-byte (BPB)** as primary — it is tokenizer-agnostic (safe even
  if the tokenizer ever changes); report PPL too since the tokenizer is unchanged.
- Same held-out documents, fixed context/stride, for base and CPT.
- Expected: PPL ↓ on the proverb corpus, ≈ flat on English.

### Dimension 2 — General English capability (forgetting check)

The CPT was Hindi-only, so English general ability should be preserved (Δ ≈ 0).

| Benchmark | Capability | Setup | Metric |
|---|---|---|---|
| **MMLU** | Knowledge (57 subj.) | 0-shot | acc |
| **BoolQ** | Reading comp (yes/no) | 0-shot | acc |
| **GSM8K** | Grade-school math | 8-shot CoT | EM |
| **HumanEval** | Code generation | 0-shot | pass@1 |

### Dimension 3 — Hindi culture / idiom (target capability) — **implemented**

The four benchmarks chosen for this project. Implemented in
`src/culture/evaluation/` (see its README; download via
`docs/plans/eval_benchmarks_download.md`).

| Task | Measures | Native? | Format | Scoring | Primary metric |
|---|---|---|---|---|---|
| **MABL** | Figurative-meaning inference | native | 2-choice | log-likelihood | `acc_norm` |
| **MILU** | India cultural-knowledge exam QA | native | 4-choice MMLU-style | log-likelihood (letter) | `acc` |
| **Global PIQA** | Cultural physical commonsense | native | 2-choice | log-likelihood | `acc_norm` |
| **IdiomCE** | En→Hi idiomatic translation | native idioms | generation | OpenAI judge (1–5) | `idiom_score_mean` |

- MC tasks report `acc` (raw) and `acc_norm` (length-normalized); `acc_norm` is
  the fair number for MABL / Global PIQA (options differ in length).
- IdiomCE is reference-less by design → graded by an **OpenAI judge**
  (`gpt-4o`, temperature 0, 1–5 idiomatic-adequacy rubric + `idiom_rendered` rate).
- Biggest expected Δ here — especially MABL and IdiomCE (the figurative tasks).

---

## Methodology

- **Paired, identical settings.** Same few-shot count, prompt template, and
  scoring method for base and CPT; disclose them per benchmark. The
  `culture.evaluation` templates are explicit constants — keep them fixed across
  both runs.
- **Report deltas.** `compare_results.py` prints the base→CPT Δ per task.
- **Base-model scoring.** Multiple-choice via log-likelihood (no instruction
  following); generation only where unavoidable (GSM8K CoT, HumanEval, IdiomCE).
- **LLM-as-judge (IdiomCE).** OpenAI judge, reference-guided when a reference
  exists, single-answer 1–5 grading, temperature 0. Validate against a small
  human-labeled sample (report agreement) before trusting judge scores at scale.
- **Forgetting is multi-dimensional.** Besides accuracy, watch for output-format
  drift (language-ID, repetition) on the CPT model.

## Tooling

| Dimension | Tool |
|---|---|
| 1 (BPB/PPL) | EleutherAI `lm-evaluation-harness` (`bits_per_byte`, `word_perplexity`) |
| 2 (English) | `lm-evaluation-harness` (mmlu, boolq, gsm8k, humaneval) |
| 3 (culture/idiom) | `culture.evaluation` module (this repo) |

## How to run

Set the two checkpoint paths once; each launcher runs base + CPT.

```bash
export BASE_MODEL=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
export CPT_MODEL=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt

# Dim 1 — perplexity / BPB (Hindi held-out adaptation + WikiText retention)
HELDOUT_PATH=data/eval/hi/hi_proverbs_heldout.jsonl \
  bash src/culture/evaluation/run_perplexity.sh

# Dim 2 — general English (+ WikiText retention) via lm-evaluation-harness
bash src/culture/evaluation/run_lm_eval.sh

# Dim 3 — Hindi culture / idiom (custom task module + OpenAI judge)
DATA_DIR=data/eval/hi JUDGE_MODEL=gpt-4o \
  bash src/culture/evaluation/run_eval.sh
```

Outputs land under `results/hi/{base,cpt}/`. Dim 3's `run_eval.sh` prints the
base→CPT delta; for Dim 1/2 compare the per-task JSON (PPL/BPB: lower is better).
Data download: `docs/plans/eval_benchmarks_download.md`.

> **Dim 1 contamination note:** `HELDOUT_PATH` must be a slice **excluded from CPT
> training** (reserve it before training, or use an independent Hindi corpus).
> A slice of the already-trained corpus is contaminated and understates PPL.

## Success criteria

The CPT run is a success if, vs. the base model:

- **Dim 3 (target):** clear positive Δ on MABL and IdiomCE (figurative), and on
  MILU/Global PIQA (cultural) — the primary result.
- **Dim 1 (adaptation):** Hindi proverb-corpus PPL/BPB ↓.
- **Dim 2 (no forgetting):** Δ ≈ 0 (within noise) on English benchmarks; no
  output-format degradation.

## Results template (fill after runs)

| Dimension | Benchmark | Metric | Base | CPT | Δ |
|---|---|---|---|---|---|
| 1 | hi-proverbs held-out | BPB | | | |
| 1 | WikiText-103 | PPL | | | |
| 2 | MMLU (0-shot) | acc | | | |
| 2 | BoolQ | acc | | | |
| 2 | GSM8K (8-shot CoT) | EM | | | |
| 2 | HumanEval | pass@1 | | | |
| 3 | MABL | acc_norm | | | |
| 3 | MILU | acc | | | |
| 3 | Global PIQA | acc_norm | | | |
| 3 | IdiomCE | idiom_score | | | |

## Status

All three dimensions are implemented in `src/culture/evaluation/`:

- **Dim 1** — `perplexity.py` (BPB/PPL over a corpus) + `run_perplexity.sh`.
- **Dim 2** — `run_lm_eval.sh` (MMLU/BoolQ/GSM8K-CoT/HumanEval + WikiText via
  `lm-evaluation-harness`).
- **Dim 3** — task module (`tasks.py`/`scorer.py`/`judge.py`) + `run_eval.sh` +
  `compare_results.py`.

Remaining prerequisites (not code): download the Dim 3 data
(`docs/plans/eval_benchmarks_download.md`), and **reserve a held-out `hi-proverbs` slice
before/outside training** for a clean Dim 1 adaptation number.
