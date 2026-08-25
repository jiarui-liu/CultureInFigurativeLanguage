# Instruction-Tuning Runbook — English anchor + Arabic / Hindi

**Owner:** automated run. **Drafted:** 2026-08-25. **Status:** ⬜ not started · 🔄 in progress · ✅ done · ⚠️ done with caveats · ❌ dropped

**One-line summary:** load each language's continued-pretrained Qwen3.5-9B checkpoint and
SFT it on a **combined, globally shuffled mixture of that language's instruction data +
English instruction data**, so the model gains in-language instruction following without
losing the English/general capability the base model came with.

**Scope:** English (as the shared anchor half), **Arabic**, **Hindi**. Chinese is deliberately
**out of scope for this draft** — the zh dataset review (`docs/literature_reviews/instruction_tuning_datasets_chinese.md`)
was still running when this was written. Section 10 says exactly what to add when it lands.

**Companion docs:**
- `docs/literature_reviews/instruction_tuning_datasets_english.md`
- `docs/literature_reviews/instruction_tuning_datasets_arabic.md`
- `docs/literature_reviews/instruction_tuning_datasets_hindi.md`
- `plans/arabic_pipeline_plan.md`, `docs/plans/paper_strengthening_plan.md`

---

## 0. Design decisions (and why)

**D0.1 — Mix target-language + English in ONE shuffled SFT run, not two sequential stages.**
Sequential (English SFT → target SFT, or the reverse) makes the second stage overwrite the
first. A single globally shuffled mixture is what Airavata, Mantra-14B and Nemotron-Mini-Hindi
all do, and it is the only variant where "English retention" and "target-language gain" are
measured on the same checkpoint.

**D0.2 — Mixture ratio: target : English = 60 : 40.** Mantra-14B (Qwen-2.5-14B, the closest
public analogue to our base) reports best results with the target-language share **above 50%**
on a Qwen backbone — and below 50% on Phi-4, so this is backbone-specific, not universal.
60/40 is the starting point; §9 lists the ablation.

**D0.3 — Target size ≈ 300K examples per language run** (180K target + 120K English), 2 epochs.
At ~700 tokens/example this is ~210M tokens ≈ 400 optimizer steps at the batch size in §6 —
hours, not days, so the ratio ablation in §9 is affordable.

**D0.4 — SFT the *augmented* CPT checkpoint first; extend to the controls only if the
headline result holds.** The paper's four variants per language (base / cpt / unfiltered /
untagged) would mean 4× the SFT runs. Run `cpt` first, then decide.

**D0.5 — Two English pools: capability-first (A) and license-clean (B).** Use **A** for
experiments. Switch to **B** before any weight release — several A datasets have no license
field at all or are CC-BY-NC / CC-BY-SA. See §2.

**D0.6 — Do NOT use `allenai/tulu-3-sft-mixture` as the English half.** It is ~22% non-English
by construction (aya_100k + wildchat_100k + OASST1); the review verified a Hausa row at offset
900,000. Using it would silently contaminate the "English anchor" arm of the experiment.

**D0.7 — Every dataset claim in §2 is second-hand until verified on this cluster.** Repo IDs,
config/column names and per-row quality fields come from the review agents' sampling. §3.0 is a
mandatory cheap verification pass before any bulk download.

---

## 1. Paths and environment (this cluster)

| Item | Value |
|---|---|
| Base model | `/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B` |
| CPT ckpt (hi) | `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt/checkpoint-2100` |
| CPT ckpt (ar) | `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-ar-cpt/checkpoint-1608` |
| SFT data root | `/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data` (new) |
| SFT ckpt root | `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-{hi,ar}-cpt-sft` |
| Code dir | `src/culture/training/instruction_tuning/` (new; sibling of `continued_pretraining/`) |
| venv | `/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv` (working `llamafactory-cli` + transformers 5.6 + deepspeed) |
| HF cache | `export HF_HOME=/lustre-storage/fsx_0/user/jiaruiliu/hfcache_sft` |
| HF token | `export HF_TOKEN=...` — **required**; the English review found the stored token expired |

> Checkpoint step numbers are the augmented runs' `global_step` from
> `docs/plans/paper_strengthening_plan.md` (hi 2100 / ar 1608). Confirm with
> `ls <ckpt_root>` before launching — take the newest `checkpoint-*` if they differ.

---

## 2. Selected datasets

### 2.1 English anchor — Pool A (capability-first, use for experiments)

| Dataset | HF repo | Rows | License |
|---|---|---|---|
| SmolTalk2 — English `no_think` splits | `HuggingFaceTB/smoltalk2` | 120,000 | ⚠️ no license field on card |
| SmolTalk2 — `multi_turn_reasoning_if_think` + `systemchats_Qwen3_32B_think` | same | 55,653 | ⚠️ same |
| Tulu-3 personas instruction-following | `allenai/tulu-3-sft-personas-instruction-following` | 29,980 | ODC-BY ✅ |
| OpenThoughts3, responses ≤ 8K tokens, domain-balanced | `open-thoughts/OpenThoughts3-1.2M` | 40,000 | Apache-2.0 ✅ |
| OpenMathInstruct-2, `augmented_*` sources only | `nvidia/OpenMathInstruct-2` | 25,000 | CC-BY-4.0 ✅ |
| StarCoder2 self-OSS-Instruct | `bigcode/self-oss-instruct-sc2-exec-filter-50k` | 20,000 | ODC-BY ✅ |
| No Robots | `HuggingFaceH4/no_robots` | 9,500 | ⚠️ CC-BY-NC-4.0 |
| OASST2, `lang == "en"` | `OpenAssistant/oasst2` | ~4,000 | Apache-2.0 ✅ |
| Dolly-15K | `databricks/databricks-dolly-15k` | ~2,000 | ⚠️ CC-BY-SA-3.0 (viral) |

**Why SmolTalk2 carries the pool:** its reasoning half is distilled from **Qwen3-32B**, i.e. the
same family as our base. GRAPE (arXiv:2502.04194) measures 3–13% from teacher/student
distributional fit. Downsample this pool to the 120K the mixture needs (§4).

### 2.2 English anchor — Pool B (license-clean, use before any release)

OpenThoughts3 60K (Apache-2.0) + OpenMathInstruct-2 30K (CC-BY-4.0) + StarCoder2-Instruct 30K
(ODC-BY) + Tulu-3-personas-IF 29,980 (ODC-BY) + OASST2-en ~4K (Apache-2.0) + Nemotron-v1 chat
~10K. Drops every unlicensed / NC / SA row.

### 2.3 Arabic

| Dataset | HF repo | Rows | License | Role |
|---|---|---|---|---|
| **SmolKalam** | `AdaMLLab/smolkalam-arabic-conversational-sft` | 1,790,478 (24 configs) | Apache-2.0 (source `SultanR/smolkalam` is gated, CC-BY-4.0) | backbone — filter to 150K |
| Quora-Arabic-GPT4 | `FreedomIntelligence/Quora-Arabic-GPT4` | 43,050 | Apache-2.0 ✅ | native-Arabic *prompts* — 20K |
| CIDAR | `arbml/CIDAR` | 10,000 | ⚠️ card Apache-2.0 vs paper CC-BY-NC | human-reviewed, culturally localized — 10K |

**Filter SmolKalam on its per-row quality fields** (`LR ≥ 0.85`, `SCR ≥ 0.95` — verify names in
§3.0). Quora-Arabic-GPT4 matters because it is the only sizeable Arabic set whose *prompts* were
natively written rather than machine-translated.

**Explicitly excluded, with reasons found by sampling:**
- `2A2I/Arabic-OpenHermes-2.5` — **translated code identifiers** (`i` → `أنا`, `package main` →
  `الحزمة الرئيسية`) and `user`/`gpt` fields swapped at offset 600K.
- Aya Collection Arabic "dialect" configs — a uniform MT fan-out of one English source (identical
  row IDs across the seven ~4.12M configs), `<unk>` leaking into text. Only `aya_dataset`'s
  **13,960** human-written Arabic rows are usable; optional top-up.
- `ClusterlabAi/InstAr-500k` — FLAN dump behind one repeated *English* system prompt, English
  answers appearing in Arabic rows.
- `2A2I/argilla-dpo-mix-7k-arabic` — translated the JSON **role values** (`"role": "مستخدم"`),
  which breaks chat templating.
- `Hala-4.6M-SFT` (4,060,575) — good STEM depth but **CC-BY-NC-4.0**; hold as an optional
  experiment-only top-up, never in a release mixture.

### 2.4 Hindi

| Dataset | HF repo / config | Rows | License | Role |
|---|---|---|---|---|
| **IndicAlign** `hin_Deva` — Wiki-Conv / Wiki-Chat / WikiHow / Indic-ShareLlama | `ai4bharat/indic-align` | 141,435 / 198,254 / 20,313 / 21,171 | CC-BY-4.0 ✅ | backbone — 120K |
| Samvaad-hi-v1 | `sarvamai/samvaad-hi-v1` | 101,476 | Apache-2.0 ✅ | India-grounded chat — 30K (Hindi-output rows only) |
| indic-instruct `wikihow/hi` | `ai4bharat/indic-instruct-data-v0.1` | 6,055 | CC-0 ✅ | native long-form — all |
| indic-instruct `anudesh/hi` | same | 7,577 | CC-BY-4.0 ✅ | native prompts — all |
| Orca-Math Hindi (filtered) | `BhabhaAI/orca-math-word-problems-200k-hindi-filtered` | 188,943 | MIT ✅ | math — 20K |

**Why IndicAlign carries the pool:** it is the only large Hindi set distilled from **open**
teachers (Llama-2-70B + Mixtral), so it carries no OpenAI-output license taint — the whole Hindi
pool can stay releasable, which is not true for English or Arabic.

**Traps to encode in the build script:**
- `IndoWordNet` config (96.8M rows) is ~100 paraphrases of one fact — **exclude**.
- `indic-align/Anudesh` turns **Marathi** at depth; `zicsx/indic-align-hindi` is **Bodo** at
  offset 200K — run per-row language ID, do not trust the config name.
- `indic-instruct wikihow/hi` ships **URL-percent-escape-corrupted** prompts in `messages` —
  rebuild prompts from the raw column.
- Verified subset duplication: `shreyas18/Hindi_instruct_1_5M_v1` ⊃ `atharvanighot/Hindi-Instruct-500K`;
  `guneetsk99/hindi_instruction_set_187K` ⊃ Bactrian-X hi. None are in the mixture; keep them out.
- **No public Hindi code-SFT dataset exists** — Hindi code ability rides entirely on the English half.

---

## 3. Download

### 3.0 Verify first (mandatory, ~10 min, no bulk transfer)

Before downloading hundreds of GB, confirm each repo ID resolves, and that the config /
column / quality-field names in §2 are real:

```bash
export HF_HOME=/lustre-storage/fsx_0/user/jiaruiliu/hfcache_sft
export HF_TOKEN=...   # the previously stored token is expired

python3 - <<'PY'
from datasets import get_dataset_config_names
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
REPOS = [
 "HuggingFaceTB/smoltalk2","allenai/tulu-3-sft-personas-instruction-following",
 "open-thoughts/OpenThoughts3-1.2M","nvidia/OpenMathInstruct-2",
 "bigcode/self-oss-instruct-sc2-exec-filter-50k","HuggingFaceH4/no_robots",
 "OpenAssistant/oasst2","databricks/databricks-dolly-15k",
 "AdaMLLab/smolkalam-arabic-conversational-sft","FreedomIntelligence/Quora-Arabic-GPT4",
 "arbml/CIDAR","ai4bharat/indic-align","sarvamai/samvaad-hi-v1",
 "ai4bharat/indic-instruct-data-v0.1","BhabhaAI/orca-math-word-problems-200k-hindi-filtered",
]
for r in REPOS:
    try:
        info = api.dataset_info(r)
        cfgs = get_dataset_config_names(r, token=os.environ["HF_TOKEN"])
        print(f"OK   {r:60s} gated={info.gated} configs={cfgs[:6]}{'...' if len(cfgs)>6 else ''}")
    except Exception as e:
        print(f"FAIL {r:60s} {type(e).__name__}: {e}")
PY
```

Then pull **5 rows** of each chosen split and eyeball: the field names, the chat-turn schema,
SmolKalam's `LR`/`SCR` columns, IndicAlign's `hin_Deva` column. Record any name that differs
from §2 in this file before proceeding.

### 3.1 Bulk download

`hf download` is idempotent — re-run to verify completeness (it can leave orphan `.incomplete`
files; this bit us during CPT).

```bash
SFT_ROOT=/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data
mkdir -p "$SFT_ROOT"

# --- English (Pool A) ---
for R in HuggingFaceTB/smoltalk2 \
         allenai/tulu-3-sft-personas-instruction-following \
         nvidia/OpenMathInstruct-2 \
         bigcode/self-oss-instruct-sc2-exec-filter-50k \
         HuggingFaceH4/no_robots OpenAssistant/oasst2 databricks/databricks-dolly-15k; do
  hf download "$R" --repo-type dataset --local-dir "$SFT_ROOT/$(basename $R)"
done
# OpenThoughts3 is 1.2M rows / large — pull only what we sample:
hf download open-thoughts/OpenThoughts3-1.2M --repo-type dataset \
  --include "data/train-0000[0-9]-*.parquet" --local-dir "$SFT_ROOT/OpenThoughts3-1.2M"

# --- Arabic ---
hf download AdaMLLab/smolkalam-arabic-conversational-sft --repo-type dataset \
  --local-dir "$SFT_ROOT/smolkalam"
hf download FreedomIntelligence/Quora-Arabic-GPT4 --repo-type dataset --local-dir "$SFT_ROOT/quora-ar-gpt4"
hf download arbml/CIDAR --repo-type dataset --local-dir "$SFT_ROOT/cidar"

# --- Hindi ---
hf download ai4bharat/indic-align --repo-type dataset \
  --include "*hin_Deva*" --local-dir "$SFT_ROOT/indic-align"      # NB: excludes IndoWordNet by config
hf download sarvamai/samvaad-hi-v1 --repo-type dataset --local-dir "$SFT_ROOT/samvaad-hi-v1"
hf download ai4bharat/indic-instruct-data-v0.1 --repo-type dataset --local-dir "$SFT_ROOT/indic-instruct"
hf download BhabhaAI/orca-math-word-problems-200k-hindi-filtered --repo-type dataset \
  --local-dir "$SFT_ROOT/orca-math-hi"
```

If SmolKalam's `AdaMLLab` mirror disappears, request access to the gated source
`SultanR/smolkalam` and re-point.

---

## 4. Build the mixtures — `build_sft_mixture.py`

New script: `src/culture/training/instruction_tuning/build_sft_mixture.py`.
Sibling in spirit to `continued_pretraining/prepare_data.py` (same reasons: LLaMA-Factory's
loader wants plain local JSONL, and we want one deterministic shuffled artifact per run).

**Output format** — LLaMA-Factory `sharegpt`, one JSON object per line:

```json
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}],
 "lang":"hi","source":"indic-align/wiki_chat"}
```

`lang` and `source` are provenance only (ignored by training); they let us report the realized
mixture and re-derive per-source ablations without rebuilding.

**Pipeline, in order:**

1. **Load + normalize** each source to the sharegpt schema. Handle per-source quirks: rebuild
   `wikihow/hi` prompts from the raw column (percent-escape corruption); read IndicAlign's
   `hin_Deva` column; drop `IndoWordNet`.
2. **Structural validity.** Drop rows with empty turns, a non-alternating human/gpt sequence,
   a `gpt` turn first, or role values that are not literally `human`/`gpt` (this is what catches
   the Arabic role-translation bug if a future source has it).
3. **Language ID per row** (`fasttext` lid.176 or `langdetect`) + **script-ratio check**:
   Hindi rows must be ≥ 70% Devanagari in the *assistant* turn; Arabic rows ≥ 70% Arabic script;
   English rows must ID as `en`. This is what keeps Marathi/Bodo out of the Hindi half and
   enforces D0.6 on the English half. Log the drop rate per source.
4. **Degeneration filter.** Reject rows whose assistant turn has a character 3-gram repeated
   beyond a threshold. Airavata's `chrF++ ≥ 50` translation gate is far too lenient — the review
   found a fully degenerate row (`-तो-तो-तो-…`) scoring **94.33**. If a source ships a translation
   score, gate at **≥ 90**, not 50, and still run this filter.
5. **Quality gates where the source provides them.** SmolKalam: `LR ≥ 0.85`, `SCR ≥ 0.95`.
   OpenThoughts3: response ≤ 8K tokens (the review found 19K–22K-token reasoning rows elsewhere
   that would silently truncate at our `cutoff_len`). OpenMathInstruct-2: `augmented_*` sources only.
6. **Length filter.** Drop rows whose rendered length exceeds `cutoff_len` (§6) rather than
   training on a truncated answer. Report the count.
7. **Near-dedup** within and across sources (MinHash over the prompt), then **decontaminate**
   against the eval sets used in `src/culture/evaluation/` — n-gram overlap against every
   benchmark prompt. Log what was removed; a benchmark hit here would invalidate the paper's numbers.
8. **Sample to quota** per source (§2 "Role" column), **concatenate, global shuffle with
   `seed=42`**, write `train_sft_{ar,hi}/part-*.jsonl` plus a `manifest.json` recording per-source
   pre/post-filter counts, realized ratio, token estimate, and the exact commit SHA of each HF repo.

**Run:**

```bash
python3 build_sft_mixture.py --lang hi --english_pool A --target_ratio 0.6 --total 300000 \
  --out_dir /lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_hi
python3 build_sft_mixture.py --lang ar --english_pool A --target_ratio 0.6 --total 300000 \
  --out_dir /lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_ar
python3 build_sft_mixture.py --verify_only --out_dir .../train_sft_hi   # counts + ratio + sample rows
```

**Gate:** read `manifest.json` and 20 random rows per language before launching training.

---

## 5. Register the datasets

Append to `src/culture/training/continued_pretraining/configs/dataset_info.json` (the SFT configs
reuse the same `dataset_dir`, so there is one registry):

```json
"hi_sft_mix": {
  "file_name": "/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_hi",
  "formatting": "sharegpt",
  "columns": { "messages": "conversations" },
  "tags": { "role_tag": "from", "content_tag": "value",
            "user_tag": "human", "assistant_tag": "gpt" }
},
"ar_sft_mix": {
  "file_name": "/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_ar",
  "formatting": "sharegpt",
  "columns": { "messages": "conversations" },
  "tags": { "role_tag": "from", "content_tag": "value",
            "user_tag": "human", "assistant_tag": "gpt" }
}
```

---

## 6. Training configs

New: `src/culture/training/instruction_tuning/configs/qwen3p5_9b_sft_{hi,ar}.yaml`.
Deltas from the CPT configs — everything else stays identical so runs remain comparable.

| Field | CPT value | SFT value | Why |
|---|---|---|---|
| `stage` | `pt` | **`sft`** | supervised instruction tuning |
| `model_name_or_path` | base Qwen3.5-9B | **the CPT checkpoint** (§1) | this is the point of the run |
| `template` | `default` | **`qwen`** | base model has no chat template; verify the exact name in this LLaMA-Factory build before launching |
| `dataset` | `hi_proverbs` / `ar_amthal` | `hi_sft_mix` / `ar_sft_mix` | §5 |
| `cutoff_len` | 16384 | **8192** | SFT rows are short; halves activation cost. §4 step 6 drops what won't fit |
| `neat_packing` | `false` | **`true`** | supported for SFT only; prevents cross-example attention contamination |
| `num_train_epochs` | 3 | **2** | standard for a ~300K-example SFT |
| `learning_rate` | 1.0e-5 | **1.0e-5** | unchanged |
| `mask_history` / train-on-prompt | n/a | **loss on assistant turns only** (LLaMA-Factory default) | confirm in the log's first batch |
| `output_dir` | `…-cpt` | `…-cpt-sft` | |

Keep `flash_attn: sdpa` (**not** `fa2` — transformers 5.6's FA2 path crashes on Qwen3.5's
optional attention-sink `s_aux`), `torch_compile: false`, `bf16: true`, ZeRO-3 via `ds_z3.json`,
`packing: true`, `seed: 42`, cosine LR with `warmup_ratio: 0.03`, `overwrite_output_dir: false`
for auto-resume on requeue.

**Batch:** `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 4`, 4 nodes × 8 GPUs
→ global 128 sequences × 8192 ≈ **1.05M tokens/step**. ~210M tokens × 2 epochs ≈ **400 steps**.
`save_steps: 100`, `save_total_limit: 5`.

---

## 7. Launch

`sft_hi.slurm` / `sft_ar.slurm` — copy `continued_pretraining/cpt_ar.slurm` and change
`CONFIG`, `--job-name`, and the `mkdir -p` output dir. It already does the right things:
`--chdir` pins CWD, `source env.sh` sets NCCL/IB, `MASTER_ADDR` from `scontrol`,
`HF_DATASETS_OFFLINE=1`, per-job `/tmp` caches for HF/Triton, `WANDB_MODE=offline`,
`FORCE_TORCHRUN=1` + `NNODES`/`NODE_RANK` per task.

```bash
cd src/culture/training/instruction_tuning
sbatch sft_hi.slurm     # Hindi:  CPT-hi ckpt + hi_sft_mix
sbatch sft_ar.slurm     # Arabic: CPT-ar ckpt + ar_sft_mix   (parallel, separate job)
```

**Watch the first 20 steps:** loss should start well below a from-scratch SFT (the base is
already converged), the first logged batch should show loss masked to assistant turns, and
throughput should be ~1.05M tok/step. If loss is flat at 0 or NaN, stop — it is almost always the
template or the role tags in §5.

---

## 8. Evaluation

Two axes, both already scripted under `src/culture/evaluation/`:

1. **Target-language instruction following** — the language's idiom/cultural tasks
   (`kinayat_meaning`, `kinayat_cloze`, `ar_figurative`; Hindi MILU etc.), plus a generative
   instruction-following check. The CPT-vs-CPT+SFT delta is the headline.
2. **English retention (Dim-1)** — MMLU / BoolQ / GSM8K / HumanEval / WikiText on the SFT'd
   checkpoint, compared against the pre-SFT CPT checkpoint. **This is the number that tests
   D0.1**: if the English half is doing its job, retention should be flat or up.

Write per-item `records` to
`/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/{hi,ar}/cpt-sft/<task>.json`,
matching the existing layout so `src/culture/evaluation/compute_cis.py` picks it up and produces
bootstrap CIs and paired tests without changes.

---

## 9. Ablations (only if the headline result holds)

| # | Ablation | Cost | Question |
|---|---|---|---|
| A1 | ratio 40/60 and 80/20 vs. 60/40 | 2 extra runs/lang | is D0.2's Qwen-specific >50% claim real for us? |
| A2 | target-only SFT (no English half) | 1/lang | quantifies exactly what the English anchor buys |
| A3 | SFT the `unfiltered` CPT checkpoint on the same mixture | 1/lang | does SFT wash out the CPT curation effect? **The most paper-relevant one** |
| A4 | English Pool B vs Pool A | 1/lang | does the license-clean mixture cost capability? |
| A5 | DPO on top | — | see below |

**A5 is worth flagging now.** NVIDIA's Nemotron-Mini-Hindi ablation, run on a Hindi-CPT'd base —
i.e. exactly our setting — found Hindi **DPO** buys the same gain as translated Hindi SFT
(4.30 vs 4.28, baseline 3.81) and that **the two do not stack**. If §8 shows a small SFT delta,
that is the expected shape, not a bug, and the preference stage is the higher-value next lever.
Budget for it rather than iterating on the SFT mixture.

---

## 10. When the Chinese review lands

Add a `zh` arm symmetric to Arabic: pick the backbone dataset from
`docs/literature_reviews/instruction_tuning_datasets_chinese.md`, add a `zh_sft_mix` entry to
§5, a `qwen3p5_9b_sft_zh.yaml` to §6, and `sft_zh.slurm` to §7. The CPT checkpoint is
`…/ckpts/qwen3p5-9b-zh-cpt` (augmented run `global_step` 11157). Everything else — build script,
filters, ratio, eval layout — carries over unchanged. Re-check the 60/40 ratio for zh: Chinese
data is far more plentiful than Hindi or Arabic, so the quota, not the pool, will be the binding
constraint.

---

## 11. Checklist

- [ ] §3.0 verification pass — every repo ID resolves, field names confirmed, deviations recorded here
- [ ] `HF_TOKEN` refreshed (the stored one is expired)
- [ ] Bulk download complete + re-run `hf download` to confirm no `.incomplete` orphans
- [ ] `build_sft_mixture.py` written and run for hi + ar
- [ ] `manifest.json` reviewed: realized ratio ≈ 60/40, drop rates sane, decontamination log empty-or-explained
- [ ] 20 random rows per language read by a human
- [ ] `dataset_info.json` entries added; `template: qwen` verified against this LLaMA-Factory build
- [ ] CPT checkpoint paths confirmed on disk (newest `checkpoint-*`)
- [ ] SFT configs + slurm launchers written
- [ ] First-20-steps sanity check passed (loss masking, throughput)
- [ ] Both runs complete; eval records written to the standard eval layout
- [ ] English retention compared pre-/post-SFT (the D0.1 test)
