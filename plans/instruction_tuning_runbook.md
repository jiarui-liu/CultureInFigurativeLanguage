# Instruction-Tuning Runbook — English anchor + Arabic / Hindi

**Owner:** automated run. **Drafted:** 2026-08-25. **Status:** ⬜ not started · 🔄 in progress · ✅ done · ⚠️ done with caveats · ❌ dropped

**One-line summary:** load each language's continued-pretrained Qwen3.5-9B checkpoint and SFT it
on a **combined, globally shuffled mixture of ONE target-language instruction dataset + ONE
English instruction dataset**, so the model gains in-language instruction following without
losing the general capability the base model came with.

**Scope:** English (the shared anchor half), **Arabic**, **Hindi**, **Chinese**.

**Companion docs:** `docs/literature_reviews/instruction_tuning_datasets_{english,arabic,hindi,chinese}.md` ·
`plans/arabic_pipeline_plan.md` · `docs/plans/paper_strengthening_plan.md`

---

## 0. Design decisions (and why)

**D0.1 — Exactly ONE dataset per language.** Each run mixes two datasets and no more: the target
language's best dataset and the English anchor. Multi-source mixtures add per-source quota,
dedup and provenance confounds that we would then have to ablate; with one source per side the
only knobs are the ratio and the size.

**D0.2 — Mix into ONE shuffled SFT run, not two sequential stages.** Sequential (English → target)
makes the second stage overwrite the first. A single globally shuffled mixture is what Airavata,
Mantra-14B and Nemotron-Mini-Hindi do, and it is the only variant where "English retention" and
"target-language gain" are measured on the same checkpoint.

**D0.3 — Ratio target : English = 60 : 40.** Mantra-14B (Qwen-2.5-14B, the closest public analogue
to our base) reports best results with the target share **above 50%** on a Qwen backbone — and
below 50% on Phi-4, so this is backbone-specific, not universal. §8 ablates it.

**D0.4 — Size ≈ 300K examples per run** (180K target + 120K English), 2 epochs. At ~700
tokens/example that is ~210M tokens ≈ 400 optimizer steps at the batch size in §5 — hours, not
days, so the ratio ablation is affordable.

**D0.5 — SFT the *augmented* CPT checkpoint first.** The paper's four variants per language
(base / cpt / unfiltered / untagged) would mean 4× the runs. Run `cpt` first, then decide.

**D0.6 — Every dataset claim below is second-hand until verified on this cluster.** Repo IDs,
config/column names and per-row quality fields come from the review agents' HF sampling. §2.1 is a
mandatory cheap verification pass before any bulk download.

---

## 1. Paths and environment (this cluster)

| Item | Value |
|---|---|
| Base model | `/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B` |
| CPT ckpt (hi) | `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt/checkpoint-2100` |
| CPT ckpt (ar) | `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-ar-cpt/checkpoint-1608` |
| CPT ckpt (zh) | `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-zh-cpt/checkpoint-11157` |
| SFT data root | `/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data` (new) |
| SFT ckpt root | `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-{hi,ar,zh}-cpt-sft` |
| Code dir | `src/culture/training/instruction_tuning/` (new; sibling of `continued_pretraining/`) |
| venv | `/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv` (working `llamafactory-cli` + transformers 5.6 + deepspeed) |
| HF cache / token | `export HF_HOME=/lustre-storage/fsx_0/user/jiaruiliu/hfcache_sft` · `export HF_TOKEN=...` — **the stored token is expired** |

> Checkpoint step numbers are the augmented runs' `global_step` from
> `docs/plans/paper_strengthening_plan.md` (hi 2100 / ar 1608 / zh 11157). Confirm with `ls <ckpt_root>`
> before launching — take the newest `checkpoint-*` if they differ.

---

## 2. The three datasets

| Role | Dataset | HF repo | Rows available | License |
|---|---|---|---|---|
| **English anchor** | **SmolTalk2** | `HuggingFaceTB/smoltalk2` | English splits, ≫120K | ⚠️ **no license field on the card** |
| **Arabic** | **SmolKalam** | `AdaMLLab/smolkalam-arabic-conversational-sft` | 1,790,478 (24 configs) | Apache-2.0 (source `SultanR/smolkalam` gated, CC-BY-4.0) |
| **Hindi** | **IndicAlign** (`hin_Deva`) | `ai4bharat/indic-align` | 381,173 across Wiki-Conv 141,435 / Wiki-Chat 198,254 / WikiHow 20,313 / Indic-ShareLlama 21,171 | CC-BY-4.0 ✅ |
| **Chinese** | **Infinity-Instruct (Chinese-only mirror)** | `lhoestq/Infinity-Instruct-Chinese-Only` | 751,313 (100% `zh-cn`) | ⚠️ mirror untagged; upstream **CC-BY-SA-4.0** |

**Why SmolTalk2 for English:** its reasoning half is distilled from **Qwen3-32B** — the same family
as our base. GRAPE (arXiv:2502.04194) measures 3–13% from teacher/student distributional fit. Use
the English splits, `no_think` as the bulk plus the reasoning/IF splits, sampled to 120K.

**Why SmolKalam for Arabic:** the only Arabic set that is both large and clean, and it ships
per-row quality scores (`LR`, `SCR`) so the 150K–180K we need can be taken off the top. Everything
else in the Arabic landscape is machine-translated: `2A2I/Arabic-OpenHermes-2.5` translated code
identifiers (`i` → `أنا`) and has `user`/`gpt` swapped at offset 600K; Aya's seven ~4.12M Arabic
"dialect" configs are one English source fanned out by MT with identical row IDs.

**Why IndicAlign for Hindi:** the only large Hindi pool distilled from **open** teachers (Llama-2-70B
+ Mixtral), so it carries no OpenAI-output license taint — the Hindi arm stays releasable, which is
not true of the English one. 100% Devanagari, robust at depth.

**Why Infinity-Instruct-Chinese-Only for Chinese:** 751,313 rows verified **100% `zh-cn`**, ungated,
deep-probed to the final shard, and it **preserves multi-turn** — twice the rows of the
`Mxode/Chinese-Instruct` route with the dialogue structure intact. If the untagged mirror is a
release blocker, the license-clean single-dataset swap is `Mxode/Chinese-Instruct` (4,845,389 rows,
CC-BY-SA-4.0); take its `stem_zh` + `firefly` + `neo_sft_phase2` subsets.

**License note.** SmolTalk2 having no license field is the one real risk to a weight release. It
does not block experiments. If release is blocked, the single-dataset swap is
`allenai/tulu-3-sft-mixture` (ODC-BY) **filtered to English-only rows** — it is ~22% non-English by
construction (aya_100k + wildchat_100k + OASST1; the review verified a Hausa row at offset 900,000),
so unfiltered it would silently contaminate the English anchor.

**Per-dataset traps to encode in §3.2:**
- IndicAlign: exclude the `IndoWordNet` config (96.8M rows, ~100 paraphrases of one fact). The
  `Anudesh` subset turns **Marathi** at depth — run per-row language ID, do not trust config names.
- SmolKalam: gate on `LR ≥ 0.85`, `SCR ≥ 0.95` (verify field names in §2.1).
- Chinese: `Magpie-Qwen2-Air-3M-v0.1` (2,133,622 zh rows, the largest pool found) is the
  **unfiltered** Magpie pool — sampled rows carry rewards of −6.3 and −9.1 and interleave zh/en.
  Not a substitute. Also: Aya's `achinese` config (8.2M rows) is **Acehnese, not Chinese**.
- SmolTalk2: cap response length — the review found 19K–22K-token reasoning rows in sibling
  datasets that silently truncate at our `cutoff_len`.

### 2.1 Verify first (mandatory, ~10 min, no bulk transfer)

```bash
export HF_HOME=/lustre-storage/fsx_0/user/jiaruiliu/hfcache_sft
export HF_TOKEN=...

python3 - <<'PY'
from datasets import get_dataset_config_names
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
for r in ["HuggingFaceTB/smoltalk2",
          "AdaMLLab/smolkalam-arabic-conversational-sft",
          "ai4bharat/indic-align",
          "lhoestq/Infinity-Instruct-Chinese-Only"]:
    try:
        info = api.dataset_info(r)
        cfgs = get_dataset_config_names(r, token=os.environ["HF_TOKEN"])
        print(f"OK   {r:50s} gated={info.gated} n_cfg={len(cfgs)} {cfgs[:8]}")
    except Exception as e:
        print(f"FAIL {r:50s} {type(e).__name__}: {e}")
PY
```

Then pull **5 rows** of each chosen split and confirm: the chat-turn schema, SmolKalam's `LR`/`SCR`
columns, IndicAlign's `hin_Deva` column, SmolTalk2's English split names. Record any deviation from
§2 in this file before proceeding.

---

## 3. Download and build

### 3.1 Download

`hf download` is idempotent — re-run to verify completeness (it can leave orphan `.incomplete`
files; this bit us during CPT).

```bash
SFT_ROOT=/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data
mkdir -p "$SFT_ROOT"

hf download HuggingFaceTB/smoltalk2 --repo-type dataset --local-dir "$SFT_ROOT/smoltalk2"
hf download AdaMLLab/smolkalam-arabic-conversational-sft --repo-type dataset --local-dir "$SFT_ROOT/smolkalam"
hf download ai4bharat/indic-align --repo-type dataset \
  --include "*hin_Deva*" --local-dir "$SFT_ROOT/indic-align"   # NB: --include also keeps IndoWordNet out
hf download lhoestq/Infinity-Instruct-Chinese-Only --repo-type dataset --local-dir "$SFT_ROOT/infinity-instruct-zh"
```

If the SmolKalam mirror disappears, request access to the gated source `SultanR/smolkalam`.

### 3.2 Build — `build_sft_mixture.py`

New script: `src/culture/training/instruction_tuning/build_sft_mixture.py`. Sibling in spirit to
`continued_pretraining/prepare_data.py` (same reasons: LLaMA-Factory's loader wants plain local
JSONL, and we want one deterministic shuffled artifact per run).

**Output** — LLaMA-Factory `sharegpt`, one JSON object per line:

```json
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}],
 "lang":"hi","source":"indic-align/wiki_chat"}
```

`lang`/`source` are provenance only (ignored by training); they let us report the realized ratio
and re-derive per-subset analyses without rebuilding.

**Pipeline, in order:**

1. **Normalize** both sides to the sharegpt schema (IndicAlign `hin_Deva` column; drop `IndoWordNet`).
2. **Structural validity** — drop empty turns, non-alternating human/gpt sequences, a `gpt` turn
   first, or role values that are not literally `human`/`gpt`.
3. **Language ID + script ratio** per row: Hindi assistant turns ≥ 70% Devanagari, Arabic ≥ 70%
   Arabic script, Chinese ≥ 70% Han, English rows must ID as `en`. This keeps Marathi out of the Hindi half and
   enforces the English-anchor purity noted in §2. Log drop rate per source.
4. **Degeneration filter** — reject assistant turns with a character 3-gram repeated past a
   threshold. Airavata's `chrF++ ≥ 50` gate is far too lenient: the review found a fully degenerate
   row (`-तो-तो-तो-…`) scoring **94.33**.
5. **Quality gates** — SmolKalam `LR ≥ 0.85` / `SCR ≥ 0.95`; response length ≤ `cutoff_len`.
6. **Near-dedup** (MinHash over prompts) and **decontaminate** against every benchmark prompt used
   in `src/culture/evaluation/`. Log removals — a hit here would invalidate the paper's numbers.
7. **Sample to quota** (180K target / 120K English), **concatenate, global shuffle `seed=42`**,
   write `train_sft_{ar,hi,zh}/part-*.jsonl` + `manifest.json` recording pre/post-filter counts,
   realized ratio, token estimate, and each HF repo's commit SHA.

```bash
python3 build_sft_mixture.py --lang hi --target_ratio 0.6 --total 300000 \
  --out_dir /lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_hi
python3 build_sft_mixture.py --lang ar --target_ratio 0.6 --total 300000 \
  --out_dir /lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_ar
python3 build_sft_mixture.py --lang zh --target_ratio 0.6 --total 300000 \
  --out_dir /lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_zh
python3 build_sft_mixture.py --verify_only --out_dir .../train_sft_hi
```

**Gate:** read `manifest.json` and 20 random rows per language before launching training.

### 3.3 Register

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
},
"zh_sft_mix": {
  "file_name": "/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/train_sft_zh",
  "formatting": "sharegpt",
  "columns": { "messages": "conversations" },
  "tags": { "role_tag": "from", "content_tag": "value",
            "user_tag": "human", "assistant_tag": "gpt" }
}
```

---

## 4. Training configs

New: `src/culture/training/instruction_tuning/configs/qwen3p5_9b_sft_{hi,ar,zh}.yaml`. Deltas from the
CPT configs — everything else stays identical so runs remain comparable.

| Field | CPT value | SFT value | Why |
|---|---|---|---|
| `stage` | `pt` | **`sft`** | supervised instruction tuning |
| `model_name_or_path` | base Qwen3.5-9B | **the CPT checkpoint** (§1) | this is the point of the run |
| `template` | `default` | **`qwen`** | base model has no chat template; verify the exact name in this LLaMA-Factory build |
| `dataset` | `hi_proverbs` / `ar_amthal` / `zh_chengyu` | `hi_sft_mix` / `ar_sft_mix` / `zh_sft_mix` | §3.3 |
| `cutoff_len` | 16384 | **8192** | SFT rows are short; halves activation cost |
| `neat_packing` | `false` | **`true`** | SFT-only in LLaMA-Factory; prevents cross-example attention contamination |
| `num_train_epochs` | 3 | **2** | standard for a ~300K-example SFT |
| loss masking | n/a | **assistant turns only** (LF default) | confirm in the first logged batch |
| `output_dir` | `…-cpt` | `…-cpt-sft` | |

Unchanged: `learning_rate: 1.0e-5` cosine w/ `warmup_ratio: 0.03`, `flash_attn: sdpa` (**not** `fa2`
— transformers 5.6's FA2 path crashes on Qwen3.5's optional attention-sink `s_aux`),
`torch_compile: false`, `bf16: true`, ZeRO-3 via `ds_z3.json`, `packing: true`, `seed: 42`,
`overwrite_output_dir: false` for auto-resume on requeue.

**Batch:** `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 4`, 4 nodes × 8 GPUs →
global 128 seq × 8192 ≈ **1.05M tokens/step**. ~210M tokens × 2 epochs ≈ **400 steps**.
`save_steps: 100`, `save_total_limit: 5`.

---

## 5. Launch

`sft_hi.slurm` / `sft_ar.slurm` / `sft_zh.slurm` — copy `continued_pretraining/cpt_ar.slurm` and change `CONFIG`,
`--job-name`, and the `mkdir -p` output dir. It already does the right things: `--chdir` pins CWD,
`source env.sh` sets NCCL/IB, `MASTER_ADDR` from `scontrol`, `HF_DATASETS_OFFLINE=1`, per-job `/tmp`
caches for HF/Triton, `WANDB_MODE=offline`, `FORCE_TORCHRUN=1` + `NNODES`/`NODE_RANK` per task.

```bash
cd src/culture/training/instruction_tuning
sbatch sft_hi.slurm     # Hindi:  CPT-hi ckpt + hi_sft_mix
sbatch sft_ar.slurm     # Arabic: CPT-ar ckpt + ar_sft_mix   (parallel, separate job)
sbatch sft_zh.slurm     # Chinese: CPT-zh ckpt + zh_sft_mix  (parallel, separate job)
```

**Watch the first 20 steps:** loss should start well below a from-scratch SFT (the base is already
converged), the first logged batch should show loss masked to assistant turns, throughput ~1.05M
tok/step. Flat-0 or NaN loss is almost always the template or the role tags in §3.3.

---

## 6. Evaluation

1. **Target-language instruction following** — the language's idiom/cultural tasks
   (`kinayat_meaning`, `kinayat_cloze`, `ar_figurative`; Hindi MILU; Chinese ChID / Chengyu-Bench) plus a generative
   instruction-following check. The CPT → CPT+SFT delta is the headline.
2. **English retention (Dim-1)** — MMLU / BoolQ / GSM8K / HumanEval / WikiText on the SFT'd
   checkpoint vs. the pre-SFT CPT checkpoint. **This is the number that tests D0.2**: if the
   English half is doing its job, retention is flat or up.

Write per-item `records` to
`/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/{hi,ar,zh}/cpt-sft/<task>.json`,
matching the existing layout so `src/culture/evaluation/compute_cis.py` picks it up unchanged.

---

## 7. Ablations (only if the headline result holds)

| # | Ablation | Cost | Question |
|---|---|---|---|
| A1 | ratio 40/60 and 80/20 vs 60/40 | 2 runs/lang | is D0.3's Qwen-specific >50% claim real for us? |
| A2 | target-only SFT (no English half) | 1/lang | quantifies exactly what the English anchor buys |
| A3 | SFT the `unfiltered` CPT checkpoint, same mixture | 1/lang | does SFT wash out the CPT curation effect? **most paper-relevant** |
| A4 | DPO on top | — | see below |

**A4 is worth flagging now.** NVIDIA's Nemotron-Mini-Hindi ablation, run on a Hindi-CPT'd base —
exactly our setting — found Hindi **DPO** buys the same gain as translated Hindi SFT (4.30 vs 4.28,
baseline 3.81) and that **the two do not stack**. If §6 shows a small SFT delta, that is the
expected shape, not a bug, and the preference stage is the higher-value next lever.

---

## 8. Chinese-specific notes

- **The quota binds, not the pool.** 751K zh rows for a 180K quota means the 60/40 ratio is a free
  choice here, unlike Hindi (381K available) or Arabic-after-filtering. Worth re-checking the ratio
  for zh specifically in the §7 A1 ablation.
- **The zh CPT run is the long one** (`global_step` 11157 vs hi 2100 / ar 1608), but SFT cost is set
  by the mixture, not the CPT length — expect the same ~400 steps as hi/ar.
- **Sampling bias warning, carried from the review.** Chinese-share estimates from partial shard
  scans are unreliable because shards are often **time-ordered**: WildChat's zh share falls
  monotonically from 25.53% (shard 0) to 2.65% (shard 11), which produced a 63% overestimate until a
  full 14-shard scan corrected it. If §3.2 ever needs a language share from a multi-shard dataset,
  scan **all** shards.

---

## 9. Checklist

- [ ] §2.1 verification pass — all three repos resolve, field names confirmed, deviations recorded here
- [ ] `HF_TOKEN` refreshed (the stored one is expired)
- [ ] Download complete + re-run `hf download` to confirm no `.incomplete` orphans
- [ ] `build_sft_mixture.py` written and run for hi + ar + zh
- [ ] `manifest.json` reviewed: realized ratio ≈ 60/40, drop rates sane, decontamination log empty-or-explained
- [ ] 20 random rows per language read by a human
- [ ] `dataset_info.json` entries added; `template: qwen` verified against this LLaMA-Factory build
- [ ] CPT checkpoint paths confirmed on disk (newest `checkpoint-*`)
- [ ] SFT configs + slurm launchers written
- [ ] First-20-steps sanity check passed (loss masking, throughput)
- [ ] All three runs complete; eval records written to the standard layout
- [ ] English retention compared pre-/post-SFT (the D0.2 test)
