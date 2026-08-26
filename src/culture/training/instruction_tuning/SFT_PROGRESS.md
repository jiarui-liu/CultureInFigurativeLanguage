# SFT Progress Tracker — English-anchored instruction tuning (hi / ar × base / cpt)

**Owner:** automated run. **Started:** 2026-08-25. Autonomous — user is notified only when
**all 4 trainings + all 4 evals** are complete.

Runbook: `../../../../plans/instruction_tuning_runbook.md`. This file records what was actually
done on the cluster (decisions, deviations, job IDs, results).

Status legend: ⬜ not started · 🔄 in progress · ✅ done · ⚠️ done w/ caveat · ❌ failed

---

## The 4 runs

| # | Run | Base weights | SFT data | Output dir |
|---|---|---|---|---|
| 1 | hi-cpt-sft  | qwen3p5-9b-hi-cpt/checkpoint-2100 (augmented CPT) | hi_sft_mix | qwen3p5-9b-hi-cpt-sft |
| 2 | ar-cpt-sft  | qwen3p5-9b-ar-cpt/checkpoint-1608 (augmented CPT) | ar_sft_mix | qwen3p5-9b-ar-cpt-sft |
| 3 | hi-base-sft | Qwen3.5-9B (original, pre-CPT)                     | hi_sft_mix | qwen3p5-9b-hi-base-sft |
| 4 | ar-base-sft | Qwen3.5-9B (original, pre-CPT)                     | ar_sft_mix | qwen3p5-9b-ar-base-sft |

"augmented CPT" = the checkpoint continued-pretrained with BOTH idiom filtering AND knowledge-block
metadata tagging (the `ar_amthal`/`hi_proverbs` variant), per user's request for the "right" CPT.

---

## Datasets (one target dataset per language + shared English anchor)

- English anchor: `HuggingFaceTB/smoltalk2` (SFT split, English `*_no_think*` configs; multilingual split excluded)
- Arabic: `AdaMLLab/smolkalam-arabic-conversational-sft` (all configs, gated on LR≥0.85 & SCR≥0.95)
- Hindi: `ai4bharat/indic-align` (wiki_chat + wikihow + indicsharellama; `hin_Deva` column only)

Downloaded to `/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data/{smoltalk2,smolkalam,indic-align}`.
Note: `indic-align/wiki_conv` did not download (34/35 parquets); wiki_chat (198K) is the backbone so
Hindi quota is still met. IndoWordNet/anudesh/dolly/oasst/toxic deliberately excluded.

Mixture: target:English = 60:40, total 300K (180K target + 120K English), **globally shuffled seed=42**,
LLaMA-Factory `sharegpt` JSONL. Output: `.../culture-sft-data/train_sft_{hi,ar}/part-*.jsonl` +
sibling `train_sft_{hi,ar}.manifest.json`.

---

## Key research decisions (verified on this cluster, not from the runbook verbatim)

- **template = `qwen3_5_nothink`** — model-native Qwen3.5 template; our SFT data has no reasoning
  traces so the no-think variant is correct. Verified it encodes ChatML and masks assistant-only.
  (Runbook said generic `qwen`; the build actually has a dedicated `qwen3_5` template.)
- **packing = FALSE, neat_packing = FALSE** — Qwen3.5 is a HYBRID model: text_config shows 32 layers,
  only every 4th is `full_attention`, the other 24 are `linear_attention` (+conv). Sequence packing
  leaks SSM/linear-attention/conv state across packed examples even with neat_packing's block-diagonal
  attention mask (that mask only fixes the 8 full-attention layers). So packing is unsafe for SFT here.
  With per_device_train_batch_size=1 there is zero padding waste anyway. **This overrides runbook §4's
  `neat_packing: true`.**
- **flash_attn = sdpa** (NOT fa2 — transformers 5.6 FA2 crashes on Qwen3.5 `s_aux`; carried from CPT).
- Optimizer identical to CPT for comparability: lr 1e-5 cosine, warmup 0.03, wd 0.01, betas 0.9/0.999,
  grad clip 1.0, bf16, ZeRO-3, seed 42.
- cutoff_len 8192; num_train_epochs 2; per_device_bsz 1 × grad_accum 4 × 32 GPU = global 128 seq.
- Deviations from runbook build pipeline (no venv changes allowed): language filtering via Unicode
  script-ratio (not fasttext/langdetect); near-dedup via normalized-prompt hashing (not MinHash);
  length filter via char cap proxy (not exact tokenization); `<think>` blocks stripped for consistency.

---

## Progress log

- 2026-08-25: repo/data/schema recon done; hyperparameter research done; instruction_tuning/ scaffolded.

### Checklist
- [x] Downloads verified (smoltalk2, smolkalam, indic-align)
- [x] Hyperparameter/template/architecture research
- [x] build_sft_mixture.py written (+ smoke-tested on real rows)
- [x] hi + ar mixtures built (300K each, 60/40, seed 42)
- [x] dataset_info.json registered (hi_sft_mix, ar_sft_mix)
- [x] 4 SFT configs + slurm launchers written
- [x] 4 training jobs submitted (dependency DAG on build)
- [x] 4 trainings complete (all COMPLETED, 2 epochs)
- [x] eval (same as CPT eval) run on 4 SFT checkpoints (all 6 eval jobs COMPLETED)
- [x] FINAL_REPORT written + user notified (2026-08-26); monitor cron deleted

### Job IDs (DAG; also in logs/dag_jobs.env)
- build: 243267 (cpu,all — builds both mixtures)
- train (afterok:build): hi-cpt=243268, ar-cpt=243269, hi-base=243270, ar-base=243271
- eval  (afterok:train):
  - hi-cpt-sft:  core=243272, gen=243273
  - hi-base-sft: core=243274, gen=243275
  - ar-cpt-sft:  243276
  - ar-base-sft: 243277
- monitor cron: **bf3ac78a** (every 2h; writes FINAL_REPORT + notifies at ALL_DONE; self-deletes).
  NOTE: durable crons auto-expire after 7 days.

### RECOVERY 2026-08-25 (all 4 trains FAILED, resubmitted)
- Cause: source datasets (e.g. Codeforces problems) contained literal `<image>`/`<video>`/`<audio>`
  placeholder text. LlamaFactory's supervised processor treats these as multimodal tokens ->
  `ValueError: The number of images does not match the number of <image> tokens`. All 4 trains
  (243268-243271) died at dataset-mapping; evals 243272-243277 stuck DependencyNeverSatisfied (cancelled).
- Fix: (a) `build_sft_mixture.py` now `strip_media()` neutralizes `<image>`->`[image]` etc. in `clean_conv`;
  (b) sanitized existing JSONL in place via sed (hi: 74, ar: 2110 image + 1 video + 1 audio -> 0 remaining).
- Resubmitted DAG (new IDs in logs/dag_jobs.env):
  - train: hi-cpt=244150, ar-cpt=244151, hi-base=244152, ar-base=244153
  - eval:  hi-cpt {core=244154,gen=244155}, hi-base {core=244156,gen=244157}, ar-cpt=244158, ar-base=244159
  - Verified 244150/151/152 pass the dataset step (0 image errors, reach "training example:"); 244153 pending GPU.

### Assets created this session
- instruction_tuning/: build_sft_mixture.py, build_sft.slurm, sft.slurm, aggregate_sft.py,
  monitor_sft.sh, configs/qwen3p5_9b_sft_{hi,ar}-{cpt,base}-sft.yaml, SFT_PROGRESS.md
- evaluation/: eval_core_sft.slurm, eval_gen_sft.slurm, eval_ar_sft.slurm
- continued_pretraining/configs/dataset_info.json: +hi_sft_mix, +ar_sft_mix

### Results
See FINAL_REPORT.md (generated by aggregate_sft.py). Headline: the CPT idiom advantage
SURVIVES instruction tuning — cpt-initialized SFT beats base-initialized SFT on every
target-language idiom/culture task in both languages:
- HI: global_piqa 0.60->0.69, milu 0.567->0.583, mabl 0.541->0.563; Hindi PPL much lower
  (samanantar 4.95->3.95, proverbs 4.48->3.01).
- AR: kinayat_meaning 0.105->0.191, ar_figurative 0.248->0.309, kinayat_cloze 0.527->0.560,
  alyah 0.308->0.337; ar PPL lower (fineweb2 14.65->12.20).
English retention (hi, raw-completion eval): flat/up vs pre-SFT cpt; HumanEval recovered
0.59->0.73, BoolQ 0.83->0.89, GSM8K ~0.89. Caveat: absolute loglik idiom-knowledge scores
dropped vs the PRE-SFT cpt checkpoint (esp. ar kinayat_meaning 0.452->0.191) — expected when
scoring an instruction-tuned model in raw-completion loglik mode (same eval as CPT, for
comparability, per the request).
