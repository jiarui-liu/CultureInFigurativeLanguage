# Bidirectional Idiom ⇄ Culture Transfer — Plan, Status, and Remote Runbook

**Owner:** automated run (Claude). **Started:** 2026-09-27. **Last updated:** 2026-09-27.
**Status legend:** ⬜ not started · 🔄 in progress · ✅ done · ⚠️ done with caveats · 🖥️ must run on the other (lustre/H100) server

This document is the single source of truth for the experiments added on top of the
current paper draft (`OverleafCultureInFigurativeLanguage/`). It records (1) what we
plan to run and why, (2) what has been run on *this* cluster (babel, CMU) with result
locations, and (3) what is left to run on the *other* server (the H100 cluster that
trained the Qwen3.5-9B CPT checkpoints under `/lustre-storage/...`), with commands.

> **Starting checkpoint (verified 2026-09-27):** the 9B study started from `Qwen/Qwen3.5-9B` — the post-trained
> release, used as a plain LM — *not* `Qwen3.5-9B-Base` (per-item log-probs of the draft's "Base" column match
> `Qwen/Qwen3.5-9B` to 0.18 nats and accuracy exactly; they differ from `-Base` by 3.3 nats). Consequently the 2B
> study's main runs use `Qwen/Qwen3.5-2B` (run prefix `i_`), and the `Qwen3.5-2B-Base` runs are kept as an
> ablation of the starting checkpoint. The 9B reverse configs already point at the same `Qwen3.5-9B` path.

> **Key discovery (2026-09-27):** the HF dataset repo `Jerry9999/CultureInFigurativeLanguage`
> holds all eight 9B CPT checkpoints except `zh-untagged`, every per-item eval record,
> all training corpora (`data/train_*`), the enriched KBs and the eval data (incl. MILU).
> So **all 9B evaluations of new benchmarks run here**; only 9B *training* (and SFT)
> must run on the other server.

---

## 0. Why these experiments (reviewer's view of the current draft)

The current draft tests one direction: **idiom data → culture tests** (Idiom-CPT on
Qwen3.5-9B, hi/zh/ar, vs a token-matched Random-CPT and a tag-stripped control); the
culture result is mostly null. An ACL/EMNLP reviewer will ask:

1. *Is the null a property of the direction or of the method?* → the **reverse
   direction**: culture-selected / culture-augmented pretraining data → idiom tests.
   The 2×2 transfer matrix {idiom data, culture data} × {idiom tests, culture tests},
   every cell relative to the same token-matched Random control, tells whether
   transfer is absent, symmetric, or asymmetric.
2. *One model, one seed?* → replicate the full design at small scale in the same model
   family (Qwen3.5-2B-Base, optionally 4B) with seeds, all on this cluster.
3. *The central argument ("idioms carry the symbolic layer; benchmarks test facts") has
   no experiment* → **symbolism probe** built from the same-entity analysis, with a
   cross-cultural *lure* option, evaluated on every checkpoint.
4. *Hindi has no idiom-meaning test* → **IdiomAtlas-MC** (hi/zh/ar) with a
   *seen-in-corpus / never-in-corpus* split.
5. Rigor `\todo{}`s in the draft (figurative-free Alyah/DziriEval, Holm correction,
   ArabCulture item count, prompt-independent divergence, LLM typology + χ², cluster
   sizes, Arabic analyses, citations, licences, AI-use statement).

Closest prior work (must be cited): Attia, Diab & Solorio (2026, arXiv 2608.18361)
LoRA-fine-tune Arabic models on ~1k benchmark QA in both directions (culture ↔
figurative) and find no culture→figurative transfer. We differ by (a) raw-web continued
pretraining at 10^8–10^9 tokens, (b) token-matched random controls in both directions,
(c) three languages, (d) selection vs augmentation, (e) the symbolism probe.
See `docs/literature_reviews/bidirectional_related_work.md`, `OverleafCultureInFigurativeLanguage/new_refs_candidates.bib`.

---

## 1. Design

### 1.1 Culture-relevance classifier (FineWeb-Edu style)
- **Pool** (same sources as each language's Random control; built by `build_pool.py`):
  ar = FineWeb-2 `arb_Arab` files `000_0000{0-4}` (every 6th row group) → 2.18M docs;
  zh = Fineweb-Edu-Chinese-V2.1 tier `4_5`, every 4th file → 4.46M docs;
  hi = mC4-hi (every 4th shard) + FineWeb-2 `hin_Deva` (file 000) → 7.35M docs.
  Gates: ar `quality_ar.reject_reason` + 300–25k chars (the Arabic idiom-corpus gate);
  zh/hi 150–100k chars (the random-control gate). Every doc stores its matched KB idioms.
  KB-idiom document rate: ar 1.0%, hi 1.3%, zh 47.3%.
- **Annotation** (`annotate_culture.py`): Gemma-4-26B-A4B-it (local vLLM) rates ~10k
  random pool docs/language on a 0–5 *culture-specificity* rubric (customs, rituals,
  festivals, food, religion, norms, values, folklore, history, arts…); idioms/proverbs
  are explicitly excluded from the rubric.
- **Classifier** (`culture_classifier.py`): Qwen3-Embedding-0.6B (first 1,024 tokens) +
  ridge regression; held-out Spearman ρ and P/R/F1 at score ≥ 3 reported.

### 1.2 Arms (per language). All arms: exactly the same number of 4,096-token blocks, 1 epoch, same steps
| Arm | Data | Tests |
|---|---|---|
| Base | no training | — |
| Random | uniform pool sample | control |
| Idiom−tags | 9B idiom corpus docs, meaning tags stripped | idiom → culture |
| Idiom-CPT | 9B idiom corpus docs + meaning tags (same docs, prefix) | idiom → culture |
| **Culture** | top classifier-scored pool docs, **idiom-free** (no KB idiom) | culture → idiom |
| **Culture+notes** | same ranking (prefix) + appended LLM *cultural notes* (analogue of meaning tags; idioms forbidden and filtered) | culture → idiom |

Choices: idiom-free culture docs make the reverse test conservative (Culture sees *fewer*
idioms than Random). Tags/notes count inside the budget; the +tags/+notes arm is a prefix
of the same ordered list as the −tags/−notes arm. Small scale: Qwen3.5-2B-Base,
300M tokens/arm (73,242 blocks → 572 steps of 128×4,096), LR 2e-5 cosine, warmup 3%,
wd 0.01, ZeRO-2, bf16 (`train_cpt.py`).

### 1.3 Evaluation (harness `culture.evaluation.run_eval`, log-likelihood MC; `eval.slurm`)
- Idiom meaning: Kinayat-Meaning (ar), Chengyu-Bench connotation + **appropriateness** (zh), **IdiomAtlas-MC seen/unseen** (hi/zh/ar).
- Figurative inference: AR-Figurative (ar), MABL (hi). Idiom cloze: Kinayat-Cloze, ChID (`chid_eval3000`, same as 9B).
- **Symbolism probe** (zh/hi/ar): accuracy + lure rate.
- Culture: ArabCulture, ArabicCulturalQA, Alyah (± figurative items), DziriEval (±), Global-PIQA (ar, hi), CCPM.
- Regional: ArabicMMLU, MILU (from the HF repo), CMMLU.

### 1.4 New benchmarks
- **IdiomAtlas-MC** (`build_idiomatlas_mc.py`): idiom → 4 figurative meanings (gold = first
  KB meaning; 3 distractors = other idioms' meanings with embedding cosine 0.35–0.75 to the
  gold and 0.5–2× its length). Items where the gold or any distractor shares >15% of the
  idiom's words (hi/ar) or any character (zh) are removed, which brings a lexical-overlap
  baseline to chance (zh 0.239, hi 0.231). Seen = idiom in the 9B idiom corpus; unseen =
  in no scanned document. Sizes: hi 600/600, ar 600/600, zh 600/78 (only 153 chengyu are
  never seen on the web).
- **Symbolism probe** (`build_symbolism_probe.py`): per entity shared by English and L,
  options = L-specific association (gold), English-specific association (lure), two
  unattested associations; generated by Gemma-4-26B-A4B-it from the evidence idioms, then
  **blind-verified by a second family (Qwen3.5-27B-FP8)** which must independently pick the
  gold as L-supported and the lure as English-supported; only agreeing items are kept.

---

## 2. Status on this cluster (babel)

Paths: `B=/data/group_data/r3lit_culture_pretrain/culture/bidir`; code
`src/culture/bidirectional/`; env `/data/group_data/r3lit_culture_pretrain/envs/bidir`
(py3.12, torch 2.13+cu130, transformers 5.17, vllm 0.30, fla 0.5.2).

| # | Item | Status | Output / note |
|---|---|---|---|
| S0 | Env; Qwen3.5-{2B,4B,9B}-Base, Qwen3-Embedding-0.6B, Gemma-4-26B-A4B-it | ✅ | `/data/group_data/r3lit_culture_pretrain/models/` |
| S1 | Benchmarks (ar all; zh ChID/Chengyu-Bench/CCPM/CMMLU; hi MABL/Global-PIQA/MILU) | ✅ | `$B/eval_data/`, `$B/hf9b/data/eval/` |
| S2 | Arabic KB rebuilt (10,386 entries = paper) ; enriched KB from HF used for entities | ✅ | `culture/data/idioms/ar/`, `$B/ar_entities.jsonl` |
| S3 | Pools (ar 2.18M, zh 4.46M, hi 7.35M docs) | ✅ | `$B/pool/` |
| S4 | 9B per-item records + ArabCulture qid bug fixed in `compute_cis.py` | ✅ | `docs/paper_stats/v2/ci_report.json` (ArabCulture n=3,463: 45.8/49.1/48.7/49.1) |
| S5 | Alyah/DziriEval w/o figurative items; Holm | ✅ | `docs/paper_stats/v2/stats_9b.json` |
| S6 | IdiomAtlas-MC hi/zh/ar | ✅ | `$B/eval_data/mc/idiomatlas_mc_*` |
| S7 | Throughput: 2B ≈ 50 s/step on 4×L40S (ZeRO-2, micro 1) → ≈ 8 h per arm | ✅ | |
| S8 | Culture annotation (Gemma-4-26B-A4B, 10k docs/lang) → classifier → pool scoring | 🔄 | ar/hi done (≥3: 8.2% / 8.1%); classifier held-out Spearman ar 0.76, hi 0.75; zh running |
| S9 | Base arms (random, idiom±tags) built + packed (300M tokens each) | 🔄 | zh ✅, ar ✅ built, hi running (`prep_arms_local.sh` on the login node) |
| S10 | Culture + Culture+notes arms | ⬜ | automatic via `orchestrate.sh` |
| S11 | 2B training, 5 arms × 3 langs | 🔄 | zh-random running; rest queued by `orchestrate.sh` |
| S12 | 2B evaluation (full suite) | 🔄 | base done (ar/zh), hi rerun at batch 4 |
| S13 | 9B checkpoints on new benchmarks | ✅ | IdiomAtlas-MC seen/unseen (3 langs), Chengyu-Bench appropriateness; `docs/paper_stats/v2/idiomatlas_9b.json` |
| S14 | Symbolism probe (Qwen3.5-27B-FP8 generates with ≥2 evidence idioms per side; Gemma-4 blind-verifies) + KB audit | 🔄 | v1 items were low quality (abstract entities, polysemy) → regenerated with grounding checks |
| S15 | Section-4 analyses (typology+χ², divergence score, cluster sizes, pair precision) | 🔄 | analysis sub-agent; outputs `docs/paper_stats/analysis_v2/` |
| S16 | 9B Culture corpora for R1: ar (stream 4 FineWeb-2 files, score, top 1.13B tok), hi (top 1.37B tok of the scored pool), zh (stream full FWE-zh 4_5, top 7.8B tok); notes for ar/hi | ⬜ | `stream_select_culture.py`, `build_arms.py --cmd full`; then upload to HF (`data/train_{l}_{culture,culturenotes}`) |

**Infrastructure lessons (see memory `reference_babel_io`):** NFS reads are ~15 MB/s, so models are
downloaded straight from the HF Hub to `/scratch` (`stage_hf.py`), the env is built per node from
PyPI (`stage_env.sh`), intermediate checkpoints go to `/scratch`; `sbatch --export` splits on commas,
so eval task lists use `+`; NCCL P2P must be disabled on some nodes.

### 2.1 Results so far
- **Alyah without its 214 figurative items (n=957):** Idiom-CPT vs Random +2.3 [0.4, 4.2], p=0.021 → the Alyah gain is *not* only figurative items. DziriEval w/o figurative (n=850): +1.7 [−0.4, 3.7], p=0.12.
- **Holm over 17 benchmarks (Idiom-CPT vs Random):** survive: Kinayat-Meaning (p_adj 2.7e-15), Chengyu-Bench (1.5e-7), Alyah (0.021). AR-Figurative p_adj=0.107 overall, 0.015 within its group.
- **IdiomAtlas-MC on the 9B checkpoints** (Idiom-CPT − Random, pp): seen ar +25.2*, hi +25.3*, zh +10.7*; unseen ar −4.3* (errors lean to distractors whose glosses were in the tags), hi +1.5, zh +3.8 (n=78). Idiom−tags − Random: seen ar +3.5*, hi +6.2*; unseen ar +3.2*, hi +1.0. ⇒ tags install item-specific dictionary knowledge; generalization comes from idioms in context. Chengyu-Bench appropriateness: Base 60.8 / Random 61.0 / Idiom 60.8 (tie). (Base = Qwen/Qwen3.5-9B, re-run 2026-09-27; the first Base runs used -Base by mistake and were discarded.)
- **ArabCulture:** the 2,168 in the draft came from a qid-collision bug (1,295 items silently dropped). Fixed: n=3,463, Base 45.8 / Random 48.7 / −tags 49.1 / Idiom-CPT 49.1; Δ vs Random +0.4 [−0.5, 1.3], unchanged conclusion.

---

## 3. Left to run on the other server 🖥️

Everything below assumes the remote layout used by the existing runs
(`CPT_DIR=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/continued_pretraining`,
data root `/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data`, checkpoints under
`/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts`, venv
`/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv`, 4 nodes × 8 H100).

### 3.0 Sync code (once)
The new/changed files live in the babel checkout (uncommitted; nothing was committed or pushed):
- `src/culture/bidirectional/` (new package), `docs/plans/bidirectional_transfer_plan.md`
- `src/culture/training/continued_pretraining/configs/qwen3p5_9b_cpt_{ar,hi,zh}_{culture,culturenotes}.yaml` and 6 new entries in `configs/dataset_info.json`
- `src/culture/training/instruction_tuning/configs/qwen3p5_9b_sft_{hi,zh}-unfiltered-sft.yaml`, `qwen3p5_9b_sft_ar-unfiltered-native-sft.yaml`
- `src/culture/evaluation/run_eval.py` (`jsonl:` tasks, `chengyu_bench_app`), `compute_cis.py` (ArabCulture qid fix), `training/mC4/filter_and_tag_ar.py` (local parquet paths)

```bash
# on babel
cd /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage && git add -A src docs && git commit -m "Bidirectional idiom<->culture study" && git push
# on the remote server
cd /storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage && git pull
```

### 3.1 R1 — 9B reverse-direction CPT (Culture, Culture+notes) — **the main missing experiment**
Corpora are built on babel (see §2, item S16) and uploaded to a **private** HF dataset repo
`Jerry9999/culture-bidir-private` (the main repo is public, so new web text and LLM-written notes are not put there)
under `data/train_{lang}_{culture,culturenotes}/`
(same `{"text": ...}` jsonl format as `data/train_ar_untagged/`). Token budgets match the
9B Idiom-CPT corpora (ar 1.13B, hi 1.37B, zh 7.8B unique tokens) and every run uses the
Idiom-CPT step count (`max_steps` ar 1608 / hi 2100 / zh 11157), so processed tokens are exactly matched.

```bash
DATA=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data
cd $DATA && huggingface-cli download Jerry9999/culture-bidir-private --repo-type dataset \
   --include "data/train_*_culture/*" "data/train_*_culturenotes/*" --local-dir hf_bidir   # needs a token with access (export HF_TOKEN=...)
for d in hf_bidir/data/train_*_culture hf_bidir/data/train_*_culturenotes; do ln -sfn $PWD/$d $DATA/$(basename $d); done
cd $CPT_DIR
for l in ar hi zh; do
  sbatch cpt_untagged.slurm qwen3p5_9b_cpt_${l}_culture.yaml
  sbatch cpt_untagged.slurm qwen3p5_9b_cpt_${l}_culturenotes.yaml     # zh: only if train_zh_culturenotes exists (see S16)
done
```
Expected wall time (4 nodes): ar ≈ 5 h, hi ≈ 5 h, zh ≈ 30 h per run.
Then upload each final model (to the private repo, or to the public one like the existing checkpoints if you prefer), so babel can evaluate them:
```bash
for m in qwen3p5-9b-{ar,hi,zh}-cpt-{culture,culturenotes}; do
  huggingface-cli upload Jerry9999/culture-bidir-private \
    /lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/$m models/$m --repo-type dataset \
    --exclude "checkpoint-*/*" "global_step*/*"
done
```
Babel evaluates them on the full suite with one command per checkpoint (`src/culture/bidirectional/eval.slurm`,
`MODEL=dataset:Jerry9999/culture-bidir-private:models/<m>`); see §4.


#### 3.1b If a culture corpus is not yet on HF: build it on the remote server
The classifier weights are tiny (`$B/clf/{ar,hi,zh}.pkl`, uploaded to the private repo under `clf/`), and the
H100 nodes embed and generate much faster than babel's shared L40S/NFS, so any missing corpus can be built remotely.
Needs the repo code (§3.0) and a Python env with `vllm>=0.30`, `transformers>=5.17`, `sentence-transformers`,
`pyahocorasick` (the babel env is frozen in `src/culture/bidirectional/requirements_bidir.txt`).
```bash
cd /storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH=$PWD/src OUT=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir
huggingface-cli download Jerry9999/culture-bidir-private --repo-type dataset --include "clf/*" --local-dir $OUT
# zh: score all of Fineweb-Edu-Chinese-V2.1 tier 4_5 (one GPU per part; 16 parts ~ 1 h on H100)
for p in $(seq 0 15); do
  srun --gres=gpu:1 --cpus-per-task=12 python -m culture.bidirectional.stream_select_culture score --lang zh \
    --repo opencsg/Fineweb-Edu-Chinese-V2.1 --pattern '4_5/*.parquet' --clf $OUT/clf/zh.pkl --min_score 0.5 \
    --out_dir $OUT/full/zh --part $p --nparts 16 --workers 11 --tmp_dir /tmp/stream_$p &
done; wait
python -m culture.bidirectional.stream_select_culture finalize --lang zh --out_dir $OUT/full/zh \
  --budget_tokens 7800000000 --chars_per_token 1.76
ln -sfn $OUT/full/zh/train /lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/train_zh_culture
# cultural notes for a corpus (e.g. ar); 8 parts on 8x2 GPUs, then concatenate doc+notes into train_ar_culturenotes
for p in $(seq 0 7); do
  srun --gres=gpu:2 python -m culture.bidirectional.culture_notes --lang ar --input $OUT/full/ar/culture_docs_ranked.jsonl.gz \
    --model google/gemma-4-26B-A4B-it --tp 2 --part $p --nparts 8 --out_dir $OUT/full/ar/notes &
done; wait
python -m culture.bidirectional.build_arms --lang ar --cmd full_notes --out_dir $OUT/full/ar
ln -sfn $OUT/full/ar/train_notes /lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/train_ar_culturenotes
```

### 3.2 R2 — Instruction tuning the Random-CPT checkpoints (decisive IT control; draft §5 discussion)
```bash
cd /storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/instruction_tuning
sbatch sft.slurm configs/qwen3p5_9b_sft_hi-unfiltered-sft.yaml
sbatch sft.slurm configs/qwen3p5_9b_sft_zh-unfiltered-sft.yaml
sbatch sft.slurm configs/qwen3p5_9b_sft_ar-unfiltered-native-sft.yaml
# evaluate with the same scripts as the other SFT runs, e.g.
sbatch ../../evaluation/eval_core_sft.slurm hi-unfiltered-sft     # run names added to the case blocks of all three scripts
sbatch ../../evaluation/eval_zh_sft.slurm zh-unfiltered-sft
sbatch ../../evaluation/eval_ar_sft_full.slurm ar-unfiltered-native-sft
```
Then upload the three SFT models (or just their `eval/` result folders) to the HF repo under
`models/` / `eval/`, and run `compute_cis.py` over base-sft / unfiltered-sft / cpt-sft.

### 3.3 R3 — zh Idiom−tags checkpoint (missing from HF)
`qwen3p5-9b-zh-cpt-untagged` is the only 9B checkpoint not on the HF repo, so the new benchmarks
(IdiomAtlas-MC, Chengyu-Bench appropriateness, symbolism probe) cannot be run on it here. Either upload it
(`huggingface-cli upload ... models/qwen3p5-9b-zh-cpt-untagged`) or evaluate it remotely with §4's task list.

### 3.4 R4 (optional) — exact-budget Hindi Random-CPT
Hindi Random processed 4.09B tokens vs 4.40B for Idiom-CPT. To remove the 7% gap, re-run with
`max_steps: 2100` (copy `qwen3p5_9b_cpt_unfiltered.yaml`, add `max_steps: 2100`, new output_dir).

---

## 4. Evaluating new 9B checkpoints (runs on babel once they are on HF)
```bash
cd /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional
B=/data/group_data/r3lit_culture_pretrain/culture/bidir; HFD=dataset:Jerry9999/culture-bidir-private:models
AR=kinayat_meaning+kinayat_cloze+ar_figurative+alyah+dzirieval+arabculture+arabic_cultural_qa+global_piqa_ar+global_piqa_ar_parallel+arabicmmlu+jsonl:idiomatlas_mc_ar_seen+jsonl:idiomatlas_mc_ar_unseen+jsonl:symbolism_ar
HI=mabl+global_piqa+milu+jsonl:idiomatlas_mc_hi_seen+jsonl:idiomatlas_mc_hi_unseen+jsonl:symbolism_hi
ZH=chengyu_bench+chengyu_bench_app+chid+ccpm+cmmlu+jsonl:idiomatlas_mc_zh_seen+jsonl:idiomatlas_mc_zh_unseen+jsonl:symbolism_zh
for arm in culture culturenotes; do
  sbatch --export=ALL,MODEL=$HFD/qwen3p5-9b-ar-cpt-$arm,OUT=$B/eval9b/ar/$arm,TASKS=$AR eval.slurm
  sbatch --export=ALL,MODEL=$HFD/qwen3p5-9b-hi-cpt-$arm,OUT=$B/eval9b/hi/$arm,TASKS=$HI eval.slurm
  sbatch --export=ALL,MODEL=$HFD/qwen3p5-9b-zh-cpt-$arm,OUT=$B/eval9b/zh/$arm,TASKS=$ZH eval.slurm
done
```
(Task lists are `+`-separated because `sbatch --export` splits on commas; `eval.slurm` converts them.)
The remote server can run the same evaluation after copying `$B/eval_data/mc/*.jsonl` (also uploaded to the HF
repo under `data/eval/mc/`) and passing `--mc_jsonl_dir`.
