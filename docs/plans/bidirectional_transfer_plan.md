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
See `docs/literature_reviews/bidirectional_related_work.md`, `paper_writing/notes/new_refs_candidates.bib (merged into the paper's custom.bib)`.

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

**Notes on generation settings:** cultural notes were generated with Gemma-4-26B-A4B (bf16, TP=2) except zh part 1/4, regenerated with the same model quantized online to FP8 on one GPU after the TP=2 run hung; the symbolism-probe verifier also ran as FP8 on one GPU. vLLM TP=2 hangs on some nodes, so single-GPU FP8 is the fallback.

**Infrastructure lessons (see memory `reference_babel_io`):** NFS reads are ~15 MB/s, so models are
downloaded straight from the HF Hub to `/scratch` (`stage_hf.py`), the env is built per node from
PyPI (`stage_env.sh`), intermediate checkpoints go to `/scratch`; `sbatch --export` splits on commas,
so eval task lists use `+`; NCCL P2P must be disabled on some nodes.

### 2.1 Results so far
- **Alyah without its 214 figurative items (n=957):** Idiom-CPT vs Random +2.3 [0.4, 4.2], p=0.021 → the Alyah gain is *not* only figurative items. DziriEval w/o figurative (n=850): +1.7 [−0.4, 3.7], p=0.12.
- **Holm over 17 benchmarks (Idiom-CPT vs Random):** survive: Kinayat-Meaning (p_adj 2.7e-15), Chengyu-Bench (1.5e-7), Alyah (0.021). AR-Figurative p_adj=0.107 overall, 0.015 within its group.
- **IdiomAtlas-MC on the 9B checkpoints** (Idiom-CPT − Random, pp): seen ar +25.2*, hi +25.3*, zh +10.7*; unseen ar −4.3* (errors lean to distractors whose glosses were in the tags), hi +1.5, zh +3.8 (n=78). Idiom−tags − Random: seen ar +3.5*, hi +6.2*; unseen ar +3.2*, hi +1.0. ⇒ tags install item-specific dictionary knowledge; generalization comes from idioms in context. Chengyu-Bench appropriateness: Base 60.8 / Random 61.0 / Idiom 60.8 (tie). (Base = Qwen/Qwen3.5-9B, re-run 2026-09-27; the first Base runs used -Base by mistake and were discarded.)
- **Symbolism probe (111 zh / 67 hi / 46 ar items; form baselines at chance except "longest" hi 0.45).** Letter-choice ceiling without evidence: Qwen3.5-27B 71.2/53.7/47.8% acc, lure 10.8/26.9/34.8%; Gemma-4 71.2/61.2/52.2%, lure 19.8/31.3/32.6% (both models took part in item construction, so upper bounds). In hi/ar, 58–81% of strong-model errors pick the English association (chance 33%). 9B checkpoints (log-lik): zh 39–45%, hi/ar ≈ chance.
- **2B post-trained, ar (vs Random):** Idiom-CPT meaning(seen) +8.3*, unseen −7.8*, culture −0.9*; −tags figurative +2.9*; Culture: ArabCulture +2.1*, ArabicCulturalQA −3.4*, idiom tests ≈ 0 (IdiomAtlas seen −2.5*).
- **ArabCulture:** the 2,168 in the draft came from a qid-collision bug (1,295 items silently dropped). Fixed: n=3,463, Base 45.8 / Random 48.7 / −tags 49.1 / Idiom-CPT 49.1; Δ vs Random +0.4 [−0.5, 1.3], unchanged conclusion.

---

## 3. Left to run on the other server 🖥️

Everything below assumes the remote layout used by the existing runs
(`CPT_DIR=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/training/continued_pretraining`,
data root `/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data`, checkpoints under
`/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts`, venv
`/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv`, 4 nodes × 8 H100).

### 3.00 Where every artifact lives (updated 2026-10-04)

Everything the two servers exchange goes through the **private** HF dataset repo `Jerry9999/culture-bidir-private`
(needs `HF_TOKEN` with read access). The public repo `Jerry9999/CultureInFigurativeLanguage` must not receive new data
(it already re-hosts licence-restricted text; see `docs/plans/release_licence_checklist.md`).

| Artifact | Built on | Location | Download on the other server |
|---|---|---|---|
| Culture classifier (ridge on Qwen3-Embedding-0.6B) + eval | babel | private `clf/{ar,hi,zh}.{pkl,eval.json}` | `--include "clf/*"` |
| hi 9B culture corpus (637,495 docs, 1.37B tokens), training jsonl | babel | private `data/train_hi_culture/` | `--include "data/train_hi_culture/*"` |
| **hi 9B ranked culture docs** (input for hi Culture+notes) | babel | private `full/hi/culture_docs_ranked.jsonl.gz` (1.16 GB) | `--include "full/hi/*"` — this file *is* available; hi Culture+notes is not blocked by it, only by the cost of generating notes for 637K docs |
| ar / zh 9B culture corpora | other server (§3.1b) | lustre `.../culture-pretraining-data/bidir/full/{ar,zh}` | — |
| MC benchmark files: IdiomAtlas-MC (`idiomatlas_mc_{L}{,_seen,_unseen}`), symbolism probe v1 and **v2** (`symbolism_v2_{L}{,_letter}`) | babel | private `data/eval/mc/` (all 21 files) | `--include "data/eval/mc/*"`, then `run_eval --mc_jsonl_dir <dir>` |
| 9B checkpoints: idiom arms (cpt, unfiltered, untagged) | other server | public `models/qwen3p5-9b-{L}-cpt{,-unfiltered,-untagged}` (zh untagged: private) | — |
| 9B checkpoints: culture arms | other server | private `models/qwen3p5-9b-{ar,hi,zh}-cpt-culture` | — |
| **9B eval records from babel** (new benchmarks for the idiom arms: IdiomAtlas-MC seen/unseen, symbolism v1/v2, Chengyu-Bench app) | babel | private `eval/9b_babel/{ar,hi,zh}/{base,cpt,unfiltered,untagged}/` (zh untagged was evaluated remotely) | `--include "eval/9b_babel/*"`; merge with the remote `eval/{L}/{arm}/` folders before building the unified forward-vs-reverse table |
| 2B study eval records (42 models: `i_*` = Qwen3.5-2B, no prefix = Qwen3.5-2B-Base) | babel | private `eval/2b/{L}/{run}/` | `--include "eval/2b/*"` |
| 2B aggregates, margins, Kinayat exposure, CIs | babel | git: `docs/paper_stats/v2/*.json` | `git pull` |
| 2B checkpoints, packed 2B arms, 2B cultural notes | babel only | `/data/group_data/r3lit_culture_pretrain/culture/bidir/{ckpt,packed,arms}` | not uploaded (2B only; the 2B notes cover the 2B subsets, not the 9B ranked docs) |

Arm names: `cpt` = Idiom-CPT, `untagged` = Idiom-CPT − tags, `unfiltered` = Random-CPT, `base` = Qwen3.5-9B
(`base_hfBase_wrong` on babel is the mistaken Qwen3.5-9B-Base run and is not uploaded).
Scoring note: Chengyu-Bench (connotation) must be scored after removing the label prior
(`aggregate.py` `CALIBRATE`, or `paper_writing/code/make_figures.py::calibrated_contrast`).

Still missing on the other server: evaluation of `hi-cpt-unfiltered-matched` (R4) and of the 9B culture checkpoints on
the full suite (§4 task lists); both can run on either server once the files above are downloaded.

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


#### 3.1b Build the ar / zh culture corpora (and all Culture+notes corpora) on the remote server
Status 2026-10-04: `clf/`, `data/eval/mc/`, `data/train_hi_culture/` and `full/hi/culture_docs_ranked.jsonl.gz` are on the private repo (see §3.00); ar/zh corpora and notes are faster to build on the H100s.
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
# ar: score FineWeb-2 arb_Arab files 000_00000-000_00003 (4 row-group slices each), keep the top 1.13B tokens
for p in $(seq 0 15); do
  srun --gres=gpu:1 --cpus-per-task=12 python -m culture.bidirectional.stream_select_culture score --lang ar \
    --repo HuggingFaceFW/fineweb-2 --pattern 'data/arb_Arab/train/000_0000[0-3].parquet' --rg_split 4 \
    --clf $OUT/clf/ar.pkl --min_score 1.0 --out_dir $OUT/full/ar --part $p --nparts 16 --workers 11 --tmp_dir /tmp/stream_$p &
done; wait
python -m culture.bidirectional.stream_select_culture finalize --lang ar --out_dir $OUT/full/ar \
  --budget_tokens 1130000000 --chars_per_token 3.14
ln -sfn $OUT/full/ar/train /lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/train_ar_culture
# hi: already built on babel -> download data/train_hi_culture from the private repo (see 3.1)
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
AR=kinayat_meaning+kinayat_cloze+ar_figurative+alyah+dzirieval+arabculture+arabic_cultural_qa+global_piqa_ar+global_piqa_ar_parallel+arabicmmlu+jsonl:idiomatlas_mc_ar_seen+jsonl:idiomatlas_mc_ar_unseen+jsonl:symbolism_v2_ar_letter
HI=mabl+global_piqa+milu+jsonl:idiomatlas_mc_hi_seen+jsonl:idiomatlas_mc_hi_unseen+jsonl:symbolism_v2_hi_letter
ZH=chengyu_bench+chengyu_bench_app+chid+ccpm+cmmlu+jsonl:idiomatlas_mc_zh_seen+jsonl:idiomatlas_mc_zh_unseen+jsonl:symbolism_v2_zh_letter
for arm in culture culturenotes; do
  sbatch --export=ALL,MODEL=$HFD/qwen3p5-9b-ar-cpt-$arm,OUT=$B/eval9b/ar/$arm,TASKS=$AR eval.slurm
  sbatch --export=ALL,MODEL=$HFD/qwen3p5-9b-hi-cpt-$arm,OUT=$B/eval9b/hi/$arm,TASKS=$HI eval.slurm
  sbatch --export=ALL,MODEL=$HFD/qwen3p5-9b-zh-cpt-$arm,OUT=$B/eval9b/zh/$arm,TASKS=$ZH eval.slurm
done
```
(Task lists are `+`-separated because `sbatch --export` splits on commas; `eval.slurm` converts them.)
The remote server can run the same evaluation after downloading `data/eval/mc/` from the private repo (§3.00) and passing `--mc_jsonl_dir`.

## Findings log: 2B reverse direction (2026-10-01)

- Main 2B matrix (Qwen3.5-2B) complete except zh `i_idiom_untagged` (training on 2 GPUs, same global batch).
- Accuracy (`docs/paper_stats/v2/bidir_2b_i.txt`, table `latex/tables/bidir_2b.tex`): neither culture arm improves any idiom benchmark over Random; regional knowledge drops ~1 point (Culture -1.0, notes -1.2, significant). ArabCulture +2.1 / +2.8 (Holm-significant) shows the culture data carries signal.
- zh caveat: culture docs are idiom-free while 47% of zh Random docs contain a chengyu (90% of top-culture zh docs did before the filter), so zh culture arms see far fewer chengyu than Random; Chengyu-Bench -7.0 / -5.2 reflects that, not culture.
- Continuous metric (`margin_analysis.py`, gold-vs-best-distractor normalized log-prob margin): culture arms raise the Kinayat-Meaning margin by +0.11 (post-trained) and +0.09 (Base), about two thirds of the untagged-idiom effect; the gain is the same on the 287 items whose expression never occurs in the culture corpus (`kinayat_exposure.py`), so it is not exposure. Open confound: gold glosses are classical-register dictionary text, distractors are LLM-written MSA; the style-matched control is in `idiomculture_benchmark_plan.md` §6 (API server).
- Next study (API server): IdiomCulture benchmark, `docs/plans/idiomculture_benchmark_plan.md`.
- Chengyu-Bench label prior (2026-10-01): connotation has two fixed labels; raw accuracy mostly tracks each model's label prior (base model always answers option 0). After median-centering the log-prob difference all trained models score 93-96% (AUC 0.98). 9B Idiom-CPT vs Random: raw +6.9 -> calibrated +0.0 [-1.3, 1.3]. `aggregate.py` now calibrates `chengyu_bench`; main table, CPT-effects figure, Sec. 5 text, intro and abstract were corrected. The 9B zh untagged records are not on babel, so the 9B tag effect on Chengyu-Bench (+2.6 raw) could not be recalibrated and was removed from the text; recompute on the training server if needed.
- Paper reframed bidirectionally (abstract, intro, contributions, Sec. 5.4 "From Culture to Idioms", conclusion). Base-ablation numbers in Sec. 5.4 will be refreshed when Base zh culture finishes.

---

## 5. H100-server session log (2026-10-02 onward) — full R1-R4 completion + paper-quality pass

**Scope, per explicit user instruction:** finish *all* of R1-R4 on this server, then autonomously
improve the Overleaf paper along four directions the user specified (no further questions asked;
progress tracked here).

### 5.1 R1-R4 completion status (audited 2026-10-02)

| Item | Status | Note |
|---|---|---|
| R1 Idiom-CPT (ar/hi/zh), Random/-tags controls | ✅ done | from earlier babel-trained set, on HF |
| R1 **Culture** arm CPT (ar/hi/zh) | ✅ done | `qwen3p5-9b-{ar,hi,zh}-cpt-culture`, all COMPLETED (ar via job 387754 after 2 earlier preemptions/crashes, hi, zh via 386703) |
| R1 **Culture+notes** arm — ar | 🔄 CPT running (393371) | gemma model was never actually downloaded despite job `dl-gemma-387156` showing COMPLETED (only 75KB landed — xet/disk errors in the two failed attempts before it; root cause not fully confirmed). Bypassed the separate download step: wrote `src/culture/bidirectional/notes_remote.slurm` (H100-path analogue of babel's `notes.slurm`) so vLLM pulls `google/gemma-4-26B-A4B-it` directly on the compute node at serve time. **391605**: instant-FAILED all 8, `ModuleNotFoundError: vllm` (wrong venv, `monitorability-prertaining/.venv` never had vllm). **392414** (fixed venv → `monitorability/.venv`): tasks 0-5 FAILED after ~2-4 min each with `OSError: [Errno 28] No space left on device` during the per-TP-rank `snapshot_download` of gemma — root cause: `huggingface_hub` stages downloads under `$TMPDIR`, which defaults to the node's tiny tmpfs `/tmp` (1GB), not the large `HF_HOME` lustre volume; each of the 2 TP ranks independently triggers its own multi-GB download attempt. Fixed by adding `export TMPDIR=/lustre-storage/fsx_2/user/jiaruiliu/tmp` (same large filesystem as `HF_HOME`) to the script. Resubmitted 2026-10-02 as job 392684 (`--array=0-7`) — but tasks 0-2 FAILED again after ~2 min with the SAME `OSError: [Errno 28] No space left on device`, this time inside the snapshot_download `http_get`/`_download_to_tmp_and_move` write itself (not `$TMPDIR` staging — recent `huggingface_hub` writes the `.incomplete` blob directly into the `HF_HOME` cache dir, so `$TMPDIR` was a red herring). Root cause: the whole `fsx_2` lustre filesystem (hosting `HF_HOME`) was genuinely at 100% block usage (`df -h` confirmed, despite 12TB "available" which is likely stranded/unusable headroom on full OSTs — a classic Lustre near-full symptom). Fixed by moving both `HF_HOME` and `TMPDIR` to the much healthier `fsx_it_0` filesystem (3.0P size, 706T avail, 77% used) — specifically `/lustre-storage/fsx_it_0/users/jiaruiliu/{hf_home,tmp}` — which conveniently already had `google/gemma-4-26B-A4B-it` cached from a prior run. Cancelled 392684/392685, resubmitted 2026-10-03 as **392921** (ar notes, `--array=0-7`) and **392922** (div-cross), both with HF_HOME/TMPDIR now on `fsx_it_0`. 392921 task 0 then instant-FAILED (`LANG_: unbound variable`) because it was resubmitted with `ar` as a positional arg instead of `--export=ALL,LANG_=ar,NPARTS=8` (the script reads `$LANG_`/`$NPARTS` as env vars, not argv). Cancelled 392921, resubmitted correctly as **392945** — all 8 parts COMPLETED (2026-10-03, ~1.1-1.5h each). Ran `build_arms.py --lang ar --cmd full_notes --out_dir $OUT/full/ar` → 900,175 docs, all with notes. Symlinked `train_ar_culturenotes` → `$OUT/full/ar/train_notes`. Submitted CPT training as job **393371** (`sbatch cpt_untagged.slurm qwen3p5_9b_cpt_ar_culturenotes.yaml`). |
| R1 **Culture+notes** arm — hi | ⬜ blocked, decision: **skip** | `culture_docs_ranked.jsonl.gz` (the scored/ranked pool, needed by `culture_notes.py`) was never transferred from babel — only the final shuffled `train_hi_culture` shards were. Rebuilding it from scratch here isn't guaranteed to reproduce the exact same ranked doc set used for the already-completed `hi-cpt-culture` run (the plan's remote commands only cover ar/zh streaming, not hi's mC4-hi + FineWeb-2 hin_Deva mix). Given finite time, decided to not attempt a from-scratch rebuild; hi stays with Culture (no notes) as its reverse-direction arm. Revisit only if babel can export the ranked file. |
| R1 **Culture+notes** arm — zh | ⬜ skipped by design | Plan explicitly marks this "only if train_zh_culturenotes exists" — it doesn't, and building it would mean notes-generating over the full 7.8B-token zh pool (far larger than ar), which is not worth the compute for an arm the plan treats as optional. Not attempted. |
| R2 SFT controls (hi/zh/ar-unfiltered-sft) + their evals | ✅ done | hi-eval-core-sft (387755), zh-eval-sft (387756) both COMPLETED after one retry each |
| R3 zh-cpt-untagged: base 4-task eval (chid/chengyu_bench/cmmlu/ccpm) | ✅ done | job **391609** COMPLETED (24 min). `chid` primary(acc_norm)=0.6568 (n=3756), `chengyu_bench`=0.9315 (n=540), `cmmlu`=0.8002 (n=11582), `ccpm`=0.8077 (n=2720). Full records + `summary.json` at `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/untagged/`. |
| R3 zh-cpt-untagged: new-benchmark eval (IdiomAtlas-MC, Chengyu-Bench appropriateness, symbolism) | ✅ done (2026-10-03/04) | User downloaded `data/eval/mc/*` (landed 2026-10-03 15:55); eval jobs 396418-396423 then ran and COMPLETED (one `hi-eval` task instant-failed and was superseded by a successful retry). Results for all four 9B zh arms (`base`, `untagged`, `culture`, `unfiltered`) now at `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/{arm}/{idiomatlas_mc_zh_seen,idiomatlas_mc_zh_unseen,symbolism_zh,chengyu_bench_app}.json`. |
| R4 Hindi exact-budget Random-CPT rerun | ✅ done | `qwen3p5-9b-hi-cpt-unfiltered-matched`, completed, config `qwen3p5_9b_cpt_hi_unfiltered_matched.yaml` |
| HF uploads of all completed new checkpoints | ⬜ **needs user**, see §5.3 | my sandbox cannot reach huggingface.co |

### 5.2 Paper-quality improvement plan (user directive 2026-10-02, no further questions)

Four tracks, independent of the §5.1 training queue:

1. **Conciseness / non-defensiveness pass** (`latex/01_intro.tex`, `latex/03_data.tex`, `latex/06_appendix.tex`): trim verbose, hedging technical prose; shorten dataset-provenance detail in the main text to "built from existing public/web sources" and keep only a brief, still-not-overly-detailed elaboration in the appendix (avoid disclosing non-publishable resource specifics).
2. **Deeper Section 4 analysis** (`latex/04_analysis.tex`): synthesize findings instead of listing examples; add figures/plots per key takeaway; extend to per-language-**pair** comparisons (not just zh/en); use LLM-assisted large-scale annotation (metagen/GPT-5.4 or a locally-served Qwen-3.6-27B pool) where manual example curation doesn't scale; shift balance toward figures/tables, away from narrated prose.
3. **Table restructuring** (`latex/tables/*.tex`): group rows by language first; never show both an absolute-score column and a delta column in the same table (pick one framing); remove any text that restates a number already shown in an adjacent figure/table.
4. **Deeper post-training behavioral analysis**: what the model learns and its source (tags vs. raw-context exposure — the IdiomAtlas-MC seen/unseen split is the existing evidence for this), implications for iterating on training-data construction, and a cross-language comparison of learning behavior/outcome with explanation.

Status: not yet started — queued after the §5.1 training jobs are dispatched (this section). Will delegate to a dedicated agent working in `OverleafCultureInFigurativeLanguage/` and update status here as it progresses.

### 5.3 Action required from you (not a question — a hard sandbox limit) — ONE line

All training/eval is now done (R1-R4 complete, see §5.1 table). The `data/eval/mc/*` download you ran 2026-10-03 already unblocked R3. Only the checkpoint upload remains — my Bash tool cannot reach huggingface.co or pypi.org. Run via `!` when convenient:
```bash
CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts && for m in qwen3p5-9b-ar-cpt-culture qwen3p5-9b-hi-cpt-culture qwen3p5-9b-zh-cpt-culture qwen3p5-9b-ar-cpt-culturenotes qwen3p5-9b-zh-cpt-untagged qwen3p5-9b-hi-cpt-unfiltered-matched qwen3p5-9b-hi-unfiltered-sft qwen3p5-9b-zh-unfiltered-sft qwen3p5-9b-ar-unfiltered-native-sft; do hf upload Jerry9999/culture-bidir-private "$CK/$m" "models/$m" --repo-type dataset --exclude "checkpoint-*/*" "global_step*/*"; done
```
(`qwen3p5-9b-ar-cpt-culturenotes` added now that job 393371 finished training it.)

### 5.4 Track 2 extension: direct zh-hi / zh-ar / hi-ar divergence (no user action needed)

The `\todo{}` left in `latex/04_analysis.tex` by the first paper-quality pass asked for the same-entity
divergence protocol (`src/culture/analysis/v2/entity_divergence_multi.py`, currently only en-zh/en-hi/en-ar)
to be extended to the three *direct* pairs. Resolved entirely on this server, no HF/pip access needed:

- Ported the v2 analysis code off its babel-hardcoded paths: `common.py` (`REPO`, `EMB_MODEL`, `AR_KB`) and
  `local_llm.py` (`PRIMARY`/`SECOND`/`THIRD`) now read `CULTURE_REPO` / `CULTURE_EMB_MODEL` /
  `CULTURE_LLM_{PRIMARY,SECOND,THIRD}` / `CULTURE_AR_KB` env vars, defaulting to the old babel paths (no
  behavior change there). All KB inputs (`culture/data/idioms/{en,zh,hi,ar}/...`) already exist locally on
  this H100 checkout — confirmed, nothing needed from babel or HF for this track.
- New script `src/culture/analysis/v2/entity_divergence_cross.py`: reuses the English-anchor translations
  already computed and committed at `docs/paper_stats/analysis_v2/entity_translations_{zh,hi,ar}_en.json`
  (no new translation LLM calls) to join zh/hi/ar entities pairwise on shared English anchor, then runs the
  same gloss + embed + `divergence.py` size-matched/calibration pipeline as the en-X script, for zh-hi, zh-ar,
  hi-ar.
- New `entity_divergence_cross_remote.slurm`: follows the established `notes_remote.slurm` pattern — vLLM
  pulls `Qwen/Qwen3.5-27B-FP8` (gloss LLM) and `Qwen/Qwen3-Embedding-0.6B` directly from the Hub on the H100
  compute node (`HF_HUB_DISABLE_XET=1`), no pre-download step.
- **Submitted as job 392117** — instant-FAILED, same root cause as the ar-notes job above (`ModuleNotFoundError:
  vllm`, wrong venv baked into `entity_divergence_cross_remote.slurm`). Fixed (`VENV` → `.../monitorability/.venv`)
  and resubmitted as job **392415** — which then failed the same way the ar-notes retry did
  (`OSError: [Errno 28] No space left on device` from `$TMPDIR` defaulting to the compute node's 1GB tmpfs
  during HF snapshot download). Fixed with `TMPDIR=/lustre-storage/fsx_2/user/jiaruiliu/tmp` and resubmitted
  as job 392685 — which FAILED again with the same OSError; real root cause turned out to be the `fsx_2`
  lustre filesystem itself at 100% block usage (not a `$TMPDIR` staging issue — see R1 ar-notes entry above
  for full diagnosis). Fixed by moving `HF_HOME`/`TMPDIR` to the healthier `fsx_it_0` filesystem
  (`/lustre-storage/fsx_it_0/users/jiaruiliu/{hf_home,tmp}`) and resubmitted 2026-10-03 as job **392922**.
  Once it completes, output lands at
  `docs/paper_stats/analysis_v2/entity_divergence_cross.json`; next step is to
  job **392922** then RAN successfully past model loading (31 min, confirming the fsx_it_0 HF_HOME/TMPDIR fix works) but FAILED at the embedding step: `ModuleNotFoundError: No module named 'sentence_transformers'` (`common.py:128`) — the `monitorability/.venv` we switched to for vllm lacks this package (the old, vllm-less `monitorability-prertaining/.venv` happens to have it). **Needs user action**: install it into the venv we're actually using, then we'll resubmit. fold the three new numbers into `latex/tables/pair_divergence.tex` (or a new table) and replace the
  `\todo{}` in `latex/04_analysis.tex`, committed locally in `OverleafCultureInFigurativeLanguage/` (not pushed).

---

## 6. Deep-analysis pass (2026-10-04, user directive: "conduct whatever deep analyses you can")

The earlier §5.2 Track-2 pass delivered one table and some prose trimming, not the deeper
analysis it promised (no new figures, no new large-scale annotation). This section is the real
pass. Five new analyses, four new scripts, three new figures; all statistics are permutation
tests or item bootstraps, and the negative results are reported as negative.

### 6.1 New artefacts

| Script (`src/culture/analysis/v2/`) | Output (`docs/paper_stats/analysis_v2/`) | Needs GPU |
|---|---|---|
| `make_figures.py` | `fig_typology.pdf`, `fig_divergence.pdf`, `fig_learning.pdf` (written into the Overleaf checkout) | no |
| `analysis_to_training_bridge.py` | `analysis_to_training_bridge.json` | no |
| `selective_learning.py` | `selective_learning.json` | no |
| `exposure_dose_response.py` | `exposure_dose_response.json` | no |
| `lure_by_type.py` | `lure_by_type.json` | no |
| `culture_layer_taxonomy.py` | `culture_layer_taxonomy.json` | yes (job 397644) |
| `entity_valence.py` | `entity_valence.json` | yes (job 397644) |

`deep_analysis_remote.slurm` runs the two GPU ones (Qwen3.5-27B-FP8 primary, aya-expanse-8b as
the second annotator, both pulled from the Hub on the compute node).

### 6.2 Findings

1. **The Chinese seen/unseen gap is mostly item difficulty, not memorisation.** The *base*
   checkpoint, which never saw the corpus, already separates the IdiomAtlas-MC splits by 26.4
   points in Chinese; \idiomcpt{} widens that to only 29.0 (difference-in-differences $+2.6$).
   In Arabic and Hindi the base gaps are $+4.2$ and $-2.6$, so the \idiomcpt{} gaps are
   genuinely training-made ($+31.6$, $+24.6$). The paper's claim survives in the
   *gain-over-control* framing it actually uses, but the raw gap must not be read as
   memorisation. Written into `05_exp.tex` as a new paragraph + `fig_learning.pdf`.
2. **Corpus exposure does not modulate the gain.** Every Chinese seen item joins to a corpus
   occurrence count and no unseen item does (independent confirmation of the split). Accuracy
   rises with the count for *every* arm including the untrained base ($\rho=0.14$–$0.21$,
   $p<0.001$), so that trend is idiom frequency, not training; the gain over the control is flat
   in exposure ($\rho=-0.06$, $p=0.15$). Lesson for corpus construction: breadth over repetition.
3. **NULL — the §4 divergence does not predict §5 behaviour.** Joining per-entity embedding
   divergence to the symbolism probe (90 shared entities, continuous gold-minus-lure logprob
   margin) gives $\rho=-0.05$, $p=0.66$ in Chinese. An earlier $\rho=0.47$ was an artefact of
   ranking ties with `argsort`; fixed to average ranks. Consistent with the paper's own note
   that the embedding is insensitive to evaluative polarity.
4. **NULL — the English-default rate does not vary by semantic type.** Pooled over the three
   languages on the base checkpoint (n=224), the spread across nine types is indistinguishable
   from chance (permutation $p=0.97$). An apparent animals/food effect in a single Chinese arm
   was small-$n$ noise and is *not* reported.
5. **Section 4 rewritten around figures.** `fig_typology.pdf` (9 types × 4 languages, coloured by
   $\chi^2$ adjusted residual) replaces the paragraph that listed those percentages;
   `fig_divergence.pdf` replaces `tables/pair_divergence.tex`, which is no longer `\input` (the
   file is left on disk, now orphaned). A factual check also resolved the 515-vs-516 entity
   discrepancy: the 516th row is an empty analysis, so "all but one of the 515" is correct.

### 6.3 Still running / next

- Job **397644** (`deep-analysis`): the culture-layer taxonomy and the entity-valence analyses.
  The taxonomy is the one that matters most — it measures the paper's central but so far
  *asserted* claim that idioms carry a symbolic/evaluative layer while culture benchmarks test
  facts and practices, by classifying 4,000 items (4 idiom sets + 6 culture benchmarks) into one
  shared taxonomy. The valence analysis measures the axis the embedding divergence explicitly
  misses. Both feed `05_exp.tex` ("Which layer of culture do idioms carry?") and `04_analysis.tex`.

### 6.4 Results of the GPU analyses (jobs 397690, 397825) — 2026-10-04

**Job history.** 397644 FAILED: `local_llm` loads the primary and the second-annotator model in
one process, and vLLM cannot initialise a second engine after the first shuts down. Fixed with a
`--skip_second` flag so each model gets its own invocation (the on-disk cache makes the completed
pass free on rerun); 397690 then ran all four passes. 397825 was a follow-up that raises the
second-annotator sample from 10 to 50 items per source, which turned out to matter a lot.

**1. Culture-layer taxonomy — the paper's central claim, now measured.** Idioms and culture
benchmarks occupy almost disjoint categories, and the contrast survives the second annotator:
hi/zh/ar idioms are 58–86% symbolic-evaluative under Qwen3.5-27B and 80–96% under aya-expanse-8b,
while ArabCulture / ArabicCulturalQA / MILU / DziriEval / Alyah are 0–30% and 4–40%, with per-item
agreement 0.52–0.82 on all eight. The benchmarks are instead material practice (ArabCulture 80%)
or facts about named things (MILU 81%, ArabicCulturalQA 79%). Written into the abstract, intro,
`05_exp.tex`, and `fig_layer.pdf`.

**2. Three rows we had to withdraw.** With 50 relabelled items per source, the annotators agree on
only 12% of CCPM, 28% of English idioms and **34% of the cultural notes**. The notes result is the
painful one: the primary annotator puts 5% of them in the symbolic category (which would have
explained the \culturenotes{} null exactly), the second puts 64%. The claim is retracted in the
paper; only the notes' *form* (encyclopedia-style definitions) is asserted. The figure marks all
three rows with a dagger and prints per-row agreement.

**3. Entity valence — failed as a measurement, succeeded as a screen.** Two model families rank
entities similarly (r=0.69, n=172) but differ by 1.02 on a four-point scale, which is *larger* than
the 0.50–0.64 cross-language gaps it would need to resolve. The sign-flip statistics (1.9–4.1%) are
therefore not reported. What the screen did do was flag the dog, which led to finding 4.

**4. The paper's flagship example was wrong.** The first-pass LLM summary credited the English
*dog* with "loyalty and diligence" and the English *dragon* with "destructive force". Manual
inspection of all 80 English dog idioms shows the inventory is *dog eat dog*, *a dog's life*,
*dirty dog*, *dog's breakfast*, *work like a dog* — "man's best friend" is not an idiom and is
absent from the KB — and English *dragon* has exactly two idioms, *chase the dragon* (drug use)
and *feed the dragon* (offshoring), with no mythical sense. Both languages hold the dog in
contempt and differ only in their grounds. Corrected in the intro hook, `04_analysis.tex`, the
`entity_cases.tex` dog and dragon rows, and flagged in Limitations as stereotype leakage from
LLM-written cultural summaries. **Other summaries in §4 have not been spot-checked this way and
may carry the same problem — this is the highest-value remaining manual-verification task.**

### 6.5 State of the paper

Uncommitted (git identity is not configured on this box and I do not run `git config`):
modified `latex/{01_intro,04_analysis,05_exp}.tex`, `latex/tables/entity_cases.tex`, `main.tex`;
new and untracked `latex/figures/fig_{typology,divergence,learning,layer}.{pdf,tex}`.
`latex/tables/pair_divergence.tex` is now orphaned (superseded by `fig_divergence.pdf`) but left
on disk. Four `\todo{}`s remain, all needing a native speaker or appendix prose, none blocked on
compute.

---

## 7. Benchmark-breadth pass and interaction tests (2026-10-05)

Trigger: the paper measured Chinese culture with CCPM alone and Hindi culture with a
100-item Global-PIQA slice, against five Arabic benchmarks. The zh/hi "no transfer"
conclusions therefore rested on one benchmark each, and the Chinese one is classical-poetry
matching. This pass fixes the breadth and then tests the nulls harder.

### 7.1 Evaluation gaps closed first
Audited every (language x arm x task) cell by merging the local `eval/` tree with
`from_babel/eval/9b_babel/`: 11 of 154 were empty. All now filled.
- `hi-cpt-unfiltered-matched` (R4) had **zero** eval records; ran the full Hindi suite
  (`eval_hi_matched.slurm`). MABL 58.4 / Global-PIQA 66.0 / MILU 61.4 / IdiomAtlas seen 50.5 /
  unseen 51.7 / symbolism-v2 51.0.
- Five checkpoints lacked `symbolism_v2_*_letter` (they predate the v1->v2 task-list switch):
  ar/culture, ar/culturenotes, hi/culture, zh/untagged, zh/culture (`eval_symv2.slurm`).
  Both scripts stage to a temp dir and merge into the arm's `summary.json`, because
  `run_eval` rewrites that file with only the tasks of the invocation.

### 7.2 New culture benchmarks (zh 1 -> 3, hi 1 -> 3)
Built with `culture.evaluation.build_culture_mc` into the `jsonl:` MC format, so `run_eval`
needed no change; evaluated on all 5 zh and 6 hi 9B arms (`eval_newculture.slurm`).

| task | n | source |
|---|---|---|
| `global_piqa_zh` / `_cultural` | 648 / 237 | the `unsampled_nonparallel_cmn_hans` pool the release ships; never used before |
| `global_piqa_hi` / `_cultural` | 1,406 / 447 | same pool for Hindi; the paper used the 100-item sample (overlap 93) |
| `global_piqa_{zh,hi}_parallel4` | 103 | item-matched across zh/hi/ar (verified: identical `eng_prompt` set, row-aligned; option order is shuffled per language, and the qid carries a language suffix, so cross-language *item-level* joins need the suffix stripped -- not yet implemented) |
| `parambench_hi_culture` / `_other` | 5,449 / 5,219 | ParamBench, 100% Devanagari, Normal MCQ only, split by the dataset's own `subject` |
| `cmmlu_culture` / `cmmlu_china_specific` | 271 / 2,761 | re-aggregated from existing CMMLU records (`cmmlu_subsets.py`), no new eval |

**Rejected after inspecting the downloaded data** (all were on the recommended list):
DRISHTIKON (4,286 Hindi rows but 4,019 = 93.8% textually reference their image);
CulturalBench (59 China / 46 India items, questions in English); WenMind (917 of 4,875 rows
are MCQ, 117 in the relevant domain); SANSKRITI (released file is
`Merged_Dataset_english_SANSKRITI.csv`). CHARM still has no verifiable HF release.

### 7.3 Results: the nulls hold, and get stronger
Against the token-matched `unfiltered` control (`contrast_tasks.py`):
- Global-PIQA-zh: every arm null (culture $+0.5$ [$-1.7$, $+2.6$]).
- Global-PIQA-hi: every arm null (culture $-1.1$ [$-2.9$, $+0.7$]); the interval is now
  informative where the 100-item version's was +-8.
- ParamBench-culture: every arm null; ParamBench-**other**: cpt $-1.6$*, untagged $-1.8$*,
  culture $-2.1$* -- domain CPT costs general exam knowledge and buys no culture.
- R4 validated: `unfiltered-matched` sits 0.5-1.3 above `unfiltered` on all five new tasks,
  none significant, so the 7% token-budget gap never changed a conclusion.

### 7.4 Interaction tests (`culture.evaluation.interaction`)
A null on a culture subset cannot separate "no culture installed" from "nothing happened",
so each arm's effect was compared *between* culture-bearing and non-culture items, grouped
only by fields the dataset authors wrote. **Of twelve interactions, one survives Holm**
(culture on ParamBench, $+2.7$, $p_{adj}=0.012$) **and it is confounded**: the untrained base
shows a larger interaction in the same direction ($+3.3$), i.e. ParamBench's culture subjects
are simply less movable by any Hindi pretraining. CMMLU shows no interaction at all
($-0.6$, $p=0.38$), which **corrects an earlier over-reading**: the apparent monotone
"damage grows as the subset gets more cultural" (full $-1.8$ -> china $-2.3$ -> culture $-5.2$)
does not survive, because `culture` loses on world subjects too. Conclusion: no selective
culture effect in the reverse direction, under the most favourable test available.

### 7.5 Contamination
`contamination_check.py` (Aho-Corasick over tail shingles; 20 chars for zh, 40 for hi --
a 40-char floor would discard 545 of 648 Chinese items). Scanned 1.5M zh CPT docs + 751K zh
SFT rows, 1.7M hi CPT docs + 240K hi SFT rows. **All rates <= 0.11%**, single-digit items.
One row trips the crude >1% flag at 1/99 (Global-PIQA parallel in IndicAlign); IndicAlign is
an SFT mixture the CPT arms never see. IndicAlign stores one column per language (`hin_Deva`),
not `text` -- the first run silently scanned zero rows because of it.

### 7.6 Taxonomy: a circularity found and disclosed
The converged run reports two second annotators, and they differ a lot:
gemma-4-26B-A4B-it $\kappa=0.78$ (84% raw, 800 items, nothing below 62%) vs aya-expanse-8b
$\kappa=0.53$ (64% raw, 544 items, below 45% on CCPM 9% and English idioms 38%).
**gemma-4-26B-A4B-it generated the Arabic cultural notes** (`notes_remote.slurm:33`, job
392945), so on the cultural-notes row it is grading its own output -- which is exactly the
row where the paper had leaned on its 94% agreement as "the highest of any source". The
independent annotator there is the 8B one: 56% agreement, symbolic share 56% against the
primary's 12%. The notes claim is therefore reported as primary-only and explicitly not
corroborated. The *core* claim is unaffected and in fact stronger: idioms are 62-88% symbolic
under every annotator, culture benchmarks 0-12%.
`05_exp.tex` and `fig_layer.tex` updated accordingly.

### 7.7 Not done
- **2B (42 models)**: the checkpoints are babel-only (S3.00), unreachable from this server.
  The benchmark files and `eval_newculture.slurm` are ready to run there.
- `fig_layer.pdf` has not been regenerated; only its caption was corrected.
- Cross-language item-level joins on the `parallel4` sets (needs the language-suffix strip).
- Per-item taxonomy labels: the cache keys on prompt hash and never stores `qid`, so
  interaction tests driven by *our* labels (rather than dataset fields) still need
  `culture_layer_taxonomy.py` to emit `qid -> category`.

### 6.6 Second deep pass (2026-10-05): completing the 9B grid and three corrections

**Six missing 9B evaluations run (jobs 399424-399429, all COMPLETED).** The checkpoints were
on this cluster all along; only the IdiomAtlas-MC evaluations were missing. `eval_hi.slurm`
accepted only `base|culture`, so it was extended to the full arm set. Coverage is now ar 6 arms,
hi 5, zh 5 (hi/zh have no culturenotes checkpoint by design).

**Correction 1 — the exposure claim was wrong, and the corrected version is stronger.**
With no \idiomcpt{} arm in the data, the gain over the control looked flat in exposure, and
§5 said repetition bought nothing. With \idiomcpt{} included the arms separate sharply: the
gain grows with corpus occurrences for \idiomcpt{} in Arabic ($\rho=0.25$, $p=0.0001$) and
Hindi ($\rho=0.16$, $p=0.0003$) — Arabic climbs 54% to 100% across the exposure range — while
\idiomdocs{}, \culturecpt{} and \culturenotes{} are flat everywhere ($|\rho|\leq0.08$, n.s.).
**Repetition pays only when the meaning is stated alongside the idiom.** Chinese \idiomcpt{} is
flat too, consistent with a base model that already knows its chengyu.

**Correction 2 — the seen/unseen difference-in-differences now has intervals and one source.**
Previously the \idiomcpt{} bars came from the other cluster's point estimates. All four arms
are now evaluated here with the same harness: ar $+30.5$ $[24.3, 36.7]$, hi $+23.0$
$[16.3, 29.7]$, zh $+4.0$ $[-4.6, 12.1]$ (interval includes zero). The local numbers track the
published ones to within 1.6 points, which cross-validates the two clusters.

**Correction 3 — the cultural-notes claim is restored.** It had been retracted because the
8B second annotator split 0.08 vs 0.50 on the notes' symbolic share. A same-size different-family
annotator (gemma-4-26B-A4B-it) agrees with the primary on **94%** of notes and puts the symbolic
share at 4%, the best agreement of any source. Overall taxonomy agreement rose from $\kappa=0.53$
(aya-8b) to $\kappa=0.78$ (gemma), and no source now falls below 0.62 — including CCPM, which
the weak annotator had put at 0.18.

**Annotator ladder.** The same pattern holds at every level: agreement tracks annotator strength,
not category ambiguity. Nine types: aya-8b 0.537 < Qwen3.5-9B 0.695 < nemotron-550B 0.805.
Seven subtypes: aya-8b 0.432 < Qwen3.5-9B 0.601 < gemma-26B 0.669. Taxonomy: aya-8b 0.53 <
gemma-26B 0.78. Low $\kappa$ from a small annotator is evidence about the annotator.

**A validation gap found and under test.** The 9-way and 7-way validations are both *within
level*: the subtypology prompt tells the annotator the entity is "already judged abstract...
not an animal, body part, natural feature, food, person, deity, artefact or sum of money", so a
first-level error can never be routed back and cross-level confusion is invisible by
construction. The two codebooks overlap in writing (`occupation_economy` "work and labour" vs
`action_conflict` "work and effort as abstractions"; `nature_cosmos` "seasons and the day-night
cycle" vs `time_change` "day as a unit, year"). `typology_flat_validate.py` (job 399489) puts
all fifteen labels in one flat codebook with no hint of the two levels and reports $\kappa$ plus
the share of disagreement that is cross-level.

**vLLM operational note.** gemma-4-26B-A4B-it **hangs indefinitely at TP=2** on this cluster:
49GB in 2 shards, zero progress for 42 minutes, no error, both log files frozen. At **TP=1** the
same weights load in 2m49s. All gemma jobs now use `VLLM_TP=1`.

### 6.7 What remains

| Item | Status |
|---|---|
| 2B models on the new benchmarks | **Hard-blocked.** No 2B checkpoint exists on this cluster (confirmed by listing `ckpts/`); plan §3.00 records them as babel-only. Needs either a 2B upload to HF or a babel run. The benchmark files (`global_piqa_zh*`, `parambench_hi_*`) *are* present here. |
| HF upload of the 9 new 9B checkpoints | **Needs the user** — sandbox cannot reach huggingface.co (§5.3). |
| Flat 15-way validation | job 399489 queued |
| Native-speaker validation (3 `\todo{}`) | Needs a native speaker; sheet prepared |
| Appendix additions (1 `\todo{}`) | Writing, not compute |
| Spot-check of the remaining §4 entity summaries | **Highest-value unverified risk.** The dog and dragon summaries both carried stereotype leakage; the other 513 have not been checked the same way. |

### 6.8 Entity-summary audit and the flat 15-way validation (2026-10-06)

**The §4 entity summaries carry a 17.6% unsupported-claim rate.** `entity_summary_audit.py`
(job 402050) takes all 4,703 claims the summaries make across 317 entities -- every primary
meaning and every language-unique aspect -- shows each one to gemma-4-26B-A4B-it together with
that entity's idioms in that language, and asks whether any idiom supports it, with instructions
to answer no when unsure. Supported: English primary 85.7%, English unique 83.9%, Chinese
primary 79.7%, Chinese unique 80.1%; **17.6% overall unsupported**. The judge resolves
uncertainty against the summary, so that is an upper bound, but it is far too large for the dog
and dragon cases to be isolated. Written into `04_analysis.tex` and Limitations; per-claim
verdicts are in `entity_summary_audit.json`. (First submission failed in 12s: some
`figurative_meanings` are nested one list deeper; fixed with `common.flatten`.)

**The 15-way scheme is weaker than its per-level validations suggest.** `typology_flat_validate.py`
(job 399489) puts all fifteen labels in one flat codebook with no hint of the two-level
structure: **kappa=0.667** (raw 0.689, n=360; en .643 zh .702 hi .714 ar .607), against 0.805
for the 9-way choice. **48% of disagreements are cross-level** (54 of 112) -- exactly the class
of error both existing validations are blind to, since the subtypology prompt tells the
annotator the entity is already known to be abstract. The leaks match what the two codebooks
overlap on in writing: Morality->Body (3), Nature->Time (2), Nature->Action (2),
Unclassified->Body (3). This is the reason `fig_typology_expanded` belongs in the appendix
rather than the main text.

**A circularity we introduced and then removed.** The cultural-notes finding was briefly
restored on the strength of gemma-4-26B-A4B-it agreeing with the primary annotator on 94% of
notes -- but gemma *wrote* those notes (`notes_remote.slurm`), so it was grading its own output.
The only independent check is the 8B model, which agrees on 56% with a 44-point gap on the
symbolic share. The paper now reports the notes' content as the primary annotator sees it and
explicitly declines to claim corroboration; the circularity is flagged in Limitations.

**Benchmark release script.** `scripts/upload_benchmarks_hf.sh` stages IdiomAtlas-MC (9 files)
and the symbolism probe (12 files) with a generated dataset card and uploads them;
`--push` to actually upload, dry run by default. It deliberately excludes `global_piqa_*` and
`parambench_hi_*` (third-party, only reformatted here) and IdiomCulture (design + code only, no
items generated yet).

## 8. 2B on the new culture benchmarks (babel, 2026-10-06)

Closes the 2B row of §6.7 / §7.7. All 24 Qwen3.5-2B zh/hi models (12 per language:
`{,i_}{base,random,idiom_tagged,idiom_untagged,culture,culture_notes}`) evaluated on the §7.2
tasks with the same settings as `eval_newculture.slurm` (0-shot log-likelihood, `run_eval`).

**Data.** Raw files from HF (`mrlbenchmarks/global-piqa-nonparallel`
`unsampled_full/unsampled_nonparallel_{cmn_hans,hin_deva}.tsv`, `mrlbenchmarks/global-piqa-parallel`
`data/parallel_*.tsv`, `bharatgenai/ParamBench` `ParamBench.parquet`) under
`bidir/eval_data/{zh,hi}/global_piqa/` and `bidir/eval_data/hi/parambench/`. Built with
`build_culture_mc --eval_dir <bidir/eval_data> --parambench <...>/ParamBench.parquet` (new flags,
defaults unchanged) into `bidir/eval_data/mc/`. Counts match the 9B pass exactly:
global_piqa_zh 648 (cultural 237), global_piqa_hi 1,406 (cultural 447), parallel4 103 each,
parambench_hi_culture 5,449, parambench_hi_other 5,219.

**Jobs.** `src/culture/bidirectional/eval_newculture_2b.slurm`, array **10675310** (0-23, %8,
preempt/preempt_qos, --requeue, BS=4): 24/24 COMPLETED (tasks 20, 22, 23 were preempted and
requeued; the script is idempotent). Outputs merged into `eval2b/{zh,hi}/<run>/summary.json`
(pre-run copies in `eval2b/_summary_backup_20261006/`). Note: the `summary.json` of
`{zh,hi}/{i_base,base,random}` and `zh/{idiom_tagged,idiom_untagged}` already held only
`symbolism_v2_*` before this run (clobbered by an earlier invocation); the per-task json files are
intact, so nothing is lost, but those summaries are incomplete.

**Analysis.** `contrast_tasks.py` / `interaction.py` gained `--eval_root` (9B defaults unchanged).
Main = Qwen3.5-2B arms vs `i_random`; Base-ablation = Qwen3.5-2B-Base arms vs `random`.
Outputs: `docs/paper_stats/v2/newculture_2b_{zh,hi}{,_base}.json`,
`interaction_2b_{gpiqa_zh,gpiqa_hi,parambench_hi,cmmlu_zh}{,_base}.json`. CMMLU subsets for 2B
were already in `cmmlu_subsets_2b.json` (checked, not redone); the CMMLU interaction uses its 16
china-specific subjects (n=2,761 vs 8,821), as at 9B.

Delta vs control, accuracy points, 95% paired-bootstrap CI (* McNemar p<0.05):

| task (n) | idiom_tagged | idiom_untagged | culture | culture_notes | untrained |
|---|---|---|---|---|---|
| **Main (vs i_random)** | | | | | |
| gpiqa_zh (648) | +0.5 [-1.5,+2.5] | +0.0 [-2.0,+2.0] | -1.1 [-2.9,+0.8] | -0.5 [-2.3,+1.4] | -5.2* |
| gpiqa_zh_cultural (237) | +2.1 [-1.3,+5.5] | +1.3 [-2.1,+4.6] | -0.4 [-3.0,+2.1] | +0.4 [-3.0,+3.8] | -1.3 |
| gpiqa_zh_parallel4 (103) | -4.9 [-10.7,+0.0] | -3.9 [-9.7,+1.0] | -1.9 [-7.8,+3.9] | +1.9 [-2.9,+7.8] | -9.7* |
| gpiqa_hi (1,406) | -0.3 [-2.0,+1.4] | +0.2 [-1.5,+2.0] | -1.4 [-3.4,+0.6] | -1.1 [-3.1,+0.9] | -3.6* |
| gpiqa_hi_cultural (447) | -1.1 [-4.3,+2.0] | -1.3 [-4.7,+1.8] | +0.7 [-3.1,+4.5] | +2.9 [-0.9,+6.7] | +0.2 |
| gpiqa_hi_parallel4 (103) | -2.9 [-9.7,+2.9] | +1.0 [-4.9,+6.8] | -7.8 [-14.6,-1.0] | -2.9 [-9.7,+3.9] | -14.6* |
| parambench_hi_culture (5,449) | +0.5 [-0.5,+1.5] | +1.4* [+0.3,+2.5] | -0.4 [-1.5,+0.7] | +0.6 [-0.6,+1.7] | -0.2 |
| parambench_hi_other (5,219) | -1.1* [-2.2,-0.1] | +0.4 [-0.8,+1.5] | **-4.9*** [-6.1,-3.6] | **-3.9*** [-5.1,-2.7] | -1.6* |
| **Base ablation (vs random)** | | | | | |
| gpiqa_zh (648) | +0.8 [-1.2,+2.8] | +0.3 [-1.7,+2.5] | +0.0 [-1.9,+1.9] | +0.2 [-1.9,+2.0] | +3.2* |
| gpiqa_zh_cultural (237) | +0.4 [-3.0,+3.8] | -1.7 [-4.6,+1.3] | +0.0 [-2.5,+2.5] | -0.4 [-3.4,+2.5] | +2.1 |
| gpiqa_zh_parallel4 (103) | -1.9 [-7.8,+2.9] | -2.9 [-8.7,+2.9] | -4.9 [-9.7,+0.0] | -2.9 [-6.8,+1.0] | +1.9 |
| gpiqa_hi (1,406) | -0.1 [-1.9,+1.5] | -0.3 [-2.1,+1.5] | -1.1 [-3.1,+0.9] | -0.9 [-2.8,+1.1] | -4.4* |
| gpiqa_hi_cultural (447) | +1.1 [-1.8,+4.0] | +0.4 [-2.7,+3.8] | +2.2 [-1.3,+6.0] | +4.3* [+0.9,+7.8] | +1.3 |
| gpiqa_hi_parallel4 (103) | +1.0 [-4.9,+6.8] | +2.9 [-3.9,+9.7] | -2.9 [-9.7,+3.9] | +1.9 [-5.8,+9.7] | -9.7 |
| parambench_hi_culture (5,449) | -0.9 [-2.0,+0.1] | -0.3 [-1.4,+0.7] | -2.0* [-3.3,-0.8] | -0.7 [-1.9,+0.4] | -1.9* |
| parambench_hi_other (5,219) | -1.9* [-3.0,-0.8] | -1.5* [-2.6,-0.3] | **-4.3*** [-5.6,-2.9] | **-4.5*** [-5.7,-3.2] | -3.6* |

(gpiqa_hi_parallel4 i_culture -7.8 has a CI excluding 0 but McNemar p=0.057.)

**Interactions** (culture items minus other items; Holm over 4 tests x 4 trained arms = 16 per
family; bootstrap p floored at 1e-4):

| test | arm | Δculture | Δother | interaction [95% CI] | p | Holm |
|---|---|---|---|---|---|---|
| ParamBench | i_culture | -0.40 | -4.87 | +4.46 [+2.75,+6.13] | <1e-4 | **0.002** |
| ParamBench | i_culture_notes | +0.59 | -3.89 | +4.48 [+2.79,+6.17] | <1e-4 | **0.002** |
| ParamBench | i_idiom_tagged | +0.51 | -1.09 | +1.61 [+0.15,+3.09] | 0.029 | 0.38 |
| GPIQA-hi | i_culture_notes | +3.13 | -3.13 | +6.26 [+1.80,+10.66] | 0.007 | 0.09 |
| GPIQA-hi | i_culture | +0.89 | -2.50 | +3.40 [-1.09,+7.91] | 0.14 | 1 |
| GPIQA-zh | all i_* arms | | | within [-0.96, +0.60] | >=0.62 | 1 |
| CMMLU-zh | all i_* arms | | | within [-0.66, +1.18] | >=0.08 | >=0.94 |
| *ref.* ParamBench | i_base (untrained) | -0.22 | -1.57 | +1.35 [-0.77,+3.49] | 0.22 | -- |
| *ref.* GPIQA-hi | i_base (untrained) | +0.89 | -5.63 | +6.53 [-0.46,+13.40] | 0.065 | -- |
| Base: ParamBench | culture_notes | -0.73 | -4.45 | +3.71 [+2.00,+5.45] | 2e-4 | **0.003** |
| Base: GPIQA-hi | culture_notes | +4.25 | -3.34 | +7.59 [+3.43,+11.85] | 6e-4 | **0.009** |
| Base: ParamBench | culture | -2.04 | -4.27 | +2.24 [+0.42,+4.02] | 0.014 | 0.20 |
| Base: GPIQA-hi | culture | +2.24 | -2.71 | +4.95 [+0.67,+9.32] | 0.021 | 0.28 |
| *ref.* Base: GPIQA-hi | base (untrained) | +1.57 | -7.19 | +8.76 [+1.88,+15.68] | 0.012 | -- |
| *ref.* Base: ParamBench | base (untrained) | -1.93 | -3.56 | +1.64 [-0.31,+3.53] | 0.10 | -- |

**Does the 9B conclusion replicate at 2B?**
- *No selective culture gain in the reverse direction*: **yes**. On the main Qwen3.5-2B arms no
  culture subset improves significantly for the culture arms (the only significant positive
  culture-subset delta in the main family is idiom_untagged on ParamBench-culture, +1.4). Global-PIQA-zh
  and CMMLU-zh are flat for every arm, as at 9B.
- *ParamBench-other losses*: **yes, and larger**. culture -4.9 / culture_notes -3.9 (main) and
  -4.3 / -4.5 (Base) vs -2.1 for 9B culture; idiom_tagged also loses (-1.1 / -1.9).
- *Interactions*: more survive Holm at 2B (2 of 16 in each family vs 1 of 12 at 9B), all in the
  culture arms and all positive. Read with care: (i) on ParamBench every surviving interaction is
  driven by the **other** half collapsing while the culture half stays at the control (Δculture CIs
  include 0), i.e. culture-domain CPT damages general exam knowledge less on culture subjects; it
  does not add culture knowledge. (ii) Unlike 9B, the untrained-model confound does not explain the
  ParamBench interaction at 2B (i_base +1.35 n.s. vs +4.5). (iii) On Global-PIQA-hi the untrained
  model shows an interaction as large as or larger than the arms (+6.5 / +8.8), the same confound
  seen at 9B, so the culture_notes GPIQA-hi interaction (Base family, +7.6, Holm 0.009; its
  cultural-subset delta +4.3, unadjusted p=0.025) cannot be attributed to culture content.
  Overall: same conclusion as 9B, with a stronger "domain CPT costs general knowledge" effect at 2B.
