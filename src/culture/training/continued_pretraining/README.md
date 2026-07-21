# Continued pretraining (Qwen 3.5 9B)

Full-parameter continued pretraining of **Qwen 3.5 9B (base)** on the project's
cultural-knowledge corpora, via LLaMA-Factory's `pt` stage + DeepSpeed ZeRO-3,
launched on 4 × 8×H100 nodes with Slurm.

Two corpora, run as **separate parallel jobs**:

| Lang | Corpus (HF) | Docs | ~Tokens | Config | Launcher |
|---|---|---|---|---|---|
| Hindi | `jiviteshjn/hi-proverbs-cpt` | 338K | ~1.4B | `configs/qwen3p5_9b_cpt.yaml` | `cpt.slurm` |
| Chinese | `jiviteshjn/fineweb-edu-zh-chengyu-cpt` | 3.74M | ~7.8B | `configs/qwen3p5_9b_cpt_zh.yaml` | `cpt_zh.slurm` |

## Files
| File | Purpose |
|---|---|
| `prepare_data.py` | Decompress/reshard `tagged_*.json.gz` → plain `train_*.jsonl` (shuffle, no encryption). |
| `configs/qwen3p5_9b_cpt.yaml` / `_zh.yaml` | LLaMA-Factory `pt`-stage training configs. |
| `configs/dataset_info.json` | Registers `hi_proverbs` and `zh_chengyu` (→ absolute lustre train dirs). |
| `configs/ds_z3.json` / `ds_z2.json` | DeepSpeed ZeRO-3 (used) / ZeRO-2 (alternative). |
| `env.sh` | Shared distributed env (NCCL/IB). Sourced by the launchers. |
| `cpt.slurm` / `cpt_zh.slurm` | Slurm batch launchers (4 nodes each). |

## Paths (this cluster)
- **Data root:** `/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data`
  - downloads: `hi-proverbs-cpt/`, `fineweb-edu-zh-chengyu-cpt/`
  - prepared shards: `train/` (Hindi), `train_zh/` (Chinese)
- **Model:** `/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B` (base)
- **Checkpoints:** `/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-{hi,zh}-cpt`
- **venv:** reuses `monitorability-prertaining/.venv` (working `llamafactory-cli`
  + transformers 5.6 + deepspeed). Configs/launchers hardcode absolute paths.

## Run
```bash
# 1. Download a corpus (idempotent; re-run to verify completeness):
hf download jiviteshjn/hi-proverbs-cpt --repo-type dataset \
  --local-dir /lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/hi-proverbs-cpt

# 2. Prepare plain JSONL shards:
python3 prepare_data.py                                    # Hindi -> <DATA_ROOT>/train
python3 prepare_data.py \
  --src_dir /lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/fineweb-edu-zh-chengyu-cpt/data \
  --out_dir /lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/train_zh   # Chinese
python3 prepare_data.py --verify_only --out_dir <DATA_ROOT>/train   # sanity check + doc count

# 3. Launch (from anywhere; --chdir pins CWD to this dir, logs land in ./logs):
sbatch cpt.slurm       # Hindi
sbatch cpt_zh.slurm    # Chinese (parallel, separate job)
```

## Recipe (recap)
`pt` stage · full-parameter · 16K context · packing · bf16 · ZeRO-3 ·
global batch 1×4×32 = 128 seq ≈ **2.1M tokens/step** · LR 1e-5 cosine w/ 3%
warmup · AdamW β=(0.9,0.999), wd 0.01 · grad clip 1.0 · 3 epochs · seed 42.
Optimizer hyperparameters follow OdysSim (arXiv:2606.14199) midtraining.

## Gotchas (learned the hard way)
- **`flash_attn: sdpa`, NOT `fa2`.** transformers 5.6's FA2 path crashes on
  Qwen3.5's optional attention-sink `s_aux` (passed as `None`) →
  `AttributeError` at training step 1.
- **`.json.gz` is not directly loadable** by LLaMA-Factory (its loader rejects
  the double extension) — hence `prepare_data.py` decompresses first.
- **Verify the download** before trusting it: `hf download` can leave orphan
  `.incomplete` files; re-run it (idempotent) and confirm the prepared doc count
  matches the dataset README before training.
- **Auto-resume:** `overwrite_output_dir: false` + no hardcoded
  `resume_from_checkpoint` → a requeued job resumes from the newest checkpoint.
