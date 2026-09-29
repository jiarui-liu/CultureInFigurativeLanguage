#!/bin/bash
# Symbolism probe: generation (Qwen3.5-27B-FP8) then blind verification (Gemma-4-26B-A4B-it).
#   sbatch --export=ALL,LANGS="zh hi ar" probe.slurm
#SBATCH --job-name=probe
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --exclude=babel-o5-28,babel-q5-32,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/probe-%j.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
S=$REPO/src/culture/bidirectional/stage_model.sh
OUTD=/data/group_data/r3lit_culture_pretrain/culture/bidir/eval_data/mc
AUD=/data/group_data/r3lit_culture_pretrain/culture/bidir/audit
STAGES=${STAGES:-"generate verify"}
if [[ $STAGES == *generate* ]]; then
export VLLM_TP=1   # Qwen3.5-27B-FP8 fits one 48GB GPU; avoids tensor-parallel hangs
export GEN_MODEL_PATH=$(python $REPO/src/culture/bidirectional/stage_hf.py Qwen/Qwen3.5-27B-FP8 | tail -1)
for l in ${LANGS:-zh hi ar}; do
  python -m culture.bidirectional.build_symbolism_probe --lang $l --stage generate --out_dir $OUTD --version "${VERSION:-}" --top_k ${TOPK:-400}
done
python -m culture.bidirectional.kb_audit judge --judge_name qwen27b --model $GEN_MODEL_PATH --out_dir $AUD || true
fi
if [[ $STAGES == *verify* ]]; then
export VLLM_TP=1 LLM_QUANT=fp8   # Gemma-4-26B-A4B quantized online to FP8 (~27GB) on one GPU: avoids TP hangs
export VERIFY_MODEL_PATH=$(python $REPO/src/culture/bidirectional/stage_hf.py google/gemma-4-26B-A4B-it | tail -1)
for l in ${LANGS:-zh hi ar}; do
  python -m culture.bidirectional.build_symbolism_probe --lang $l --stage verify --out_dir $OUTD --version "${VERSION:-}" --top_k ${TOPK:-400}
done
python -m culture.bidirectional.kb_audit judge --judge_name gemma --model $VERIFY_MODEL_PATH --out_dir $AUD || true
python -m culture.bidirectional.kb_audit summarize --out_dir $AUD || true
fi
