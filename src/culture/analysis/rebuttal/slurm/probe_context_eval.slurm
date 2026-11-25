#!/bin/bash
# Score one 9B checkpoint on the symbolism probe without and with idiom context (one GPU, ~15 min).
#   sbatch --export=ALL,LANG_=ar,RUN=cpt,MODEL=<hf spec> probe_context_eval.slurm
#SBATCH --job-name=probectx
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-o5-28,babel-q5-32,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/probectx-%j.out
set -euo pipefail
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
source $REPO/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface PYTHONPATH=$REPO/src
P=/data/group_data/r3lit_culture_pretrain/culture/bidir/eval_data/mc
OUT=/data/group_data/r3lit_culture_pretrain/culture/bidir/eval9b_probectx/$LANG_/$RUN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOCAL=$(python $REPO/src/culture/bidirectional/stage_hf.py "$MODEL" | tail -1)
T=${TASKS:-jsonl:symbolism_v2_${LANG_}_letter,jsonl:symbolism_ctxB_${LANG_}_letter,jsonl:symbolism_ctxC_${LANG_}_letter,jsonl:symbolism_ctxD_${LANG_}_letter}
python -m culture.evaluation.run_eval --model_path "$LOCAL" --output_dir "$OUT" --tasks "$T" --run_name $RUN \
  --batch_size ${BS:-8} --seed 42 --mc_jsonl_dir $P --num_fewshot 0
