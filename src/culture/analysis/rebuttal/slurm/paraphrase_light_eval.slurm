#!/bin/bash
# Score one 9B checkpoint on the lightly reworded IdiomAtlas-MC seen split (one GPU, ~10 min).
#   sbatch --export=ALL,LANG_=ar,RUN=cpt,MODEL=<hf spec> paraphrase_light_eval.slurm
#SBATCH --job-name=para2eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --exclude=babel-o5-28,babel-q5-32,babel-x9-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/para2eval-%j.out
set -euo pipefail
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
source $REPO/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface PYTHONPATH=$REPO/src
P=/data/group_data/r3lit_culture_pretrain/culture/bidir/eval_data/mc_para2
OUT=/data/group_data/r3lit_culture_pretrain/culture/bidir/eval9b_para2/$LANG_/$RUN
LOCAL=$(python $REPO/src/culture/bidirectional/stage_hf.py "$MODEL" | tail -1)
T=jsonl:idiomatlas_mc_${LANG_}_seen_para2
python -m culture.evaluation.run_eval --model_path "$LOCAL" --output_dir "$OUT" --tasks "$T" --run_name $RUN \
  --batch_size 16 --seed 42 --mc_jsonl_dir $P --num_fewshot 0
