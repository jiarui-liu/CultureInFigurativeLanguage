#!/bin/bash
# Evaluate one checkpoint on a comma-separated task list with the shared harness.
#   sbatch --export=ALL,MODEL=<dir|hf spec>,OUT=<dir>,TASKS=<t1+t2+...>[,BS=16] eval.slurm
# (tasks are '+'-separated because sbatch --export splits on commas)
# Task groups (see bidirectional_transfer_plan.md): jsonl:<name> tasks read $E/mc/<name>.jsonl.
#SBATCH --job-name=bidir-eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/eval-%j.out
set -euo pipefail
TASKS=${TASKS//+/,}
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage; export PYTHONPATH=$REPO/src
E=/data/group_data/r3lit_culture_pretrain/culture/bidir/eval_data
H=/data/group_data/r3lit_culture_pretrain/culture/bidir/hf9b/data/eval
if [[ "$MODEL" == /* ]]; then LOCAL=$($REPO/src/culture/bidirectional/stage_model.sh "$MODEL")
else LOCAL=$(python $REPO/src/culture/bidirectional/stage_hf.py "$MODEL" | tail -1); fi
python -m culture.evaluation.run_eval --model_path "$LOCAL" --output_dir "$OUT" --tasks "$TASKS" \
  --run_name "$(basename $OUT)" --batch_size ${BS:-16} --seed 42 \
  --mc_jsonl_dir $E/mc \
  --ar_data_dir $E/ar/raw \
  --mabl_path $E/hi/mabl_hi.csv --global_piqa_path $E/hi/global_piqa_hi.tsv \
  --milu_path $H/hi/milu_hi_test.jsonl --milu_fewshot_path $H/hi/milu_hi_val.jsonl --milu_num_fewshot 5 \
  --chid_path $H/zh/chid_eval3000.jsonl --chid_answer_path "" \
  --chengyu_bench_dir $E/zh/ChengyuBench --chengyu_bench_subtask connotation \
  --ccpm_path $E/zh/CCPM/valid.jsonl --cmmlu_dir $E/zh/cmmlu --cmmlu_subjects all --cmmlu_num_fewshot 5 \
  --num_fewshot 0
