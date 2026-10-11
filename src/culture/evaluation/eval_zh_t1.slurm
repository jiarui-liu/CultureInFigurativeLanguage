#!/bin/bash
# Evaluate one T1 arm (the Chinese idiom-free-restriction experiment) on the Chinese suite,
# the Chinese perplexity probes and English retention.
#   sbatch eval_zh_t1.slurm <random|culture_free|culture_all> [step|final]
#SBATCH --job-name=zh-t1-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --output=logs/zh-t1-eval-%x-%j.out
#SBATCH --error=logs/zh-t1-eval-%x-%j.err
set -uo pipefail
ARM="${1:?usage: sbatch eval_zh_t1.slurm <random|culture_free|culture_all> [step|final]}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
STEP="${2:-final}"
D="$CK/qwen3p5-9b-zh-t1-${ARM//_/-}"
if [ "$STEP" = final ]; then M="$D"; else M="$D/checkpoint-$STEP"; fi
[ -f "$M/config.json" ] || { echo "no checkpoint at $M"; exit 1; }
# intermediate checkpoints carry no tokenizer; take it from the base model
BASE=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json chat_template.jinja; do
  [ -f "$BASE/$f" ] && [ ! -f "$M/$f" ] && cp "$BASE/$f" "$M/" 2>/dev/null
done
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/zh
MC=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval/mc
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/zh/t1_${ARM}_${STEP}
mkdir -p "$OUT/en"

echo "### Chinese perplexity probes"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "t1_${ARM}_${STEP}" \
  --data_path "$DATA/zh_wiki_heldout.jsonl" --text_field text --output_dir "$OUT/ppl_zh_wiki"
python -m culture.evaluation.perplexity --model_path "$M" --run_name "t1_${ARM}_${STEP}" \
  --data_path "$DATA/zh_chengyu_heldout.jsonl" --text_field text --output_dir "$OUT/ppl_zh_chengyu"

echo "### Chinese benchmarks"
python -m culture.evaluation.run_eval --model_path "$M" --run_name "t1_${ARM}_${STEP}" \
  --tasks chid,chengyu_bench,cmmlu,ccpm,chengyu_bench_app,jsonl:idiomatlas_mc_zh_seen,jsonl:idiomatlas_mc_zh_unseen,jsonl:symbolism_v2_zh_letter,jsonl:global_piqa_zh,jsonl:global_piqa_zh_cultural \
  --chid_path "$DATA/chid_eval3000.jsonl" --chid_answer_path "" \
  --chengyu_bench_dir "$DATA/ChengyuBench" --chengyu_bench_subtask connotation \
  --ccpm_path "$DATA/CCPM/valid.jsonl" \
  --cmmlu_dir "$DATA/cmmlu" --cmmlu_subjects all \
  --mc_jsonl_dir "$MC" --num_fewshot 0 --cmmlu_num_fewshot 5 --batch_size 8 \
  --output_dir "$OUT"

echo "### English retention"
python -m culture.evaluation.run_eval --model_path "$M" --run_name "t1_${ARM}_${STEP}" \
  --tasks mmlu,boolq --num_fewshot 0 --batch_size 8 --no_judge --output_dir "$OUT/en" \
  || echo "[warn] mmlu/boolq"
echo "T1_EVAL_DONE $ARM"
