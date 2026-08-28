#!/bin/bash
# FULL Arabic eval for ONE instruction-tuned checkpoint, covering EVERY Arabic
# benchmark in the paper, in one GPU job. New run name -> new output folder
# (existing folders untouched).
#   sbatch eval_ar_sft_full.slurm <ar-cpt-cidar-sft|ar-base-cidar-sft|ar-cpt-sft|ar-base-sft>
# Covers: LM probes (fineweb2/wiki/amthal in-domain + WikiText English) ; idiom/cultural
# MC (loglik) ; English MC MMLU/BoolQ (loglik) ; English generation GSM8K/HumanEval (chat).
#SBATCH --job-name=ar-eval-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ar/eval-arfull-%x-%j.out
#SBATCH --error=logs/ar/eval-arfull-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_ar_sft_full.slurm <run>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs/ar
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
# ONLINE (mmlu/boolq/wikitext/gsm8k/humaneval pull from HF on the node); idiom data is local.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  ar-cpt-native-sft)  M="$CK/qwen3p5-9b-ar-cpt-native-sft" ;;
  ar-base-native-sft) M="$CK/qwen3p5-9b-ar-base-native-sft" ;;
  ar-cpt-sft)        M="$CK/qwen3p5-9b-ar-cpt-sft" ;;
  ar-base-sft)       M="$CK/qwen3p5-9b-ar-base-sft" ;;
  *) echo "bad RUN=$RUN"; exit 1;;
esac
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/ar
KB=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/ar-kb/data/idioms/ar/idioms_merged_llm_formatted.jsonl
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar/$RUN
mkdir -p "$OUT" "$OUT/en" "$OUT/en_gen_chat"
echo "RUN=$RUN MODEL=$M"

# --- (2) Arabic LM probes: fineweb2/wiki held-out + amthal in-domain (PPL/BPB) ---
for probe in ar_fineweb2_heldout ar_wiki_heldout; do
  python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
    --data_path "$DATA/$probe.jsonl" --text_field text \
    --output_dir "$OUT/ppl_${probe#ar_}" || echo "[warn] $probe failed"
done
# amthal in-domain probe: file is ar_amthal_heldout.jsonl; keep output dir ppl_ar_amthal
# (matches the convention used by the other runs' backfill).
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --data_path "$DATA/ar_amthal_heldout.jsonl" --text_field text \
  --output_dir "$OUT/ppl_ar_amthal" || echo "[warn] amthal failed"
# --- (1) English LM retention: WikiText-103 ---
python -m culture.evaluation.perplexity --model_path "$M" --run_name "$RUN" \
  --hf_dataset Salesforce/wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
  --output_dir "$OUT/en/ppl_wikitext" || echo "[warn] wikitext failed"

# --- (3,4) Arabic idiom/cultural MC (base-model loglik) ---
python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
  --tasks kinayat_cloze,kinayat_meaning,ar_figurative,arabculture,arabic_cultural_qa,arabicmmlu,global_piqa_ar,alyah,dzirieval,global_piqa_ar_parallel \
  --ar_data_dir "$DATA/raw" --ar_kb_path "$KB" --acqa_dialects msa \
  --ar_num_fewshot 0 --batch_size 8 --output_dir "$OUT" || echo "[warn] ar MC failed"

# --- (1) English MC retention: MMLU/BoolQ (loglik) ---
python -m culture.evaluation.run_eval --model_path "$M" --run_name "$RUN" \
  --tasks mmlu,boolq --num_fewshot 0 --batch_size 8 --no_judge \
  --output_dir "$OUT/en" || echo "[warn] mmlu/boolq failed"

# --- (1) English generation retention: GSM8K/HumanEval (native chat template) ---
python -m culture.evaluation.run_english_gen --model_path "$M" --run_name "$RUN" \
  --tasks gsm8k,humaneval --gen_batch_size 16 --chat \
  --output_dir "$OUT/en_gen_chat" || echo "[warn] gsm8k/humaneval chat failed"

echo "AR_EVAL_FULL_DONE $RUN"
