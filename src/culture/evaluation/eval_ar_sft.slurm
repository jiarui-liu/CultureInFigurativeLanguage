#!/bin/bash
# Arabic CPT-eval pipeline applied to the SFT checkpoints (SAME tasks/harness as
# eval_ar.slurm). sbatch eval_ar_sft.slurm ar-cpt-sft ; sbatch eval_ar_sft.slurm ar-base-sft
#SBATCH --job-name=ar-eval-sft
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-ar-%x-%j.out
#SBATCH --error=logs/eval-ar-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_ar_sft.slurm <ar-cpt-sft|ar-base-sft>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "$RUN" in
  ar-cpt-sft)  M="${CK}/qwen3p5-9b-ar-cpt-sft" ;;
  ar-base-sft) M="${CK}/qwen3p5-9b-ar-base-sft" ;;
  *) echo bad RUN; exit 1;;
esac
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/ar
KB=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/ar-kb/data/idioms/ar/idioms_merged_llm_formatted.jsonl
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar/$RUN
mkdir -p "$OUT"
for probe in ar_fineweb2_heldout ar_wiki_heldout; do
  python -m culture.evaluation.perplexity \
    --model_path "$M" --run_name "$RUN" \
    --data_path "$DATA/$probe.jsonl" --text_field text \
    --output_dir "$OUT/ppl_${probe#ar_}" || echo "[warn] $probe failed"
done
python -m culture.evaluation.run_eval \
  --model_path "$M" --run_name "$RUN" \
  --tasks kinayat_cloze,kinayat_meaning,ar_figurative,arabculture,arabic_cultural_qa,arabicmmlu,global_piqa_ar,alyah,dzirieval,global_piqa_ar_parallel \
  --ar_data_dir "$DATA/raw" \
  --ar_kb_path "$KB" \
  --acqa_dialects msa \
  --ar_num_fewshot 0 \
  --batch_size 8 \
  --output_dir "$OUT"
echo "DONE $RUN -> $OUT"
