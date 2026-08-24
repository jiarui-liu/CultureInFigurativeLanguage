#!/bin/bash
# Arabic eval for ONE checkpoint: Dim-2 BPB probes + Dim-3/4 multiple-choice.
#   sbatch eval_ar.slurm base   ;   sbatch eval_ar.slurm cpt
#SBATCH --job-name=ar-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-ar-%x-%j.out
#SBATCH --error=logs/eval-ar-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_ar.slurm <base|cpt|unfiltered>}"
CIFL=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
cd "$CIFL"; mkdir -p logs
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:$PATH"
export PYTHONPATH="$CIFL/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

BASE_MODEL=/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B
CPT_MODEL=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-ar-cpt
case "$RUN" in
  base) M="$BASE_MODEL";;
  cpt)  M="$CPT_MODEL";;
  unfiltered) M="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-ar-cpt-unfiltered";;
  untagged) M="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-ar-cpt-untagged";;
  *) echo bad RUN; exit 1;;
esac

# Everything under DATA is produced offline by download_ar.sh + build_ar_probes.py
# (HF_HUB_OFFLINE=1 here, so nothing may hit the network at eval time).
DATA=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/data/eval/ar
KB=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/ar-kb/data/idioms/ar/idioms_merged_llm_formatted.jsonl
OUT=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/ar/$RUN
mkdir -p "$OUT"

# --- Dimension 2: BPB. In-domain probe = FineWeb-2's OFFICIAL arb_Arab test
# split (genuinely held out from the `train` split we CPT on); OOD probe =
# Arabic Wikipedia with training titles excluded.
for probe in ar_fineweb2_heldout ar_wiki_heldout; do
  echo "### [Dim2] $probe"
  python -m culture.evaluation.perplexity \
    --model_path "$M" --run_name "$RUN" \
    --data_path "$DATA/$probe.jsonl" --text_field text \
    --output_dir "$OUT/ppl_${probe#ar_}" || echo "[warn] $probe failed"
done

# --- Dimensions 3 + 4. Report `primary` (= acc_norm for continuation tasks,
# acc for letter tasks). acc_norm is MANDATORY for alyah/ar_figurative: the gold
# option is the longest one 57.5% of the time.
echo "### [Dim3+4] Arabic MC benchmarks"
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
