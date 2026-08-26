#!/bin/bash
# Hindi CPT-eval pipeline applied to the SFT checkpoints (SAME tasks/harness as
# eval_core.slurm, for apples-to-apples deltas vs the CPT numbers).
#   sbatch eval_core_sft.slurm hi-cpt-sft
#   sbatch eval_core_sft.slurm hi-base-sft
#SBATCH --job-name=hi-eval-core-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-core-%x-%j.out
#SBATCH --error=logs/eval-core-%x-%j.err
#SBATCH --partition=h100
set -uo pipefail
RUN="${1:?usage: sbatch eval_core_sft.slurm <hi-cpt-sft|hi-base-sft>}"
REPO_ROOT="/storage/home/jiaruiliu/local/git-repos/culture-pretraining"
CIFL="${REPO_ROOT}/CultureInFigurativeLanguage"; cd "${CIFL}"; mkdir -p logs
VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${CIFL}/.eval_deps:${CIFL}/src"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export no_proxy="*" NO_PROXY="*"
CK=/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts
case "${RUN}" in
  hi-cpt-sft)  MODEL="${CK}/qwen3p5-9b-hi-cpt-sft" ;;
  hi-base-sft) MODEL="${CK}/qwen3p5-9b-hi-base-sft" ;;
  *) echo "unknown RUN='${RUN}'"; exit 1 ;;
esac
DATA="${REPO_ROOT}/data/eval/hi"
OUT="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi/${RUN}"
OUT_EN="${OUT}/en"; mkdir -p "${OUT}" "${OUT_EN}"
export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"
echo "RUN=${RUN} MODEL=${MODEL} OUT=${OUT}"

# ---- BLOCK A (offline core) ----
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
python -m culture.evaluation.perplexity --model_path "${MODEL}" --run_name "${RUN}" \
  --data_path "${DATA}/hi_samanantar_heldout.jsonl" --text_field text \
  --output_dir "${OUT}/ppl_hi_samanantar"
python -m culture.evaluation.perplexity --model_path "${MODEL}" --run_name "${RUN}" \
  --data_path "${DATA}/hi_proverbs_heldout.jsonl" --text_field text \
  --output_dir "${OUT}/ppl_hi_proverbs"
python -m culture.evaluation.run_eval --model_path "${MODEL}" --run_name "${RUN}" \
  --tasks mabl,milu,global_piqa,idiomce \
  --mabl_path "${DATA}/mabl_hi.csv" \
  --milu_path "${DATA}/milu_hi_test.jsonl" \
  --milu_fewshot_path "${DATA}/milu_hi_val.jsonl" \
  --global_piqa_path "${DATA}/global_piqa_hi.tsv" \
  --idiomce_path "${DATA}/idiomce_hi.jsonl" \
  --num_fewshot 0 --milu_num_fewshot 5 \
  --batch_size 8 --gen_batch_size 8 --no_judge \
  --output_dir "${OUT}"
echo "### BLOCK A done."

# ---- BLOCK B (best-effort network extras) ----
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"
python -m culture.evaluation.perplexity --model_path "${MODEL}" --run_name "${RUN}" \
  --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
  --output_dir "${OUT}/ppl_wikitext" || echo "[warn] WikiText skipped"
python -m culture.evaluation.run_eval --model_path "${MODEL}" --run_name "${RUN}" \
  --tasks mmlu,boolq --num_fewshot 0 --batch_size 8 --no_judge \
  --output_dir "${OUT_EN}" || echo "[warn] MMLU/BoolQ skipped"
echo "DONE ${RUN}."
