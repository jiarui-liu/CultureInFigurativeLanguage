#!/bin/bash
# Core Hindi CPT evaluation for ONE checkpoint, on a single H100. Submit once per model:
#   sbatch src/culture/evaluation/eval_core.slurm base
#   sbatch src/culture/evaluation/eval_core.slurm cpt
#
# Two blocks:
#   A. OFFLINE core (no network): Dim1 Hindi PPL (Samanantar clean + proverb in-domain)
#      and Dim3 MABL/MILU/Global PIQA (loglik) + IdiomCE generation. All inputs are
#      local files and the model is local, so this always runs.
#   B. BEST-EFFORT network extras: Dim1 WikiText retention + Dim2 MMLU/BoolQ. These
#      pull from HuggingFace; if the compute node has no egress they are skipped
#      without affecting block A. Written to a separate output dir.
#
# The judge (OpenAI) import is lazy, so block A needs no openai/together key.
#SBATCH --job-name=hi-eval-core
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-core-%x-%j.out
#SBATCH --error=logs/eval-core-%x-%j.err
#SBATCH --partition=h100

set -uo pipefail   # not -e: block B failures must not abort the job

RUN="${1:?usage: sbatch eval_core.slurm <base|cpt>}"

REPO_ROOT="/storage/home/jiaruiliu/local/git-repos/culture-pretraining"
CIFL="${REPO_ROOT}/CultureInFigurativeLanguage"
cd "${CIFL}"
mkdir -p logs

VENV="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${CIFL}/.eval_deps:${CIFL}/src"   # .eval_deps optional (only for judging)

# Drop the submit-time sandbox proxy (10.0.2.2) so the compute node uses its own
# network for block B; unreachable proxy otherwise times out every HF call.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export no_proxy="*" NO_PROXY="*"

BASE_MODEL="/lustre-storage/fsx_2/user/jiaruiliu/models/Qwen3.5-9B"
CPT_MODEL="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt"
UNF_MODEL="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt-unfiltered"
UNTAG_MODEL="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/ckpts/qwen3p5-9b-hi-cpt-untagged"
case "${RUN}" in
  base) MODEL="${BASE_MODEL}" ;;
  cpt) MODEL="${CPT_MODEL}" ;;
  unfiltered) MODEL="${UNF_MODEL}" ;;
  untagged) MODEL="${UNTAG_MODEL}" ;;
  *) echo "unknown RUN='${RUN}' (use base|cpt|unfiltered|untagged)"; exit 1 ;;
esac

DATA="${REPO_ROOT}/data/eval/hi"
OUT="/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi/${RUN}"
OUT_EN="${OUT}/en"                      # Dim2 English extras (best-effort)
mkdir -p "${OUT}" "${OUT_EN}"

export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID}"

echo "=========================================================="
echo "RUN=${RUN}  MODEL=${MODEL}"
echo "OUT=${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=========================================================="

# =================== BLOCK A — OFFLINE CORE (must succeed) ==================
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

echo "### [Dim1] Samanantar Hindi (clean, independent adaptation probe)"
python -m culture.evaluation.perplexity --model_path "${MODEL}" --run_name "${RUN}" \
  --data_path "${DATA}/hi_samanantar_heldout.jsonl" --text_field text \
  --output_dir "${OUT}/ppl_hi_samanantar"

echo "### [Dim1] Proverb in-domain slice (CONTAMINATED: from training corpus)"
python -m culture.evaluation.perplexity --model_path "${MODEL}" --run_name "${RUN}" \
  --data_path "${DATA}/hi_proverbs_heldout.jsonl" --text_field text \
  --output_dir "${OUT}/ppl_hi_proverbs"

echo "### [Dim3] MABL / MILU / Global PIQA (loglik) + IdiomCE generation"
python -m culture.evaluation.run_eval \
  --model_path "${MODEL}" --run_name "${RUN}" \
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

# =================== BLOCK B — BEST-EFFORT NETWORK EXTRAS ===================
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}" HF_DATASETS_CACHE="/tmp/hfds_${SLURM_JOB_ID}"

echo "### [Dim1] WikiText-103 (English retention) [best-effort]"
python -m culture.evaluation.perplexity --model_path "${MODEL}" --run_name "${RUN}" \
  --hf_dataset wikitext --hf_config wikitext-103-raw-v1 --hf_split test \
  --output_dir "${OUT}/ppl_wikitext" || echo "[warn] WikiText skipped (no network?)"

echo "### [Dim2] MMLU (0-shot) + BoolQ (0-shot) [best-effort]"
python -m culture.evaluation.run_eval \
  --model_path "${MODEL}" --run_name "${RUN}" \
  --tasks mmlu,boolq \
  --num_fewshot 0 --batch_size 8 --no_judge \
  --output_dir "${OUT_EN}" || echo "[warn] MMLU/BoolQ skipped (no network?)"

echo "=========================================================="
echo "DONE ${RUN}. Core results under ${OUT}; English extras under ${OUT_EN}"
echo "=========================================================="
