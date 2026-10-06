#!/bin/bash
# Task 1b end-to-end: split abstract_other into seven subtypes, bootstrap the expanded table,
# then validate the subtype labels against two independent annotators. Mirrors the first-level
# typology pass; every step is cached in $LLM_CACHE_DIR, so a requeue resumes rather than
# re-annotates.
#   sbatch entity_subtypology_remote.slurm
#SBATCH --job-name=ent-subtyp
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=h100
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v2
#SBATCH --output=logs/ent-subtyp-%j.out
#SBATCH --error=logs/ent-subtyp-%j.err
set -euo pipefail
mkdir -p logs

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL

REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH="${REPO}/src:${REPO}/src/culture/analysis/v2"
VENV="/storage/home/jiaruiliu/local/git-repos/monitorability/.venv"
export PATH="${VENV}/bin:${PATH}"
export HF_HOME=/lustre-storage/fsx_it_0/users/jiaruiliu/hf_home
export HF_HUB_DISABLE_XET=1
export TMPDIR=/lustre-storage/fsx_it_0/users/jiaruiliu/tmp

export CULTURE_REPO="$REPO"
export CULTURE_LLM_PRIMARY=Qwen/Qwen3.5-27B-FP8
export CULTURE_LLM_SECOND=Qwen/Qwen3.5-9B
export CULTURE_LLM_THIRD=CohereLabs/aya-expanse-8b
export CULTURE_EVAL_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/eval
export LLM_CACHE_DIR=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data/bidir/llm_cache
export VLLM_TP=2

echo "=== 1/4 subtypology pass (Qwen3.5-27B-FP8, all abstract entities) ==="
python entity_subtypology.py

echo "=== 2/4 idiom-level bootstrap on the expanded table ==="
python entity_subtypology_bootstrap.py

# one model per process, so the two second annotators run as separate invocations
echo "=== 3/4 validation: second annotator Qwen3.5-9B ==="
python entity_subtypology_validate.py --only Qwen3.5-9B

echo "=== 4/4 validation: third annotator aya-expanse-8b ==="
python entity_subtypology_validate.py --only aya

echo "=== done ==="
