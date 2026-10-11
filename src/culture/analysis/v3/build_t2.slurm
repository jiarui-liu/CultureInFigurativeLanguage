#!/bin/bash
# T2 corpus build (pass 4): the Hindi symbolism triple, or the Arabic no-holdout arm.
# CPU only -- it is one pass over the idiom-bearing shards.
#   sbatch build_t2.slurm hi
#   sbatch build_t2.slurm ar_symall
#SBATCH --job-name=t2-build
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --chdir=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage/src/culture/analysis/v3
#SBATCH --output=logs/t2-build-%j.out
#SBATCH --error=logs/t2-build-%j.err
set -uo pipefail
mkdir -p logs
WHICH="${1:?usage: sbatch build_t2.slurm <hi|ar_symall>}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY X2P_PROXY_URL

REPO=/storage/home/jiaruiliu/local/git-repos/culture-pretraining/CultureInFigurativeLanguage
export PYTHONPATH="${REPO}/src:${REPO}/src/culture/analysis/v2:${REPO}/src/culture/analysis/v3"
export PATH="/storage/home/jiaruiliu/local/git-repos/monitorability-prertaining/.venv/bin:${PATH}"
export HF_HOME=/lustre-storage/fsx_it_0/users/jiaruiliu/hf_home
export TMPDIR=/lustre-storage/fsx_it_0/users/jiaruiliu/tmp
export CULTURE_REPO="$REPO"
D=/lustre-storage/fsx_0/user/jiaruiliu/culture-pretraining-data

case "$WHICH" in
  hi)
    python -u build_t2.py --lang hi --arms untagged,dict,sym --target_tokens 1.05e9
    ;;
  ar_symall)
    # same documents as the existing triple (410,669), so the only difference from
    # `ar_t2_sym` is that the probe's own entities now carry a statement
    python -u build_t2.py --lang ar --arms sym \
      --summaries "$D/t2_ar/ar_entity_symbolism_all.json" \
      --sym_name ar_t2_symall --max_docs 410669 \
      --report build_report_symall.json
    ;;
  *) echo "bad arg: $WHICH"; exit 1;;
esac
echo "=== done ==="
