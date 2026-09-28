#!/bin/bash
# Source me. Provides the bidir Python environment on node-local /scratch, once per node:
#   1. if already present (marker file), just activate it;
#   2. else build it from PyPI with uv using the frozen requirements (the Hub/PyPI CDN is
#      far faster than our NFS), plus the prebuilt causal_conv1d extension;
#   3. if that fails, unpack the NFS tarball; if that is missing, use the NFS env.
BIDIR=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional
ENV_NFS=/data/group_data/r3lit_culture_pretrain/envs/bidir
ENV_TAR=/data/group_data/r3lit_culture_pretrain/envs/bidir_env.tar
EXTRA=/data/group_data/r3lit_culture_pretrain/envs/extra
LOCAL_ROOT=/scratch/jiaruil5/envs
mkdir -p "$LOCAL_ROOT"
(
  flock -w 5400 9
  if [ ! -f "$LOCAL_ROOT/.complete" ]; then
    rm -rf "$LOCAL_ROOT/bidir" "$LOCAL_ROOT/python"
    export UV_CACHE_DIR=/scratch/jiaruil5/uvcache UV_PYTHON_INSTALL_DIR="$LOCAL_ROOT/python"
    UV=/home/jiaruil5/.local/bin/uv
    if $UV python install 3.12 >/dev/null 2>&1 || ls -d "$LOCAL_ROOT"/python/cpython-3.12*/bin/python3.12 >/dev/null 2>&1; then
      PY=$(ls -d "$LOCAL_ROOT"/python/cpython-3.12*/bin/python3.12 | head -1)
      $UV venv -q -p "$PY" "$LOCAL_ROOT/bidir" && \
      VIRTUAL_ENV="$LOCAL_ROOT/bidir" $UV pip install -q -r "$BIDIR/requirements_bidir.txt" && \
      cp -r "$EXTRA"/causal_conv1d* "$LOCAL_ROOT/bidir/lib/python3.12/site-packages/" && \
      touch "$LOCAL_ROOT/.complete"
    fi
    if [ ! -f "$LOCAL_ROOT/.complete" ] && [ -f "$ENV_TAR" ]; then
      rm -rf "$LOCAL_ROOT/bidir" "$LOCAL_ROOT/python"
      tar -xf "$ENV_TAR" -C "$LOCAL_ROOT" && touch "$LOCAL_ROOT/.complete"
    fi
  fi
) 9>"/scratch/jiaruil5/.env_lock"
if [ -f "$LOCAL_ROOT/.complete" ]; then
  export VIRTUAL_ENV="$LOCAL_ROOT/bidir"
else
  export VIRTUAL_ENV="$ENV_NFS"
fi
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r
export HF_TOKEN=${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}
export NCCL_P2P_DISABLE=1  # P2P hangs NCCL init on some babel nodes
