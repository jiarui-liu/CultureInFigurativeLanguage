#!/bin/bash
# Copy a model directory to node-local /scratch (sequential reads are far faster than
# safetensors mmap page faults over NFS) and print the local path. Reuses a complete
# copy; the local name includes a hash of the source path so checkpoints never collide.
set -euo pipefail
SRC=$(readlink -f "$1")
H=$(echo -n "$SRC" | md5sum | cut -c1-8)
DST=/scratch/jiaruil5/models/$(basename "$SRC")-$H
mkdir -p /scratch/jiaruil5/models
# reuse a complete copy staged under the older naming scheme (basename only)
OLD=/scratch/jiaruil5/models/$(basename "$SRC")
if [ -f "$OLD/.complete" ]; then echo "$OLD"; exit 0; fi
if [ ! -f "$DST/.complete" ]; then
  rm -rf "$DST" "$DST".tmp.*
  mkdir -p "$DST.tmp.$$"
  # skip optimizer/trainer state; only what is needed to load the model
  for f in "$SRC"/*; do
    case "$(basename "$f")" in checkpoint-*|global_step*|*.bin.index.json|training_args.bin|trainer_state.json|optimizer*|rng_state*) continue;; esac
    cp -r "$f" "$DST.tmp.$$/"
  done
  mv "$DST.tmp.$$" "$DST" && touch "$DST/.complete"
fi
echo "$DST"
