#!/bin/bash
# Small-scale CPT arm. Usage:
#   sbatch --export=ALL,MODEL=<base>,DATA=<packed prefix>,OUT=<ckpt dir>[,SEED=42,LR=2e-5,MAXSTEPS=-1] train_cpt.slurm
#SBATCH --job-name=bidir-cpt
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/group_data/r3lit_culture_pretrain/culture/bidir/logs/cpt-%j.out
set -euo pipefail
source /home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/bidirectional/stage_env.sh
export HF_HOME=/data/group_data/r3lit_culture_pretrain/.cache/huggingface
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
REPO=/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage
export PYTHONPATH=$REPO/src
NGPU=$(nvidia-smi -L | wc -l)
if [ -f "$OUT/model.safetensors" ] || ls "$OUT"/model-*.safetensors >/dev/null 2>&1; then echo "exists: $OUT"; exit 0; fi
# stage base model and packed data on node-local disk (NFS random reads are slow)
if [[ "$MODEL" == /* ]]; then MODEL_LOCAL=$($REPO/src/culture/bidirectional/stage_model.sh "$MODEL")
else MODEL_LOCAL=$(python $REPO/src/culture/bidirectional/stage_hf.py "$MODEL" | tail -1); fi
DL=/scratch/jiaruil5/packed/$(echo -n "$DATA" | md5sum | cut -c1-8)
mkdir -p $DL; [ -f $DL/.complete ] || { cp "$DATA.bin" "$DATA.json" $DL/ && touch $DL/.complete; }
DATA_LOCAL=$DL/$(basename "$DATA")
python -m torch.distributed.run --nproc_per_node $NGPU --master_port $((20000 + SLURM_JOB_ID % 20000)) \
  -m culture.bidirectional.train_cpt --model "$MODEL_LOCAL" --data "$DATA_LOCAL" --out "$OUT" \
  --seed "${SEED:-42}" --lr "${LR:-2e-5}" --max_steps "${MAXSTEPS:--1}" \
  --micro_batch "${MICRO:-1}" --global_batch "${GBS:-128}" \
  --ckpt_dir /scratch/jiaruil5/ckpt/$(echo -n "$OUT" | md5sum | cut -c1-8) \
  --deepspeed $REPO/src/culture/bidirectional/ds_z2.json
