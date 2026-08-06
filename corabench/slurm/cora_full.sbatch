#!/bin/bash
#SBATCH --job-name=cora-full
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --time=3-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CoRA FULL training run -- 30 epochs on the complete OPV2V train split
# through the A1+B1 path. PREPARED ON HOLD: submission gated by the user
# (smoke 557653 PASSED: 8.16 -> 5.99 -> 3.71, finite, 5/400 guarded skips).
# Baseline sweep jobs keep GPU priority.

set -euo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"

echo "=== node $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# ── stage the FULL train split, LiDAR+yaml only (~63 GB) ────────────────
mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' \
    /datasets/eemcs/ps/cv/opencood/opv2v/train/ "$SCRATCH/data/"
echo "staged $(find "$SCRATCH/data" -name '*.pcd' | wc -l) pcd files"

cp ~/opencood-eval/cobevt/config.yaml "$SCRATCH/hypes.yaml"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/hypes.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/hypes.yaml"
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/hypes.yaml" \
    | wc -l | grep -qx 2 || { echo "FATAL: hypes paths not rewritten"; exit 1; }

source activate opencood-official
export PYTHONPATH="$HOME/opencood-official:$REPO:${PYTHONPATH:-}"
cd "$REPO"

OUT="$SCRATCH/out"
python -m corabench.scripts.train_opencood \
    --hypes "$SCRATCH/hypes.yaml" \
    --out "$OUT" \
    --epochs 30 \
    --batch_size 2 --lr 1e-3 --num_workers 8 --seed 2026 --amp \
    --ckpt_every 2000 &
TRAIN_PID=$!

# periodic copy-back so a wall/crash keeps the latest checkpoints + loss log
DEST="$REPO/results/cora_full/$SLURM_JOB_ID"
mkdir -p "$DEST"
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 1800
    rsync -a "$OUT/" "$DEST/" 2>/dev/null || true
done
wait $TRAIN_PID
RC=$?

rsync -a "$OUT/" "$DEST/"
echo "training exit $RC; results -> $DEST"
exit $RC
