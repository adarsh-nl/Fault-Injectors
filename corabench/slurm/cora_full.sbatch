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

# CoRA FULL training run -- 30 epochs on the complete OPV2V train split.
#
#   sbatch --nice=<n> corabench/slurm/cora_full.sbatch <seed>
#
# Launched TWO-UP on different seeds. Basin-entry base rate is 1/5 (job
# 559461, 5 seeds x 500 steps), and all three candidate mitigations are
# CLOSED with contradicting evidence (docs/cora_spec.md sec 7: dt_bound,
# init_scale, cssm_in breaker). Two independent seeds take a 20% risk of
# losing the run down to ~4% of losing both, at the same wall-clock.
#
# The RUNTIME GATE is the only working safety net and stays ACTIVE: the
# skip-rate breaker (>20% non-finite LOSS in the trailing 200 steps, checked
# every step, discriminating true forward failure from GradScaler backoff)
# aborted job 559279 at step 199 rather than burning the full budget. That is
# what makes launching without a fix acceptable.
#
# Baseline sweep jobs keep GPU priority -- pass --nice.

set -euo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
# COPY BACK ON ANY EXIT PATH, THEN wipe. Job 559639 lost rows 190-361 of its
# loss log -- exactly the window containing basin entry and the gate abort --
# because the script ran under `set -e` and `wait $TRAIN_PID` returned the
# trainer's non-zero GateAbort exit, terminating the script AT THE WAIT and
# skipping the final rsync. Only the last 30-minute tick survived. $OUT/$DEST
# are set later; if this fires before then, the rsync no-ops harmlessly.
trap 'rsync -a "${OUT:-/nonexistent}/" "${DEST:-/nonexistent}/" 2>/dev/null || true; rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"
SEED="${1:?usage: sbatch cora_full.sbatch <seed>}"
OCOOD="$HOME/opencood-official"

echo "=== seed=$SEED node=$(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# ── stage the FULL train split, LiDAR+yaml only (~63 GB) ────────────────
mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' \
    /datasets/eemcs/ps/cv/opencood/opv2v/train/ "$SCRATCH/data/"
echo "staged $(find "$SCRATCH/data" -name '*.pcd' | wc -l) pcd files"

# ── hypes: OpenCOOD's OPV2V-NATIVE config, NOT a baseline model's ──────
#
# ROOT CAUSE OF THE 559595/559596 MIS-LAUNCH, do not undo this.
# This script used to copy ~/opencood-eval/cobevt/config.yaml and rewrite
# only root_dir. That silently inherited CoBEVT's ARCHITECTURE choices:
#   range +-38.4 (not +-40)  ->  canvas (192,704) instead of (200,704)
#   feature_stride 4 (not 2) ->  head   (48,176)  instead of (100,352)
# CoBEVT uses those values on OPV2V *and* V2XSet, so they are not a V2XSet
# artifact -- they are CoBEVT's, and they are baked into any weights trained
# under them. The CoRA paper says "We adopted the basic settings from
# OpenCOOD", and docs/cora_spec.md sec 4 records range (-140.8,-40,-3,
# 140.8,40,1) -> 704x200 canvas, feature 352x100. The OPV2V-native OpenCOOD
# config below is the one that matches; a baseline model's config is never
# the right base for CoRA.
HYPES_SRC="$OCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml"
[ -f "$HYPES_SRC" ] || { echo "FATAL: missing $HYPES_SRC"; exit 1; }
cp "$HYPES_SRC" "$SCRATCH/hypes.yaml"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/hypes.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/hypes.yaml"
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/hypes.yaml" \
    | wc -l | grep -qx 2 || { echo "FATAL: hypes paths not rewritten"; exit 1; }
# eval-time, corrected while we are here: OpenCOOD's OPV2V default is 0.20;
# CoBEVT's config carried 0.25. Unlike the geometry this one WOULD have been
# fixable by re-evaluating a trained checkpoint.
sed -i "s|^\( *score_threshold:\).*|\1 0.20|" "$SCRATCH/hypes.yaml"
echo "hypes: $HYPES_SRC"
grep -E "cav_lidar_range|voxel_size|feature_stride|score_threshold" \
    "$SCRATCH/hypes.yaml" | head

source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"

OUT="$SCRATCH/out"
python -m corabench.scripts.train_opencood \
    --hypes "$SCRATCH/hypes.yaml" \
    --out "$OUT" \
    --epochs 30 \
    --batch_size 2 --lr 1e-3 --num_workers 8 --seed "$SEED" --amp \
    --dt_bound 0.2 --gate_interval 100 --ckpt_every 2000 \
    --assert_canvas 200x704 --assert_head 100x352 &
TRAIN_PID=$!

# periodic copy-back so a wall/crash keeps the latest checkpoints + loss log
DEST="$REPO/results/cora_full/$SLURM_JOB_ID"
mkdir -p "$DEST"
TICK=0
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
    TICK=$((TICK + 1))
    if [ $((TICK % 30)) -eq 0 ]; then
        rsync -a "$OUT/" "$DEST/" 2>/dev/null || true
    fi
done
# `|| RC=$?` so a non-zero trainer exit does NOT abort the script here.
RC=0
wait $TRAIN_PID || RC=$?

rsync -a "$OUT/" "$DEST/"
echo "training exit $RC (seed $SEED); results -> $DEST"
exit $RC
