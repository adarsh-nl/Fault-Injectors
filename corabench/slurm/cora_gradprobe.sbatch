#!/bin/bash
#SBATCH --job-name=cora-gprobe
#SBATCH --partition=main-gpu   # a40 nodes (ctit086,090-094) are NOT in the ps partition; ps,main-gpu + a40 -> BadConstraints
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.err
#SBATCH --error=logs/%x-%j.err

# sec 7.14 backward-side precedence experiment.
#
#   sbatch cora_gradprobe.sbatch <seed> <max_steps> [resume_ckpt]
#
#   replay:  sbatch ... 2029 9000  $REPO/results/cora_full/561546/ckpt_step6000.pt
#   fresh:   sbatch ... 2026 10000
#
# --constraint=a40: 561546 ran on ctit091 (a40, 13.5 s/step); the replay
# must match the SKU (determinism floor is per-node, sec "cluster is
# heterogeneous"), and the fresh run keeps the same rate arithmetic.
# The GATE STAYS ACTIVE: where it aborts is part of the measurement.

set -uo pipefail

SEED="${1:?usage: sbatch cora_gradprobe.sbatch <seed> <max_steps> [resume]}"
MAXSTEPS="${2:?max_steps required}"
RESUME="${3:-}"

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
DEST="$REPO/results/cora_gradprobe/${SEED}_$SLURM_JOB_ID"
mkdir -p "$DEST"
OUT="$SCRATCH/out"
trap 'rsync -a "${OUT:-/nonexistent}/" "$DEST/" 2>/dev/null || true; rm -rf "$SCRATCH"' EXIT

echo "=== gradprobe seed=$SEED max_steps=$MAXSTEPS resume='$RESUME' node=$(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' \
    /datasets/eemcs/ps/cv/opencood/opv2v/train/ "$SCRATCH/data/"
echo "staged $(find "$SCRATCH/data" -name '*.pcd' | wc -l) pcd files"

HYPES_SRC="$OCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml"
cp "$HYPES_SRC" "$SCRATCH/hypes.yaml"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/hypes.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/hypes.yaml"
sed -i "s|^\( *score_threshold:\).*|\1 0.20|"              "$SCRATCH/hypes.yaml"
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/hypes.yaml" \
    | wc -l | grep -qx 2 || { echo "FATAL: hypes paths not rewritten"; exit 1; }

source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"

RESUME_ARG=""
[ -n "$RESUME" ] && RESUME_ARG="--resume $RESUME"

mkdir -p "$OUT"
python -u -m corabench.scripts.train_opencood \
    --hypes "$SCRATCH/hypes.yaml" --out "$OUT" \
    --max_steps "$MAXSTEPS" \
    --batch_size 2 --lr 1e-3 --num_workers 8 --seed "$SEED" --amp \
    --dt_bound 0.2 --gate_interval 100 --ckpt_every 2000 \
    --assert_canvas 200x704 --assert_head 100x352 \
    --grad_probe $RESUME_ARG &
PID=$!
TICK=0
HEARTBEAT_DONE=0
while kill -0 $PID 2>/dev/null; do
    sleep 60
    TICK=$((TICK + 1))
    if [ "$HEARTBEAT_DONE" = "0" ]; then
        HB=$(ls "$OUT"/*.csv 2>/dev/null | head -1)
        if [ -n "$HB" ]; then
            HBROWS=$(( $(wc -l < "$HB") - 1 ))
            if [ "$HBROWS" -ge 1 ] 2>/dev/null; then
                echo "LAUNCH OK (heartbeat, t+${TICK}min): $HBROWS CSV data rows -- probe columns live"
                head -1 "$HB" | tr ',' '\n' | grep -c "g_\|_gradnorm" \
                  | xargs -I{} echo "  probe columns in header: {}"
                HEARTBEAT_DONE=1
            fi
        fi
    fi
    if [ "$HEARTBEAT_DONE" = "0" ] && [ "$TICK" -ge 240 ]; then
        echo "LAUNCH WARNING (t+${TICK}min): still ZERO CSV rows"
        HEARTBEAT_DONE=2
    fi
    [ $((TICK % 30)) -eq 0 ] && rsync -a "$OUT/" "$DEST/" 2>/dev/null || true
done
RC=0
wait $PID || RC=$?
rsync -a "$OUT/" "$DEST/"
echo "gradprobe seed $SEED exit $RC; results -> $DEST"
exit $RC
