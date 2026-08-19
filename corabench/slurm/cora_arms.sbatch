#!/bin/bash
#SBATCH --job-name=cora-arm
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.err
#SBATCH --error=logs/%x-%j.err

# docs/cora_spec.md sec 7.7 -- four arms, 9,000 steps, PRE-REGISTERED.
#
#   sbatch cora_arms.sbatch control|A|B|C
#
#     control  unchanged                       (same-batch baseline)
#     A        --lambda_align 0.0
#     B        --lambda_align 0.1
#     C        --ema_momentum 0.99   (tau 100 instead of 999)
#
# --constraint=l40 so the step rate is ~8.7 s/step, not the 13.5 s/step
# 561546 saw on ctit091. 9,000 steps ~= 22 h at that rate.
#
# BOUND: 9,000 steps clears 561546's abort (7,667) but NOT 559640's turn
# (~16k). A surviving arm at 9k is NOT evidence of a fix.
#
# The gate stays ACTIVE: an arm that diverges should abort rather than burn
# the budget, and WHERE it aborts is part of the measurement.

set -uo pipefail

ARM="${1:?usage: sbatch cora_arms.sbatch control|A|B|C|rung2|rung3}"
case "$ARM" in
  control) EXTRA="" ;;
  A)       EXTRA="--lambda_align 0.0" ;;
  B)       EXTRA="--lambda_align 0.1" ;;
  C)       EXTRA="--ema_momentum 0.99" ;;
  # sec 7.11 ablation ladder. 2000 steps: 2000 x 8.7 s/step = 4.83 h training
  # + ~3.0 h staging = ~7.8 h; submit at 10 h (~28% headroom). NOT 12h
  # carried over -- that is what truncated the 9000-step arms.
  rung2)   EXTRA="--no_pac --no_teacher" ;;   # local + LC only
  rung3)   EXTRA="--no_teacher" ;;            # local + LC + PAC
  *) echo "FATAL: unknown arm $ARM"; exit 1 ;;
esac

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
DEST="$REPO/results/cora_arms/${ARM}_$SLURM_JOB_ID"
mkdir -p "$DEST"
OUT="$SCRATCH/out"
trap 'rsync -a "${OUT:-/nonexistent}/" "$DEST/" 2>/dev/null || true; rm -rf "$SCRATCH"' EXIT

echo "=== arm=$ARM extra='$EXTRA' node=$(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

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

mkdir -p "$OUT"
python -m corabench.scripts.train_opencood \
    --hypes "$SCRATCH/hypes.yaml" --out "$OUT" \
    --max_steps ${STEPS:-9000} \
    --batch_size 2 --lr 1e-3 --num_workers 8 --seed ${SEED:-2029} --amp \
    --dt_bound 0.2 --gate_interval 100 --ckpt_every 3000 \
    --assert_canvas 200x704 --assert_head 100x352 \
    $EXTRA &
PID=$!
TICK=0
HEARTBEAT_DONE=0
while kill -0 $PID 2>/dev/null; do
    sleep 60
    TICK=$((TICK + 1))
    # MID-RUN LAUNCH HEARTBEAT. The end-of-job guard was silent on 3 of 4 arms
    # because Slurm kills a TIMEOUT job before any line after `wait` runs, and
    # before `wait` there is nothing final to check. Assert DURING the run
    # instead: once a CSV exists with >=1 data row, say so once. That line
    # survives a wall kill because it is emitted while the job is alive.
    if [ "$HEARTBEAT_DONE" = "0" ]; then
        HB=$(ls "$OUT"/*.csv 2>/dev/null | head -1)
        if [ -n "$HB" ]; then
            HBROWS=$(( $(wc -l < "$HB") - 1 ))
            if [ "$HBROWS" -ge 1 ] 2>/dev/null; then
                echo "LAUNCH OK (heartbeat, t+${TICK}min): $HBROWS CSV data rows -- the run has taken optimiser steps"
                HEARTBEAT_DONE=1
            fi
        fi
    fi
    # and if we are 20 minutes past staging with still no rows, say THAT too
    if [ "$HEARTBEAT_DONE" = "0" ] && [ "$TICK" -ge 240 ]; then
        echo "LAUNCH WARNING (t+${TICK}min): still ZERO CSV rows -- the run may never have started"
        HEARTBEAT_DONE=2
    fi
    [ $((TICK % 30)) -eq 0 ] && rsync -a "$OUT/" "$DEST/" 2>/dev/null || true
done
RC=0
wait $PID || RC=$?

rsync -a "$OUT/" "$DEST/"

# ── LAUNCH GUARD ────────────────────────────────────────────────────────
# Four arms produced 0-byte CSVs for three days because a NameError killed
# them at model construction, before a single optimiser step. The job exit
# code alone did not make that loud enough. Assert explicitly that the run
# actually STARTED: at least one CSV data row, i.e. at least one step.
CSV=$(ls "$DEST"/*.csv 2>/dev/null | head -1)
ROWS=0
[ -n "$CSV" ] && ROWS=$(( $(wc -l < "$CSV") - 1 ))
[ "$ROWS" -lt 0 ] && ROWS=0
echo ""
echo "######## LAUNCH GUARD arm $ARM ########"
if [ "$ROWS" -ge 1 ]; then
    echo "LAUNCH OK: $ROWS CSV data rows written (>=1 optimiser step taken)"
else
    echo "LAUNCH FAILED: csv='$CSV' rows=$ROWS -- the arm never took a step."
    echo "  This is NOT a CoRA result. Do not read it as one."
    exit 2      # distinct from the trainer's GateAbort exit 1
fi

echo ""
echo "######## arm $ARM per-200-step decomposition ########"
"$REPO/.venv-hpc/bin/python" "$REPO/tools/cora_arm_report.py" "$DEST" "$ARM"
echo "arm $ARM exit $RC; results -> $DEST"
exit $RC
