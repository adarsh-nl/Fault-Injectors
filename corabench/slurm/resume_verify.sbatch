#!/bin/bash
#SBATCH --job-name=cora-resume-verify
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# END-TO-END RESUME VERIFICATION.
#
#   run A: 2N steps uninterrupted
#   run B: N steps -> checkpoint -> --resume -> N more
#   compare loss, lr, scaler scale, step-for-step.
#
# A SMALL SCENARIO SUBSET is staged on purpose, not the full split: staging
# the whole train split costs 7.7-9.7 h and would dwarf a 200-step test, and
# a small set makes epochs short so the resume CROSSES AN EPOCH BOUNDARY --
# which is the case that actually exercises the scheduler step, the per-epoch
# loader reseed and the batch-skip path.

set -uo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rsync -a "${OUT_ROOT:-/nonexistent}/" "${DEST:-/nonexistent}/" 2>/dev/null || true; rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
N=""   # DERIVED at run time from the staged dataset -- see the probe
SRC=/datasets/eemcs/ps/cv/opencood/opv2v/train

echo "=== resume verify: N=$N node=$(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

mkdir -p "$SCRATCH/data"
# ONE scenario: with 3 the dataset was 572 samples = 286 steps/epoch, so a
# 200-step run never left epoch 0 and sched.step() / the per-epoch reseed
# were NEVER REACHED -- the very paths the resume has to get right. One
# scenario gives ~95 steps/epoch, so N=60 checkpoints mid-epoch-0 and the
# resumed run crosses into epoch 1.
for s in $(ls "$SRC" | head -1); do
    rsync -a --exclude='*.png' "$SRC/$s" "$SCRATCH/data/"
done
echo "staged $(find "$SCRATCH/data" -name '*.pcd' | wc -l) pcd files from $(ls "$SCRATCH/data" | wc -l) scenarios"

cp "$OCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml" "$SCRATCH/hypes.yaml"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/hypes.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/hypes.yaml"

source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"

OUT_ROOT="$SCRATCH/out"
DEST="$REPO/results/resume_verify/$SLURM_JOB_ID"
mkdir -p "$OUT_ROOT" "$DEST"

COMMON="--hypes $SCRATCH/hypes.yaml --batch_size 2 --lr 1e-3 --num_workers 4 \
        --seed 4242 --amp --dt_bound 0.2 --no_gate \
        --assert_canvas 200x704 --assert_head 100x352"

# ── SIZE BY MEASUREMENT, NOT ESTIMATION ────────────────────────────────
# Three consecutive runs missed the epoch boundary because N was set from an
# ESTIMATED sample count: 3 scenarios gave 572 samples, so one scenario was
# assumed to be ~190 -- the first scenario alone actually holds 419, i.e.
# 209 steps/epoch, and a 120-step run never left epoch 0. Scenario sizes
# vary, so averaging across them is exactly the wrong move.
#
# Probe the REAL count from the staged data, then derive N so that the full
# 2N run overshoots one epoch with margin.
echo ""; echo "######## probe: measure the staged dataset ########"
PROBE=$(python -m corabench.scripts.train_opencood $COMMON \
        --out "$SCRATCH/probe" --max_steps 1 --ckpt_every 1000000 2>&1 \
        | tee /dev/stderr | grep -m1 -oE "dataset [0-9]+ samples")
SAMPLES=$(echo "$PROBE" | grep -oE "[0-9]+")
[ -n "$SAMPLES" ] || { echo "FATAL: could not measure sample count"; exit 1; }
SPE=$((SAMPLES / 2))                       # batch_size 2, drop_last
N=$(( SPE / 2 + 40 ))                      # so 2N = SPE + 80
echo "MEASURED samples=$SAMPLES  steps_per_epoch=$SPE  ->  N=$N  2N=$((2*N))"
[ $((2*N)) -gt "$SPE" ] || { echo "FATAL: 2N=$((2*N)) does not exceed SPE=$SPE"; exit 1; }
echo "epoch boundary at step $SPE will be crossed at 2N=$((2*N))"

echo ""; echo "######## run A: $((2*N)) steps uninterrupted ########"
python -m corabench.scripts.train_opencood $COMMON \
    --out "$OUT_ROOT/A" --max_steps $((2*N)) --ckpt_every 1000000

echo ""; echo "######## run B1: first $N steps ########"
python -m corabench.scripts.train_opencood $COMMON \
    --out "$OUT_ROOT/B1" --max_steps "$N" --ckpt_every "$N"

echo ""; echo "######## run B2: resume -> $N more ########"
python -m corabench.scripts.train_opencood $COMMON \
    --out "$OUT_ROOT/B2" --max_steps $((2*N)) --ckpt_every 1000000 \
    --resume "$OUT_ROOT/B1/ckpt_step$N.pt"

rsync -a "$OUT_ROOT/" "$DEST/"
echo ""; echo "######## COMPARISON ########"
"$REPO/.venv-hpc/bin/python" "$REPO/corabench/slurm/resume_compare.py" \
    "$DEST/A/smoke_loss.csv" "$DEST/B1/smoke_loss.csv" "$DEST/B2/smoke_loss.csv" \
    "$N" "$DEST/B1/ckpt_step$N.pt"
echo "results -> $DEST"
