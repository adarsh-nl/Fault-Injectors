#!/bin/bash
#SBATCH --job-name=cora-proof
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CoRA PROOF run -- 5000 steps, deliberately PAST the step-3501 failure.
#
# The previous proofs were insufficient by construction: a synthetic init
# check and a 400-step smoke both PASSED while delta was already escaping,
# and the run died at 3501. This run covers the regime where it actually
# broke, and reports the observables that were never being read:
#   delta_max / delta_p99 / delta_mean / sat_frac   (bound = 0.2)
#   teacher/student magnitude ratio
#   non-finite-loss step count
# Full OPV2V train split, so the data distribution matches the real run.

set -uo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
STEPS=${STEPS:-5000}
DT_BOUND=${DT_BOUND:-0.2}

echo "=== dt_bound=$DT_BOUND steps=$STEPS node=$(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

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
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"

DEST="$REPO/results/cora_proof/$SLURM_JOB_ID"
mkdir -p "$DEST"

python -m corabench.scripts.train_opencood \
    --hypes "$SCRATCH/hypes.yaml" \
    --out "$SCRATCH/out" \
    --max_steps "$STEPS" \
    --batch_size 2 --lr 1e-3 --num_workers 8 --seed 2026 --amp \
    --dt_bound "$DT_BOUND" --gate_interval 100 --ckpt_every 1000
TRAIN_RC=$?

rsync -a "$SCRATCH/out/" "$DEST/"
CSV="$DEST/smoke_loss.csv"; [ -f "$CSV" ] || CSV="$DEST/train_loss.csv"
python "$REPO/corabench/slurm/proof_verdict.py" "$CSV" "$DT_BOUND"
echo "results -> $DEST"
# exit code drives --dependency=afterok on the chained full run
exit $TRAIN_RC
