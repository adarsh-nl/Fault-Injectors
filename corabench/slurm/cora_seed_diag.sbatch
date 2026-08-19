#!/bin/bash
#SBATCH --job-name=cora-seeddiag
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CoRA 5-SEED BASIN DIAGNOSTIC.
#
# What this is measuring, and why it is not another dt_bound run:
# job 559279 aborted at step 199 (50% non-finite) while 559057 ran 5000 steps
# at the SAME dt_bound=0.2. Controlling for time, delta_p99 does NOT separate
# non-finite steps from finite ones (within-window separation is ~-0.003, i.e.
# the wrong sign), and there are 41 steps with p99 >= 0.1995 and a finite loss
# plus 3 non-finite steps with p99 < 0.17. So Delta saturation is neither
# necessary nor sufficient and is retained here as a CO-SYMPTOM readout only.
#
# The three questions:
#   1. how many of 5 seeds enter the bad basin, and at what step
#   2. does LC activation magnitude diverge BEFORE the first non-finite value
#      (i.e. does an early warning exist at all)
#   3. do failing seeds differ from surviving seeds in their OPENING skip
#      pattern -- all runs are bit-identical through step 3 and diverge at
#      step 4 on a GradScaler skip/no-skip flip
#
# --no_gate is deliberate: the skip-rate breaker would abort a bad seed around
# step 199 and truncate its trajectory. Every seed runs the full 500 steps and
# basin entry is detected offline from the per-step CSV.

set -uo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
STEPS=${STEPS:-500}
SEEDS=${SEEDS:-"2026 2027 2028 2029 2030"}

echo "=== seeds=[$SEEDS] steps=$STEPS node=$(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# stage once, reuse for all five seeds
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

DEST="$REPO/results/cora_seeddiag/$SLURM_JOB_ID"
mkdir -p "$DEST"

FAILED=0
for S in $SEEDS; do
    echo ""
    echo "################ seed $S ################"
    # config IDENTICAL across seeds except --seed. dt_bound stays at the
    # approved 0.2; this experiment does not vary it.
    python -m corabench.scripts.train_opencood \
        --hypes "$SCRATCH/hypes.yaml" \
        --out "$SCRATCH/out/seed$S" \
        --max_steps "$STEPS" \
        --batch_size 2 --lr 1e-3 --num_workers 8 --seed "$S" --amp \
        --dt_bound 0.2 --no_gate --act_probe --ckpt_every 100000
    RC=$?
    [ $RC -eq 0 ] || { echo "seed $S exited $RC"; FAILED=1; }
    # copy back after EVERY seed, so a wall loses at most the in-flight seed
    mkdir -p "$DEST/seed$S"
    rsync -a "$SCRATCH/out/seed$S/" "$DEST/seed$S/" 2>/dev/null || true
    echo "seed $S -> $DEST/seed$S"
done

echo ""
echo "=== ANALYSIS ==="
"$REPO/.venv-hpc/bin/python" "$REPO/corabench/slurm/seed_diag_report.py" "$DEST"
echo ""
echo "=== SEED DIAG DONE failed=$FAILED  results -> $DEST ==="
exit $FAILED
