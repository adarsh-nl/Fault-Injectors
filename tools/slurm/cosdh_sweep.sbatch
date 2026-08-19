#!/bin/bash
#SBATCH --job-name=cosdh-sweep
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CoSDH sweep, one (model, part) per job. Same part structure and the same
# cells file as the other three models -- tools/sweep/grid.py is still the
# single source of truth -- but CoSDH is an OpenCOOD FORK, so it needs its own
# env, its own PYTHONPATH and its own driver.
#
#   sbatch --time=... tools/slurm/cosdh_sweep.sbatch <part>
#
# Results are copied back AFTER EVERY CELL; a failed cell is reported and the
# job continues.

set -uo pipefail

PART="$1"
REPO="$HOME/Fault-Injectors"
SEED=1234
CKPT=/datasets/eemcs/ps/cv/opencood/cosdh_checkpoints/opv2v_cosdh
DEST="$REPO/results/sweep/cosdh"
mkdir -p "$DEST" "$REPO/logs"

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "=== cosdh part=$PART node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# path MUST contain both 'OPV2V' and 'test': inference.main() asserts it.
DATA="$SCRATCH/OPV2V/test"
mkdir -p "$DATA"
rsync -a --exclude='*.png' /datasets/eemcs/ps/cv/opencood/opv2v/test/ "$DATA/"
echo "staged $(find "$DATA" -name '*.pcd' | wc -l) pcd files"

cp -r "$CKPT" "$SCRATCH/model"
sed -i "s|^root_dir:.*|root_dir: '$DATA'|"         "$SCRATCH/model/config.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$DATA'|" "$SCRATCH/model/config.yaml"
sed -i "s|^test_dir:.*|test_dir: '$DATA'|"         "$SCRATCH/model/config.yaml"
grep -E "^(root_dir|validate_dir|test_dir): '$DATA'" "$SCRATCH/model/config.yaml" \
    | wc -l | grep -qx 3 || { echo "FATAL: config paths not rewritten"; exit 1; }

source $(conda info --base)/etc/profile.d/conda.sh
conda activate cosdh
# PYTHONPATH FIRST: this env carries an inherited editable VANILLA OpenCOOD.
export PYTHONPATH="$HOME/cosdh-official:$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -c "import opencood; assert 'cosdh-official' in opencood.__file__; print('opencood ->', opencood.__file__)" || exit 1

CELLS=$("$REPO/.venv-hpc/bin/python" "$REPO/tools/sweep/grid.py" cosdh "$PART") \
    || { echo "FATAL: grid.py failed"; exit 1; }
echo "cells:"; echo "$CELLS" | sed 's/^/  /'

FAILED=0
while IFS='|' read -r OUT SPEC; do
    [ -n "$OUT" ] || continue
    echo ""; echo "################ cosdh/$OUT ################"
    if python tools/cosdh_inference.py \
        --model_dir "$SCRATCH/model" \
        --fi_spec "$SPEC" --fi_seed "$SEED" \
        --fi_out "$SCRATCH/out/$OUT"; then
        mkdir -p "$DEST/$OUT"
        rsync -a "$SCRATCH/out/$OUT/" "$DEST/$OUT/"
        echo "cell $OUT done -> $DEST/$OUT"
    else
        echo "CELL FAILED: cosdh/$OUT (exit $?)"
        FAILED=1
        mkdir -p "$DEST/$OUT"
        rsync -a "$SCRATCH/out/$OUT/" "$DEST/$OUT/" 2>/dev/null || true
    fi
done <<< "$CELLS"

echo ""
echo "=== SWEEP PART DONE model=cosdh part=$PART failed_cells=$FAILED ==="
exit $FAILED
