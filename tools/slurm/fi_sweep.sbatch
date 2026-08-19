#!/bin/bash
#SBATCH --job-name=fi-sweep
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Generic sweep job: runs the cells of one (model, part) through
# tools/fi_inference.py, one inference pass per cell, at the model's SHIPPED
# wild_setting (no forcing). Results are copied back AFTER EVERY CELL, so a
# wall or crash loses at most the in-flight cell (lesson of job 557245).
#
#   sbatch --time=... tools/slurm/fi_sweep.sbatch <model> <part>
#   e.g.   sbatch --time=12:00:00 tools/slurm/fi_sweep.sbatch cobevt a
#
# Cells come from tools/sweep/grid.py (the single source of truth); a cell
# that fails is reported as "CELL FAILED" and the job continues, so one bad
# cell cannot take down the rest of the grid.

set -uo pipefail          # NOT -e: per-cell failure is handled explicitly

MODEL="$1"; PART="$2"
REPO="$HOME/Fault-Injectors"
SEED=1234

NS=opencood
CODE_ROOT="$HOME/opencood-official"
case "$MODEL" in
  cobevt)     MODEL_DIR="$HOME/opencood-eval/cobevt"
              DATA_SRC="/datasets/eemcs/ps/cv/opencood/v2xset/test" ;;
  # where2comm checkpoint is V2XSET-trained (md5-identical to
  # point_pillar_where2comm_v2xset.zip); paired with V2XSet test.
  where2comm) MODEL_DIR="$HOME/opencood-eval/where2comm"
              DATA_SRC="/datasets/eemcs/ps/cv/opencood/v2xset/test" ;;
  v2xvit)     MODEL_DIR="/datasets/eemcs/ps/cv/opencood/v2xset_checkpoints/v2x-vit"
              DATA_SRC="/datasets/eemcs/ps/cv/opencood/v2xset/test"
              NS=v2xvit
              CODE_ROOT="$HOME/v2xvit-official" ;;
  *) echo "FATAL: unknown model $MODEL"; exit 1 ;;
esac
DEST="$REPO/results/sweep/$MODEL"
mkdir -p "$DEST" "$REPO/logs"

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "=== model=$MODEL part=$PART node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# ── stage (LiDAR+yaml only) ─────────────────────────────────────────────
mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' "$DATA_SRC/" "$SCRATCH/data/"
echo "staged $(find "$SCRATCH/data" -name '*.pcd' | wc -l) pcd files"

cp -r "$MODEL_DIR" "$SCRATCH/model"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/model/config.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/model/config.yaml"
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/model/config.yaml" \
    | wc -l | grep -qx 2 \
    || { echo "FATAL: config data paths not rewritten"; exit 1; }

source activate opencood-official
export PYTHONPATH="$CODE_ROOT:$REPO:${PYTHONPATH:-}"
cd "$CODE_ROOT"

# ── cells from the grid (generated fresh, no drift) ─────────────────────
CELLS=$("$REPO/.venv-hpc/bin/python" "$REPO/tools/sweep/grid.py" "$MODEL" "$PART") \
    || { echo "FATAL: grid.py failed"; exit 1; }
echo "cells:"; echo "$CELLS" | sed 's/^/  /'

FAILED=0
while IFS='|' read -r OUT SPEC; do
    [ -n "$OUT" ] || continue
    echo ""
    echo "################ $MODEL/$OUT ################"
    if python "$REPO/tools/fi_inference.py" \
        --model_dir "$SCRATCH/model" \
        --fusion_method intermediate \
        --fi_spec "$SPEC" \
        --fi_ns "$NS" \
        --fi_seed "$SEED" \
        --fi_out "$SCRATCH/out/$OUT"; then
        mkdir -p "$DEST/$OUT"
        rsync -a "$SCRATCH/out/$OUT/" "$DEST/$OUT/"
        echo "cell $OUT done -> $DEST/$OUT"
    else
        echo "CELL FAILED: $MODEL/$OUT (exit $?)"
        FAILED=1
        # copy whatever partial output exists, for the post-mortem
        mkdir -p "$DEST/$OUT"
        rsync -a "$SCRATCH/out/$OUT/" "$DEST/$OUT/" 2>/dev/null || true
    fi
done <<< "$CELLS"

echo ""
echo "=== SWEEP PART DONE model=$MODEL part=$PART failed_cells=$FAILED ==="
exit $FAILED
