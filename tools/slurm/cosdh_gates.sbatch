#!/bin/bash
#SBATCH --job-name=cosdh-gates
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CoSDH gates 1-3, ONE job so gates 1 and 2 share a session and a node.
#
#   gate 1  clean baseline, UNWRAPPED       (vs paper 96.83 / 92.99)
#   gate 2  wrapped-null, same session      (d50/d70 vs gate 1)
#   gate 3  pose fire-check WITH magnitude, per-frame spread, clean-slot
#           integrity, and yaw-index verification
#
# No sweep cells are run here.

set -uo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
DEST="$HOME/Fault-Injectors/results/cosdh_gates/$SLURM_JOB_ID"
mkdir -p "$DEST"
trap 'rsync -a "$SCRATCH/out/" "$DEST/" 2>/dev/null || true; rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"
CKPT=/datasets/eemcs/ps/cv/opencood/cosdh_checkpoints/opv2v_cosdh

echo "=== cosdh gates | node $(hostname) | gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# ── stage: the path MUST contain both 'OPV2V' and 'test' ────────────────
# inference.main() does hypes['validate_dir'] = hypes['test_dir'] and then
# asserts "test" in validate_dir when the path contains "OPV2V". A generic
# $SCRATCH/data would trip that assert.
DATA="$SCRATCH/OPV2V/test"
mkdir -p "$DATA"
rsync -a --exclude='*.png' /datasets/eemcs/ps/cv/opencood/opv2v/test/ "$DATA/"
echo "staged $(find "$DATA" -name '*.pcd' | wc -l) pcd files -> $DATA"

cp -r "$CKPT" "$SCRATCH/model"
sed -i "s|^root_dir:.*|root_dir: '$DATA'|"         "$SCRATCH/model/config.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$DATA'|" "$SCRATCH/model/config.yaml"
sed -i "s|^test_dir:.*|test_dir: '$DATA'|"         "$SCRATCH/model/config.yaml"
grep -E "^(root_dir|validate_dir|test_dir): '$DATA'" "$SCRATCH/model/config.yaml" \
    | wc -l | grep -qx 3 || { echo "FATAL: config paths not rewritten"; exit 1; }

source $(conda info --base)/etc/profile.d/conda.sh
conda activate cosdh
# PYTHONPATH FIRST: the env carries an inherited editable vanilla OpenCOOD.
export PYTHONPATH="$HOME/cosdh-official:$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -c "import opencood; assert 'cosdh-official' in opencood.__file__; print('opencood ->', opencood.__file__)" || exit 1

OUT="$SCRATCH/out"; mkdir -p "$OUT"

echo ""; echo "######## GATE 1: clean baseline, UNWRAPPED ########"
python tools/cosdh_inference.py --model_dir "$SCRATCH/model" \
    --fi_spec 'null' --fi_out "$OUT/gate1_unwrapped"
rsync -a "$OUT/" "$DEST/" 2>/dev/null || true

echo ""; echo "######## GATE 2: wrapped-null, SAME session/node ########"
python tools/cosdh_inference.py --model_dir "$SCRATCH/model" \
    --fi_spec '{}' --fi_out "$OUT/gate2_wrapped_null"
rsync -a "$OUT/" "$DEST/" 2>/dev/null || true

echo ""; echo "######## GATE 3: pose fire-check + magnitude verification ########"
python "$REPO/tools/cosdh_gate3.py" --model_dir "$SCRATCH/model" \
    --out "$OUT/gate3_pose" --frames 60 --sigma_xy 0.2 --sigma_yaw 0.2
rsync -a "$OUT/" "$DEST/" 2>/dev/null || true

echo ""; echo "######## VERDICTS ########"
python "$REPO/tools/cosdh_gate_report.py" "$DEST"
echo "results -> $DEST"
