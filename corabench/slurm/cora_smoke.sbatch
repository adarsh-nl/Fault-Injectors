#!/bin/bash
#SBATCH --job-name=cora-smoke
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CoRA smoke run (HOLD until sweep parts a+c complete; user gates launch):
# 400 optimiser steps on REAL OPV2V train scenarios through the A1+B1 path
# (OpenCOOD's unmodified IntermediateFusionDataset + CoRABatchAdapter, in
# the opencood-official env). Gate: loss descends past step 14 (the
# historical divergence point) and does not diverge. Trajectory lands in
# results/cora_smoke/<jobid>/smoke_loss.csv.

set -euo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
N_SCEN=${N_SCEN:-10}            # smoke: subset of the 44 train scenarios
STEPS=${STEPS:-400}

echo "=== node $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# ── stage N_SCEN train scenarios, LiDAR+yaml only ───────────────────────
mkdir -p "$SCRATCH/data"
for S in $(ls /datasets/eemcs/ps/cv/opencood/opv2v/train | head -$N_SCEN); do
    rsync -a --exclude='*.png' \
        "/datasets/eemcs/ps/cv/opencood/opv2v/train/$S" "$SCRATCH/data/"
done
echo "staged $N_SCEN scenarios, $(find "$SCRATCH/data" -name '*.pcd' | wc -l) pcd files"

# hypes: the CoBEVT baseline's shipped preprocessing (identical pipeline),
# pointed at scratch. Only the dataset sections are consumed.
cp ~/opencood-eval/cobevt/config.yaml "$SCRATCH/hypes.yaml"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/hypes.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/hypes.yaml"
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/hypes.yaml" \
    | wc -l | grep -qx 2 || { echo "FATAL: hypes paths not rewritten"; exit 1; }

source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"

python -m corabench.scripts.train_opencood \
    --hypes "$SCRATCH/hypes.yaml" \
    --out "$SCRATCH/out" \
    --max_steps "$STEPS" \
    --batch_size 2 --lr 1e-3 --num_workers 8 --seed 2026 --amp

# ── smoke verdict: descent past step 14, no divergence ──────────────────
python - "$SCRATCH/out/smoke_loss.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
loss = [float(r["loss"]) for r in rows]
n = len(loss)
early = sum(loss[:10]) / max(len(loss[:10]), 1)
late = sum(loss[-50:]) / max(len(loss[-50:]), 1)
post14 = sum(loss[14:64]) / max(len(loss[14:64]), 1)
finite = all(l == l and abs(l) != float("inf") for l in loss)
skipped = sum(1 for r in rows if r["stepped"] == "0")
print("steps %d | mean loss first10 %.3f -> steps14-64 %.3f -> last50 %.3f"
      % (n, early, post14, late))
print("all losses finite: %s | skipped steps: %d" % (finite, skipped))
verdict = finite and late < early and skipped < n * 0.05
print("SMOKE VERDICT:", "PASS (descending, no divergence)" if verdict
      else "FAIL -- inspect before any full run")
PY

mkdir -p "$HOME/Fault-Injectors/results/cora_smoke/$SLURM_JOB_ID"
rsync -a "$SCRATCH/out/" "$HOME/Fault-Injectors/results/cora_smoke/$SLURM_JOB_ID/"
echo "results -> results/cora_smoke/$SLURM_JOB_ID"
