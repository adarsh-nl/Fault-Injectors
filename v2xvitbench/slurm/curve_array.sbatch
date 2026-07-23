#!/bin/bash
# The severity curves: the paper's own robustness axes, one per array task.
#
#   CKPT=... sbatch v2xvitbench/slurm/curve_array.sbatch dataset=v2xset
#
# Task 0 sweeps pose error over the paper's sigma grid; task 1 sweeps
# latency; task 2 runs the correction-matrix sweep whose difference from
# task 0 at equal sigma is the warp-only share of pose sensitivity.
#SBATCH --job-name=v2xvit-curve
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
RESULTS="${RESULTS:-$HOME/v2xvit-results}"
mkdir -p "$RESULTS"

module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8 || true

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r "$REPO/requirements.txt"
    "$VENV/bin/pip" install -r "$REPO/requirements-bench.txt"
fi

export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

FAULTS=(pose_error latency correction_matrix)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAULT="${FAULTS[$TASK_ID]}"
echo "array task $TASK_ID -> faults=$FAULT"

CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then
    CKPT_ARG="--checkpoint $CKPT"
fi

srun "$VENV/bin/python" -m v2xvitbench.scripts.benchmark \
    faults="$FAULT" taps=stats \
    results_dir="$RESULTS" \
    $CKPT_ARG "$@"
