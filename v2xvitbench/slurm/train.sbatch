#!/bin/bash
# Train V2X-ViT on the UT EEMCS HPC.
#
#   # the paper's configuration on V2XSet
#   sbatch v2xvitbench/slurm/train.sbatch dataset=v2xset
#
#   # quick smoke on synthetic data (also runs on ps,main-cpu)
#   sbatch --partition=ps,main-cpu v2xvitbench/slurm/train.sbatch \
#          model=v2xvit_tiny trainer=smoke
#
# TRAINING IS CLEAN BY DESIGN. Checkpoint selection must not depend on a
# fault condition, and the metadata fault plane is evaluation-only by
# construction (assumption A10); faults belong to the benchmark entry point.
#SBATCH --job-name=v2xvit-train
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
RESULTS="${RESULTS:-$HOME/v2xvit-results}"
SCRATCH="/local/$USER/${SLURM_JOB_ID:-manual}"
mkdir -p "$SCRATCH" "$RESULTS"

module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8 || true

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r "$REPO/requirements.txt"
    "$VENV/bin/pip" install -r "$REPO/requirements-bench.txt"
fi

export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"     # required by deterministic mode

srun "$VENV/bin/python" -m v2xvitbench.scripts.train \
    results_dir="$RESULTS" \
    "$@"

rm -rf "$SCRATCH"
