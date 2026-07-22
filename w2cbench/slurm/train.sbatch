#!/bin/bash
# Train Where2comm on the UT EEMCS HPC.
#
#   # LiDAR track, OPV2V
#   sbatch w2cbench/slurm/train.sbatch dataset=opv2v_lidar
#
#   # camera track
#   sbatch w2cbench/slurm/train.sbatch \
#          model=where2comm_camera dataset=opv2v_camera
#
#   # multi-round, which the protocol fault family needs
#   sbatch w2cbench/slurm/train.sbatch model.communication.rounds=3
#
# One template for both tracks: only the encoder differs, so there is nothing
# track-specific for a job script to know.
#
# TRAINING IS CLEAN BY DESIGN. Checkpoint selection must not depend on a fault
# condition; faults belong to the benchmark entry point. Training does use the
# bandwidth curriculum (A17) regardless of the configured selector, which is
# what lets one checkpoint serve the whole accuracy-versus-bandwidth curve.
#SBATCH --job-name=w2c-train
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
RESULTS="${RESULTS:-$HOME/w2c-results}"
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
# Pretrained ResNet weights are fetched once on the camera track. A compute
# node without egress fails loudly at construction rather than hanging, but
# pre-warming the cache from a head node avoids the round trip entirely.
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

srun "$VENV/bin/python" -m w2cbench.scripts.train \
    results_dir="$RESULTS" \
    "$@"

rm -rf "$SCRATCH"
