#!/bin/bash
# Train the CoBEVT LiDAR track on the UT EEMCS HPC.
#
#   sbatch cobevtbench/slurm/train_lidar.sbatch \
#          dataset=opv2v_lidar model=cobevt_lidar trainer=lidar
#
# The LiDAR track is a single detection model (no dynamic/static split), so
# one job trains it. FuseBEVT is the same module the camera track uses --
# that is the claim paper Table 2 makes.
#SBATCH --job-name=cobevt-train-lidar
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
RESULTS="${RESULTS:-$HOME/cobevt-results}"
SCRATCH="/local/$USER/${SLURM_JOB_ID:-manual}"
mkdir -p "$SCRATCH" "$RESULTS"

module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8 || true

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch torchvision einops pyyaml numpy \
        matplotlib scipy tensorboard
fi

export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

srun "$VENV/bin/python" -m cobevtbench.scripts.train \
    dataset=opv2v_lidar model=cobevt_lidar trainer=lidar \
    results_dir="$RESULTS" \
    "$@"

rm -rf "$SCRATCH"
