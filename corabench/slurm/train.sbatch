#!/bin/bash
# Train CoRA on the UT EEMCS HPC (submit from a head node with `sbatch`).
#   sbatch corabench/slurm/train.sbatch dataset=opv2v
# Notes
#   * submit ONLY from hpc-head1/hpc-head2; never ssh into compute nodes.
#   * datasets are read from /deepstore/datasets; heavy scratch goes to
#     /local on the node (cleared after the job).
#SBATCH --job-name=cora-train
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
SCRATCH="/local/$USER/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"

# software environment (see https://hpc.wiki.utwente.nl/eemcs-hpc:software)
module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch torchvision pyyaml tensorboard matplotlib numpy
fi

cd "$REPO"
export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

srun "$VENV/bin/python" -m corabench.scripts.train \
    results_dir="$HOME/cora-results" \
    trainer.num_workers=8 \
    "$@"

rm -rf "$SCRATCH"
