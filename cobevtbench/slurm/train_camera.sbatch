#!/bin/bash
# Train the CoBEVT camera track on the UT EEMCS HPC.
#
#   # smoke test on synthetic data, one GPU
#   sbatch cobevtbench/slurm/train_camera.sbatch trainer=smoke
#
#   # the paper's dynamic (vehicle) model on OPV2V
#   sbatch cobevtbench/slurm/train_camera.sbatch \
#          dataset=opv2v_camera model=cobevt_camera_dynamic
#
#   # the static (road+lane) model -- trained SEPARATELY (assumption A8)
#   sbatch cobevtbench/slurm/train_camera.sbatch \
#          dataset=opv2v_camera model=cobevt_camera_static
#
# Why two training jobs
#   Dynamic and static are two independently trained models merged only at
#   inference (evaluation/merge.py). Reproducing the paper's IoU means
#   running this script twice with different `model=`, not one multi-head run.
#SBATCH --job-name=cobevt-train-cam
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
RESULTS="${RESULTS:-$HOME/cobevt-results}"
SCRATCH="/local/$USER/${SLURM_JOB_ID:-manual}"
mkdir -p "$SCRATCH" "$RESULTS"

module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8 || true

# Self-bootstrapping venv. einops and torchvision are mandatory for this
# package; a missing dep otherwise surfaces only once the job is running.
if [ ! -d "$VENV" ]; then
    echo "creating venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install torch torchvision einops pyyaml numpy \
        matplotlib scipy tensorboard
fi

export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # required for deterministic cuBLAS
# CoBEVT starts from an ImageNet-pretrained ResNet34. Compute nodes may lack
# egress, so the weights must be cached on a head node first:
#   python -c "import torchvision; torchvision.models.resnet34(weights='DEFAULT')"
# This points torchvision at that shared cache and fails loudly if it is cold.
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# Datasets are read-only under /deepstore; results go to $HOME, scratch to
# node-local /local. Never write to /deepstore.
srun "$VENV/bin/python" -m cobevtbench.scripts.train \
    results_dir="$RESULTS" \
    "$@"

echo "cleaning scratch $SCRATCH"
rm -rf "$SCRATCH"
