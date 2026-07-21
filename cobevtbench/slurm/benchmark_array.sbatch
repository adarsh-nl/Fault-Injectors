#!/bin/bash
# CoBEVT fault sweep as a SLURM job array: one array task per fault family.
#
#   # smoke test the whole pipeline (CPU, synthetic, untrained)
#   sbatch --partition=ps,main-cpu cobevtbench/slurm/benchmark_array.sbatch
#
#   # the camera-dropout reproduction on a trained model (GPU)
#   CKPT=$HOME/cobevt-results/..._train/checkpoints/best.pt \
#     sbatch cobevtbench/slurm/benchmark_array.sbatch \
#            dataset=opv2v_camera model=cobevt_camera_dynamic
#
# Why an array over fault families rather than one long job
#   Conditions are independent by construction: each rebuilds the dataset
#   around a fresh fault bridge and never touches the model weights. One
#   failed family does not lose the rest, and the families run in parallel.
#
# Edit FAULTS and --array together: the array index selects a family below.
#SBATCH --job-name=cobevt-bench
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out

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
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# One fault family per array index. Keep this list and --array in sync.
FAULTS=(camera_dropout calibration_error pose_error agent_drop latency comm_stress)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAULT="${FAULTS[$TASK_ID]}"
echo "array task $TASK_ID -> faults=$FAULT"

# CKPT is optional: without it the sweep runs on an untrained model, which
# exercises the plumbing and the sweep shape but yields meaningless metrics.
CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then
    CKPT_ARG="--checkpoint $CKPT"
fi

srun "$VENV/bin/python" -m cobevtbench.scripts.benchmark \
    faults="$FAULT" taps=stats \
    results_dir="$RESULTS" \
    $CKPT_ARG "$@"

rm -rf "$SCRATCH"
