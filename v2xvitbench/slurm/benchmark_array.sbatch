#!/bin/bash
# V2X-ViT fault sweep as a SLURM job array: one array task per fault family.
#
#   # smoke the whole pipeline (CPU, synthetic, untrained)
#   sbatch --partition=ps,main-cpu v2xvitbench/slurm/benchmark_array.sbatch \
#          model=v2xvit_tiny max_frames=8
#
#   # the real thing, on a trained checkpoint
#   CKPT=$HOME/v2xvit-results/..._train/checkpoints/best.pt \
#     sbatch v2xvitbench/slurm/benchmark_array.sbatch dataset=v2xset
#
# Why an array over fault families rather than one long job
#   Conditions are independent by construction: each rebuilds the dataset
#   around a fresh fault bridge and never mutates the model weights. One
#   failed family does not lose the rest, and the families run in parallel.
#
# BEFORE READING THE RESULTS: without CKPT the model is untrained, and an
# untrained detection head sits at the focal prior sigmoid(-4.59) = 0.0101,
# BELOW the released score threshold of 0.27 -- no boxes decode, every AP is
# zero, and only the plumbing is being exercised. The CLI warns about it.
#
# Edit FAULTS and --array together: the array index selects a family below.
#SBATCH --job-name=v2xvit-bench
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm-%x-%A_%a.out

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
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# One fault family per array index. Keep this list and --array in sync.
FAULTS=(pose_error latency agent_drop lidar_weather v2x_noise
        delay_encoding type_flip)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAULT="${FAULTS[$TASK_ID]}"
echo "array task $TASK_ID -> faults=$FAULT"

CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then
    CKPT_ARG="--checkpoint $CKPT"
fi

srun "$VENV/bin/python" -m v2xvitbench.scripts.benchmark \
    faults="$FAULT" taps=attention \
    results_dir="$RESULTS" \
    $CKPT_ARG "$@"

rm -rf "$SCRATCH"
