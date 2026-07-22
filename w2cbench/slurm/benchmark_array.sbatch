#!/bin/bash
# Where2comm fault sweep as a SLURM job array: one array task per fault family.
#
#   # smoke the whole pipeline (CPU, synthetic, untrained)
#   sbatch --partition=ps,main-cpu w2cbench/slurm/benchmark_array.sbatch
#
#   # the real thing, on a trained checkpoint
#   CKPT=$HOME/w2c-results/..._train/checkpoints/best.pt \
#     sbatch w2cbench/slurm/benchmark_array.sbatch dataset=opv2v_lidar
#
# Why an array over fault families rather than one long job
#   Conditions are independent by construction: each rebuilds the dataset
#   around a fresh fault bridge and never mutates the model weights. One failed
#   family does not lose the rest, and the families run in parallel.
#
# TWO THINGS TO KNOW BEFORE READING THE RESULTS
#
#   1. Without CKPT the model is untrained, and an untrained detection head
#      sits at the focal prior sigmoid(-4.59) = 0.010051 -- just ABOVE the
#      released threshold of 0.01. Selection saturates, the whole map is
#      transmitted, and the bandwidth column shows no compression. That is a
#      training diagnostic, not a bug. The CLI warns about it.
#
#   2. The `protocol` family is a PROVABLE no-op at the default rounds=1,
#      because nothing consumes a request map in a single round. Index 5 below
#      therefore sets rounds=3; without that it would report perfect
#      robustness by construction.
#
# Edit FAULTS and --array together: the array index selects a family below.
#SBATCH --job-name=w2c-bench
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
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# One fault family per array index. Keep this list and --array in sync.
FAULTS=(pose_error agent_drop latency lidar_weather comm_stress protocol)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAULT="${FAULTS[$TASK_ID]}"

# The protocol family is meaningless at K=1; see the header.
EXTRA=""
if [ "$FAULT" = "protocol" ]; then
    EXTRA="model.communication.rounds=3"
fi
echo "array task $TASK_ID -> faults=$FAULT $EXTRA"

CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then
    CKPT_ARG="--checkpoint $CKPT"
fi

srun "$VENV/bin/python" -m w2cbench.scripts.benchmark \
    faults="$FAULT" taps=comm \
    results_dir="$RESULTS" \
    $EXTRA $CKPT_ARG "$@"

rm -rf "$SCRATCH"
