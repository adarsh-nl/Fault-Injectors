#!/bin/bash
# Fault-benchmark sweep as a SLURM job array: one array task per fault
# condition group, so the whole grid runs in parallel on the cluster.
#   sbatch corabench/slurm/benchmark_array.sbatch \
#       CKPT=$HOME/cora-results/.../checkpoints/best.pt dataset=opv2v
#SBATCH --job-name=cora-bench
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
CKPT="${CKPT:?pass CKPT=/path/to/best.pt}"

# one faults group per array index
FAULT_GROUPS=(pose_error latency comm_stress)
FAULTS="${FAULT_GROUPS[$SLURM_ARRAY_TASK_ID]}"

cd "$REPO"
export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8

srun "$VENV/bin/python" -m corabench.scripts.benchmark \
    --checkpoint "$CKPT" \
    results_dir="$HOME/cora-results" \
    faults="$FAULTS" taps=stats \
    "$@"
