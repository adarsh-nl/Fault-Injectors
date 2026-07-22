#!/bin/bash
# The paper's headline figure: accuracy against log2(bytes).
#
#   CKPT=$HOME/w2c-results/..._train/checkpoints/best.pt \
#     sbatch w2cbench/slurm/curve_array.sbatch dataset=opv2v_lidar
#
# Each array task traces the whole bandwidth curve under ONE fault condition,
# so the output is a family of curves rather than a single line: index 0 is the
# clean reproduction, the rest are the same curve displaced by a fault.
#
# That displacement is the result this package exists to produce. It is not
# "AP fell by x" but "the entire performance-bandwidth frontier moved, in this
# direction" -- and a benchmark reporting AP alone cannot express it.
#
# Note that each bandwidth group carries its own clean reference. Scoring a
# faulted run at one budget against a clean run at another would attribute the
# budget reduction to the fault, and the inflation grows as the budget shrinks.
#SBATCH --job-name=w2c-curve
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
RESULTS="${RESULTS:-$HOME/w2c-results}"
mkdir -p "$RESULTS"

module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8 || true
export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

FAULTS=(none pose_error agent_drop lidar_weather)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAULT="${FAULTS[$TASK_ID]}"

# Six budgets, roughly a decade apart on the log2 axis. BudgetSelector converts
# each to a cell count using the model's own feature width, and a test asserts
# the real MessageChannel never exceeds it.
BUDGETS='[{kind: budget, budget_bytes: 1024},
          {kind: budget, budget_bytes: 4096},
          {kind: budget, budget_bytes: 16384},
          {kind: budget, budget_bytes: 65536},
          {kind: budget, budget_bytes: 262144},
          {kind: budget, budget_bytes: 1048576}]'

CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then
    CKPT_ARG="--checkpoint $CKPT"
fi

srun "$VENV/bin/python" -m w2cbench.scripts.benchmark \
    faults="$FAULT" taps=comm \
    "bandwidth_sweep=$BUDGETS" \
    results_dir="$RESULTS" \
    $CKPT_ARG "$@"
