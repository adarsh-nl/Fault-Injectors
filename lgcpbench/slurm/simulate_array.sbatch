#!/bin/bash
# Network-only latency sweep (paper Fig. 7: 5-30 CAVs), CPU-only.
#
#   sbatch lgcpbench/slurm/simulate_array.sbatch
#
# The control plane -- partitioning, grouping, leader election, Algorithm 2,
# the latency model -- consumes and produces plain data and needs no backbone,
# no dataset and no GPU. So the dense multi-CAV curve the paper reports from a
# CARLA/OpenCDA/NS3 co-simulation is reproduced here analytically, on CPU, in
# minutes. Trend only: this is not the co-simulator.
#SBATCH --job-name=lgcp-sim
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
RESULTS="${RESULTS:-$HOME/lgcp-results/simulate}"
mkdir -p "$RESULTS"

cd "$REPO"
export PYTHONPATH="$REPO"

N_CAVS=(5 10 15 20 25 30)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N="${N_CAVS[$TASK_ID]}"

echo "== array task $TASK_ID : n_cavs=$N"
srun "$VENV/bin/python" -m lgcpbench.scripts.simulate \
    --n-cavs "$N" --results-dir "$RESULTS" "$@"
