#!/bin/bash
# LGCP-Bench fault sweep as a SLURM job array (one array task per condition).
#
#   # CPU, native backbone, synthetic data -- smoke test the whole pipeline
#   sbatch lgcpbench/slurm/benchmark_array.sbatch
#
#   # GPU, paper model on OPV2V
#   sbatch --partition=ps,main-gpu --gres=gpu:1 \
#          lgcpbench/slurm/benchmark_array.sbatch \
#          model=where2comm dataset=opv2v faults=comm_stress \
#          model.hypes_yaml=$HOME/OpenCOOD/opencood/hypes_yaml/point_pillar_where2comm.yaml \
#          model.checkpoint=$HOME/weights/where2comm.pth
#
# Why an array rather than one long job
#   Conditions are independent by construction: each rebuilds the dataset
#   around a fresh fault bridge and never touches the model. The Delta_g x
#   paradigm x fault grid therefore parallelises perfectly, and one failed
#   condition does not lose the rest.
#SBATCH --job-name=lgcp-bench
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
RESULTS="${RESULTS:-$HOME/lgcp-results}"
SCRATCH="/local/$USER/${SLURM_JOB_ID:-manual}"
mkdir -p "$SCRATCH" "$RESULTS"

module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8 || true

# Which Python: the OpenCOOD conda env when a paper model is requested,
# otherwise the core venv. Selecting the wrong one is the most likely
# operator error here, so it is decided from the arguments rather than left
# to the submitter.
PYTHON="$VENV/bin/python"
if printf '%s\n' "$@" | grep -qE '^model=(where2comm|cobevt|coalign)$'; then
    ENV_NAME="${ENV_NAME:-opencood-py37}"
    module load miniconda3 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
        echo "ERROR: a paper model was requested but conda env '${ENV_NAME}'" \
             "does not exist. Build it first:" >&2
        echo "  sbatch lgcpbench/slurm/opencood_env.sbatch" >&2
        exit 1
    fi
    conda activate "$ENV_NAME"
    PYTHON="$(command -v python)"
    echo "== using OpenCOOD env: $ENV_NAME"
else
    if [ ! -d "$VENV" ]; then
        python3 -m venv "$VENV"
        "$VENV/bin/pip" install --upgrade pip
        "$VENV/bin/pip" install torch pyyaml numpy matplotlib tensorboard
    fi
    echo "== using core env: $VENV"
fi

cd "$REPO"
export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # required for deterministic cuBLAS
export TMPDIR="$SCRATCH"

# One array task per Delta_g value (Fig. 3 / Table II). Override by editing
# --array and this list together.
DELTA_G=(0.05 0.075 0.1 0.125)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DG="${DELTA_G[$TASK_ID]}"

echo "== array task $TASK_ID : delta_g=$DG"
echo "== args: $*"

srun "$PYTHON" -m lgcpbench.scripts.benchmark \
    results_dir="$RESULTS" \
    lgcp.confidence.delta_g="$DG" \
    taps=stats \
    "$@"

echo "== done; results under $RESULTS"
