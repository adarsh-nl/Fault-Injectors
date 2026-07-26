#!/bin/bash
# Train CoRA on the UT EEMCS HPC. Submit from hpc-head1/hpc-head2 only.
#
#   sbatch train.sbatch dataset=opv2v
#
# Differences from corabench/slurm/train.sbatch, all of which matter for a
# multi-day run:
#   1. dataset paths come from $CPBENCH_DATA_ROOT, not a literal in a config
#   2. automatic resume from last.pt, so requeue after the wall clock does not
#      restart from epoch 0
#   3. --signal + --requeue so SLURM checkpoints cleanly instead of SIGKILL
#   4. OMP_NUM_THREADS pinned so 8 dataloader workers do not oversubscribe
#      8 allocated cores
#   5. results_dir is job-stable, so resume finds the previous run
#   6. venv is NOT built inside the job (compute nodes may lack egress);
#      build it once on the head node and fail loudly if it is missing

#SBATCH --job-name=cora-train
#SBATCH --partition=ps,main-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
VENV="${VENV:-$REPO/.venv-hpc}"
# One resolution point for data: CPBENCH_DATA_ROOT (see cpbench/utils/paths.py).
# On the UT EEMCS cluster the shared CV tree is /datasets/eemcs/ps/cv.
export CPBENCH_DATA_ROOT="${CPBENCH_DATA_ROOT:-/datasets/eemcs/ps/cv}"
DATA_ROOT="$CPBENCH_DATA_ROOT/opencood/opv2v"

# Job-stable results dir: reused across requeues so --resume can find last.pt.
RUN_TAG="${RUN_TAG:-cora_opv2v_baseline}"
RESULTS="$HOME/cora-results/$RUN_TAG"
mkdir -p "$RESULTS"

if [ ! -x "$VENV/bin/python" ]; then
    echo "FATAL: no venv at $VENV. Build it on a head node first:" >&2
    echo "  python3 -m venv $VENV && $VENV/bin/pip install -r $REPO/requirements-bench.txt" >&2
    exit 1
fi
if [ ! -d "$DATA_ROOT/train" ]; then
    echo "FATAL: $DATA_ROOT/train not found. Run verify_opv2v.py first." >&2
    exit 1
fi

module purge
module load nvidia/cuda-12.4 2>/dev/null || module load nvidia/cuda-11.8

# The UT python/3.13.14 module exports PYTHONHOME globally, and SLURM copies
# the submitting environment into the job. A stale PYTHONHOME makes any
# interpreter that is not the module's own die with
# "ModuleNotFoundError: No module named 'encodings'" before it runs a line.
# We call $VENV/bin/python by absolute path, so just clear the poison.
unset PYTHONHOME PYTHONSTARTUP

cd "$REPO"
export PYTHONPATH="$REPO"
# Must be exported BEFORE the process starts: seed_everything() uses
# os.environ.setdefault(), which is a no-op once CUDA has initialised.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# 8 cores total, 8 dataloader workers, so 1 thread each and none for the
# main process's intra-op pool. Without this, torch defaults to 8 threads
# per worker and the node thrashes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Resume from the most recent last.pt under this run tag, if one exists.
RESUME_ARGS=()
LAST_CKPT=$(ls -t "$RESULTS"/*/checkpoints/last.pt 2>/dev/null | head -1 || true)
if [ -n "$LAST_CKPT" ]; then
    echo "resuming from $LAST_CKPT"
    RESUME_ARGS=(--resume "$LAST_CKPT")
else
    echo "no prior checkpoint under $RESULTS, starting fresh"
fi

# Requeue on SIGUSR1 (sent 5 min before the wall clock) so the epoch's
# last.pt is the resume point rather than a truncated write.
trap 'echo "caught USR1, requeueing"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

srun "$VENV/bin/python" -m corabench.scripts.train \
    "${RESUME_ARGS[@]}" \
    dataset=opv2v \
    results_dir="$RESULTS" \
    trainer.num_workers=7 \
    "$@" &
wait
