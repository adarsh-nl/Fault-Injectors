#!/bin/bash
# Run B: does correct Delta initialisation alone stabilise LC?
#   sbatch corabench/slurm/dt_init_fix.sbatch
#
# IDENTICAL to 549227 in every respect -- same seed (2026), scenarios, lr
# (1e-3), grad_clip (10), grid, amp=true, log_every=1, taps=stats, TEACHER ON
# -- with ONE variable moved: model.cssm.dt_init=mamba.
#
# That changes dt_proj.bias ONLY, from a constant -2.0 to the inverse softplus
# of dt ~ exp(U(log 1e-3, log 1e-1)). The weight is untouched: nn.Linear's
# default kaiming-uniform bound is 1/sqrt(fan_in) = d_inner**-0.5, which is
# already exactly Mamba's dt_init_std, and re-drawing it would consume global
# RNG and shift every module built afterwards -- making this a many-variable
# run. The dt sample uses a LOCAL generator for the same reason.
#
# MEASURED EFFECT at 549227's feature scale (Delta p50/p95/amax):
#     constant  0.128 / 0.523 / 2.42     horizon p50/p95  1.09 / 9.8 positions
#     mamba     0.011 / 0.126 / 1.65     horizon p50/p95  13.8 / 290 positions
#
# PREDICTIONS, all falsifiable from taps.csv:
#   1. lc/ssm_decay_horizon_p50 rises from ~1 to ~10-15 positions, p95 from
#      ~10 to ~300.
#   2. lc/ssm_logE_saturated FALLS but does NOT reach zero. Delta amax stays
#      1.65 because weight*x (amax 4.33) still dominates the bias in the tail,
#      so |dA| ~ 26 at the extreme still saturates a 64-chunk in two steps.
#   3. lc/ssm_logE_integrator becomes NON-ZERO for the first time (it read
#      exactly 0.0000 at every step on synthetic). Delta near the 1e-4 floor
#      with |A| = 1 puts logE above -0.01.
#   4. lc/output stays bounded across steps, instead of 0.331 -> 879.5.
#
# If (4) holds, Delta init was the cause and the clamp is secondary. If (4)
# fails while (1)-(3) hold, the Delta fix landed and was not sufficient: the
# clamp is load-bearing too and is the next single variable.
#
#SBATCH --job-name=cora-dtinit
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=dtinit-%j.out
#SBATCH --error=dtinit-%j.err

set -euo pipefail

REPO="$HOME/Fault-Injectors"
export CPBENCH_DATA_ROOT="${CPBENCH_DATA_ROOT:-/datasets/eemcs/ps/cv}"

module purge
unset PYTHONHOME PYTHONSTARTUP

cd "$REPO"
export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

DATA_ROOT=$(.venv-hpc/bin/python -c \
    "from cpbench.utils.paths import dataset_root; print(dataset_root('opv2v'))")
SOURCE=$(.venv-hpc/bin/python -c \
    "from cpbench.utils.paths import describe_source; print(describe_source())")
echo "resolver: dataset_root('opv2v') -> $DATA_ROOT"
echo "resolver: source                -> $SOURCE"
if [ ! -d "$DATA_ROOT/train" ] || [ ! -d "$DATA_ROOT/validate" ]; then
    echo "FATAL: resolved root has no train/ and validate/: $DATA_ROOT" >&2
    exit 1
fi

SCEN_TRAIN=$(ls "$DATA_ROOT/train"    | head -1)
SCEN_VAL=$(ls   "$DATA_ROOT/validate" | head -1)
echo "train scenario:    $SCEN_TRAIN"
echo "validate scenario: $SCEN_VAL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

srun .venv-hpc/bin/python -m corabench.scripts.train \
    dataset=opv2v \
    dataset.scenarios="[\"$SCEN_TRAIN\",\"$SCEN_VAL\"]" \
    trainer.epochs=1 trainer.num_workers=7 \
    trainer.log_every=1 \
    model.cssm.dt_init=mamba \
    taps=stats \
    results_dir="$HOME/cora-dtinit"
