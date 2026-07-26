#!/bin/bash
# Re-run of 549333 (dt_init=mamba) with the scale floor lowered so the
# trajectory is actually observed.
#   sbatch corabench/slurm/dt_init_trajectory.sbatch
#
# WHY 549333 TOLD US ALMOST NOTHING. It did not abort on the scale floor. It
# died in the FIRST forward pass, 4 seconds in, exit 13 with an empty stderr:
# torch.quantile refuses inputs above 2**24 elements, and the scan's
# diagnostic horizon tensor is Bt*L*D*N = 2*8800*64*16 = 18.0M on this grid
# (only 1.18M on the synthetic grid, which is why every local check passed).
# lc/ssm_delta is emitted BEFORE the scan call and the band taps AFTER it, so
# ssm_delta was the last thing ever written -- which is exactly what the
# bundle showed. Fixed by using kthvalue instead of quantile, with a
# regression test at 16.78M elements.
#
# The one real result from 549333 stands: ssm_delta amax 1.150 -> 0.318, into
# Mamba's range. dt_init reaches dt_proj and works.
#
# IDENTICAL to 549333 with ONE variable changed: trainer.scale_floor=1e-8
# (was the raised 1e-4 default). Deliberately diagnostic -- a floor that low
# lets the run continue through a dead scaler, which is the wrong trade for a
# production run and the right one here, because the trajectory IS the result.
#
# Also note the abort now fires AFTER the TrainRecord is written. In 549333's
# predecessor the raise preceded log_train, so an early abort produced a
# bundle with zero train rows: the instrument fired before the measurement it
# existed to preserve. The step that triggers an abort is now always in
# metrics.csv. With log_every=1 that means every step is captured.
#
# PREDICTIONS -- check in this order, from step 0:
#   1. lc/ssm_delta amax stays ~0.32 across steps. This tests whether the fix
#      HOLDS UNDER TRAINING rather than only at init; 549333 only ever showed
#      step 0. If it climbs back toward 1.1+, dt_init is not sufficient.
#   2. lc/ssm_logE_saturated drops well below 0.46. At delta amax 0.318 the
#      largest |A| = 16 gives dA ~ 5.1, so logE reaches -30 at position ~6 of
#      each 64-chunk instead of position ~2; at the median delta it does not
#      reach it at all within a chunk. Expect a large drop, plausibly near 0.
#   3. lc/ssm_logE_integrator may become non-zero for the first time (it read
#      exactly 0.0000 at every synthetic step). Delta near the 1e-4 floor with
#      |A| = 1 puts logE above -0.01.
#   4. lc/output -- THE QUESTION. Bounded across steps means dt_init was the
#      cause and the clamp is secondary. Still exploding (549227: 0.331 ->
#      879.5 in one step) means the delta fix landed and was not sufficient,
#      and the clamp / b-over-E form is load-bearing and next.
#
# Nothing about the clamp, dt_proj or the scan form is changed here.
#SBATCH --job-name=cora-dttraj
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
#SBATCH --exclude=hpc-node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=dttraj-%j.out
#SBATCH --error=dttraj-%j.err

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
    trainer.scale_floor=1.0e-8 \
    model.cssm.dt_init=mamba \
    taps=stats \
    results_dir="$HOME/cora-dttraj"
