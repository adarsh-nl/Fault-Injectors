#!/bin/bash
# Full epoch with Delta bounded at runtime. The chain should finally hold.
#   sbatch corabench/slurm/full_epoch_bounded.sbatch
#
# 549412 reached 34 steps and 549373 reached 48 on the SAME trajectory (they
# did not diverge; the floor only decides when the abort fires). Both
# exploded identically:
#
#   step  Delta amax  ssm_out  lc/output  head_logits  saturated
#     0      0.318      3.62      0.288       7.00        2.9%
#     1      4.85        192     31.4        33.75        3.4%
#     2     11.1       1612    297.8       271.25          --
#
# ssm_out moved 445x while clamp saturation moved 2.9% -> 3.4%, so the
# degeneracy was NOT the path: b = (Delta * x) (x) B scales linearly in Delta
# and everything downstream followed. dt_init fixed where Delta STARTS;
# nothing held it. 028d559 adds the runtime bound.
#
# ONE VARIABLE vs 549412: model.cssm.dt_max=0.2, applied as
# dt_max * tanh(Delta/dt_max). Soft, not clamp(max=...) -- clamp is
# backward-dead above the bound, the same trap as the fp16 focal clamp and
# the asin clamp. 0.2 rather than 0.1 because [0.001, 0.1] is Mamba's INIT
# range, not a runtime ceiling, and 0.1 would distort its top by 24% against
# under 2% at 0.2.
#
# PREDICTION -- this run tests it, check in this order:
#   1. lc/ssm_delta amax pinned at <= 0.2 every step (the bound is applied
#      each forward, so this is near-tautological; it is the control).
#   2. lc/ssm_out stays SINGLE DIGITS. Extrapolating step 0 linearly in
#      Delta: 3.62 * (0.2/0.318) ~ 2.3, i.e. O(1), against 192 at step 1.
#   3. head/cls_logits stays ~8-15, not 271.
#   4. AsinBackward0 never fires: the gradient reaching asin stays small, so
#      the reporter disappears with its cause.
#   5. The epoch COMPLETES 210 steps and validation runs.
#
# IF ssm_out STILL EXPLODES, b ∝ Delta was not the whole story and RECON-4
# option 1 (the divide-free SSD scan) moves up. That is a real outcome, not a
# failure -- the bound would have been shown insufficient rather than wrong.
#
# THE AP IS STILL A THROWAWAY. RECON-5: yaw is one sine channel, arcsin covers
# half the circle, sin(D) = sin(pi-D) makes heading 180-degree ambiguous, and
# cpbench/data/postprocessing.py decodes with the same arcsin outside autograd
# -- so VALIDATION AP IS COMPUTED ON SATURATED, DIRECTION-AMBIGUOUS YAW. The
# decode fix is Nreg 7->8 across shared cpbench code touching all five paper
# packages and does not ride into this run. Quote any AP as pre-decode-fix.
#
# READ THE RESULT IN THIS ORDER:
#   1. does the epoch complete (210 steps)   2. does validation run
#   3. peak GPU memory and epoch wall clock  4. AP, as a completes-signal only
#SBATCH --job-name=cora-bounded
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
#SBATCH --exclude=hpc-node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=bounded-%j.out
#SBATCH --error=bounded-%j.err

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
    trainer.scale_floor=1.0e-4 \
    model.cssm.dt_init=mamba \
    model.cssm.dt_max=0.2 \
    taps=stats \
    results_dir="$HOME/cora-bounded"
