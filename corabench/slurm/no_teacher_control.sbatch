#!/bin/bash
# Control: is the LC blowup align-driven or intrinsic?
#   sbatch corabench/slurm/no_teacher_control.sbatch
#
# IDENTICAL to 549227 in every respect -- same seed (2026), scenarios, lr
# (1e-3), grad_clip (10), grid, amp=true, log_every=1, taps=stats -- with ONE
# variable moved: model.teacher_enabled=false. With no teacher there is no
# L_align term at all (losses.py falls back to `out["f_out"].sum() * 0.0`),
# so the objective is w_local*L_local + w_lc*L_lc + w_pac*L_pac + u_reg*L_u.
# This is the same control that diagnosed the shared-weight teacher bug.
#
# PREDICTION: this will NOT stop the blowup. Three reasons from 549227:
#   1. At step 0 align was 4.83e-06 -- effectively 0% of the objective -- so
#      the optimizer step taken AT step 0 carried essentially no align
#      gradient. That step still produced lc/output amax 879.5 (2660x).
#      Align cannot have caused a blowup it was not part of the gradient for.
#   2. lc/teacher_feature moved only 0.344 -> 1.185 (3.4x) while lc/output
#      moved 2660x. The teacher is nearly stationary, as an EMA at momentum
#      0.999 must be after one step.
#   3. MSE to a stop-gradient target is CONTRACTIVE: its gradient pulls f_out
#      TOWARD f_teacher. It cannot push f_out away from a near-zero target.
#
# So align at 232 is a REPORTER of the blowup, not its cause, and the
# discriminating outcome is:
#   lc/output still explodes  -> confirmed intrinsic to LC/CSSM; next
#                                investigation is the LC branch itself and
#                                the cssm.py:193 fp16 cast-back edge
#   lc/output stays bounded   -> the reasoning above is wrong somewhere and
#                                align does drive it after all
#
# Expect the run to end on the scale-floor abort again unless LC stabilises;
# 549227 reached step 115 in ~4 min before aborting at scale 1.49e-08.
#SBATCH --job-name=cora-noteach
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
#SBATCH --exclude=hpc-node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=noteach-%j.out
#SBATCH --error=noteach-%j.err

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
    model.teacher_enabled=false \
    taps=stats \
    results_dir="$HOME/cora-noteacher"
