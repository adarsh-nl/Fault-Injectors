#!/bin/bash
# Name the backward-only op that survives the SSD scan.
#   sbatch corabench/slurm/anomaly_ssd.sbatch
#
# 549494 (SSD scan, dt_max=0.2) did what the rewrite was for: THE FORWARD IS
# CLEAN. Loss bounded and steady across all 34 steps -- at the abort,
# total=3.736 cls=2.977 reg=0.312 align=0.135, essentially step 0's values.
# Every earlier abort had cls 38-69. ZERO of 50 forward tap locations record a
# single non-finite value.
#
# It still aborts at batch 33 on a non-finite GRADIENT. Backward-only, and
# disconnected from forward magnitude.
#
# SCALE-INDEPENDENT: the GradScaler fell from 6.55e4 to 6.10e-5 -- a factor of
# 1.07e9 -- and the gradient stayed non-finite. Plain fp16 overflow would have
# been cured somewhere in those nine orders of magnitude. Only 4 of 34 steps
# landed.
#
# THE SUBTREE IS ALREADY KNOWN, from grad_norm_by_module:
#   step  5   local_head=inf  encoder=nan   lc=0.891  lc_head=0.393
#   step 32   local_head=nan  encoder=nan   pac=nan   lc=0.354  lc_head=0.617
#   step 33   local_head=nan  encoder=nan   pac=nan   lc=0.528  lc_head=0.686
#
# lc -- the module holding the CSSM and its cumsum -- is FINITE and small at
# exactly the aborting steps. The scan is NOT the source, and the
# reverse-cumsum candidate is dead: cumsum's backward is ADDITIVE (it sums
# <=64 terms per chunk), so it amplifies but cannot manufacture a non-finite
# value from finite inputs the way 1/E^2 did at 1e26 from one element -- and
# an additive amplifier is scale-DEPENDENT, which this is not.
#
# The live subtree is local_head -> pac -> encoder. Note step 5 shows
# local_head=inf (an overflow) with encoder=nan downstream, consistent with
# inf-inf or inf*0 mixing. That subtree carries PAC's _decode_params, whose
# asin has derivative 707 at the epsilon bound (96fbb2a), against
# head/cls_logits reaching 330 in this run.
#
# SO THIS RUN IS EXPECTED TO NAME SOMETHING IN local_head/PAC, PLAUSIBLY
# AsinBackward0 FOR A THIRD TIME. If it does, asin is again the reporter and
# the question becomes what makes its upstream large -- reg[:, 6] is an
# unconstrained prediction that saturates the clamp routinely (RECON-5), and
# the decode itself is the unfixed defect. If it names something else in that
# subtree, that is the new lead.
#
# IDENTICAL to 549494 apart from detect_anomaly=true. Expect a raise around
# batch 1-2 (grad_norm is already nan at step 1), so 1-2 minutes. The failing
# step will NOT be in metrics.csv -- the row is written after backward -- so
# the traceback is its record; steps before it survive via explog.close().
#SBATCH --job-name=cora-anomssd
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
#SBATCH --exclude=hpc-node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=anomssd-%j.out
#SBATCH --error=anomssd-%j.err

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
    trainer.detect_anomaly=true \
    taps=stats \
    results_dir="$HOME/cora-anomssd"
