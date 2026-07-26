#!/bin/bash
# Name the op producing the non-finite backward that survives the asin fix.
#   sbatch corabench/slurm/anomaly_probe2.sbatch
#
# Successor to 549416, which named AsinBackward0. That is fixed (96fbb2a) and
# a DIFFERENT non-finite gradient is now dominant.
#
# READ THIS BEFORE INTERPRETING THE RUN -- 549373 AND 549412 DID NOT DIVERGE.
# They are the same trajectory truncated at different points. Side by side:
#
#   step   549373 total / logit_amax     549412 total / logit_amax
#     0        4.096  /  8.281              4.096  /  8.281
#     1        4.661  / 33.78               4.660  / 33.75
#     2        10.43  / 270.8               10.45  / 271.2
#     3        4.637  / 34.91               4.634  / 34.72
#     5        5.178  / 44.72               5.183  / 45.12
#
# Skip rate 43/48 (90%) vs 30/34 (88%). grad_norm finite at step 0 in BOTH
# (13.77) and at step 5 in BOTH (123.7 / 125.6). So:
#   * steps LANDED in the probe too -- the floor never gated landing, it only
#     decides when the run aborts (48 steps at 1e-8, 34 at 1e-4)
#   * the probe's logits were NOT stuck at 8; they hit 270.8 at step 2
#   * the asin fix moved the trajectory by <0.5%, consistent with a 1e-6
#     change to a clamp bound. It fixed a real singularity; it is not what
#     let a step land
#
# scale_floor is therefore raised to 1.0e-4 here as requested, but it is
# INCONSEQUENTIAL for this run: the non-finite gradient appears at step 1,
# roughly 30 backoffs before either floor could fire.
#
# WHERE THE EXPLOSION IS. Per-step amax, corrected for taps that emit twice
# per step (head/cls_logits and pac/output_cls emit per branch/frame):
#
#   step  lc/ssm_delta  lc/ssm_out  lc/output  head/cls_logits  pac/output_cls
#     0        0.318        3.62       0.288        7.00            8.28
#     1        4.85         192       31.4         33.75            8.02
#     2       11.1         1612      297.8        271.25            7.90
#
# The LC/CSSM branch and the detection heads it feeds explode together; PAC
# output stays flat at ~8. dt_init fixed delta AT INIT (0.318) and did not
# hold it under a landed update (4.85 at step 1, 11.1 at step 2).
#
# THE TWO SUSPECTS, settled from the taps:
#   * fp32-island 1/(1-p): NOT REACHABLE. fusion/recalibrated_scores amax
#     never exceeds 0.605 across the whole run, so 1/(1-p) stays in
#     [1.02, 2.53]. The A4 recalibration sigmoid(cls)*sigmoid(-U) caps p well
#     below 1 -- U mean is -0.57, so sigmoid(-U) ~ 0.64 -- and the focal clamp
#     is never approached even at head logits of 271.
#   * b/E division in the scan: REACHABLE. 2.9-3.4% of logE entries sit at the
#     -30 floor where E = 9.4e-14, and d(b/E)/dE = -b/E^2 carries 1/E^2 =
#     1.1e26. With b = (delta*x) (x) B at delta 4.85 and x ~ 22 that is ~1e28
#     in fp32, and the island exit casts to fp16 (max 65504).
#
# CAVEAT ON WHAT THE ANOMALY WILL NAME. It fires on the FIRST non-finite in
# backward traversal order -- nearest the loss -- which need not be the root.
# At step 2, ssm_out is 1612 and the loss scale is ~16384, so ordinary
# gradients through the LC branch reach 1e7-1e8 and overflow the fp16 cast
# with no division involved. If the named op is a plain multiply or a cast
# rather than a division, that is evidence of magnitude overflow, not of the
# b/E hazard -- read which SUBTREE it lands in, not just the op name.
#
# Identical to 549412 except detect_anomaly=true. Expect a raise at step 1-2,
# so ~1-2 minutes. The failing step will NOT be in metrics.csv (the row is
# written after backward); the traceback is its record.
#SBATCH --job-name=cora-anom2
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
#SBATCH --exclude=hpc-node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=anom2-%j.out
#SBATCH --error=anom2-%j.err

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
    trainer.detect_anomaly=true \
    model.cssm.dt_init=mamba \
    taps=stats \
    results_dir="$HOME/cora-anom2"
