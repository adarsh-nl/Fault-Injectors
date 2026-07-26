#!/bin/bash
# Name the op producing the backward-only NaN in 549373.
#   sbatch corabench/slurm/anomaly_probe.sbatch
#
# 549373 settled the main question: dt_init=mamba keeps the loss BOUNDED and
# steady for 47 steps (cls ~2.70, reg ~0.24, align ~0.013, pac ~0.92, total
# ~3.19), with no align spike and no lc/output explosion. Delta init was the
# cause of the LC blowup; the teacher was a bystander.
#
# It then aborted on a DIFFERENT defect the fix unmasked: from batch 31,
# grad_norm_by_module reads encoder=nan and local_head=nan while pac, lc,
# lc_head and adaptive stay FINITE -- with a finite loss (3.19), a healthy
# logit_amax of 12-15, and no non-finite forward tap anywhere. Finite forward,
# finite loss, non-finite gradient. The taps observe activations and are
# structurally blind to this.
#
# WHAT WAS ALREADY RULED IN AND OUT, by direct measurement of the loss code:
#   * clamp() passes ZERO gradient outside its bounds, so a saturated fp16
#     prob of exactly 1.0 gets no gradient at all -- the clamp protects the
#     backward rather than endangering it (measured grad amax = 0).
#   * The probability branch CAN still produce a non-finite gradient from a
#     finite forward, just INSIDE the clamp: at 1-p = 2e-6 the gradient
#     reaches 3.7e5 and at 1e-5 it reaches 7.5e4, both above fp16's 65504, so
#     the cast back across the float32-island boundary overflows -- and that
#     is before the GradScaler's loss scale multiplies it. This is the
#     derivative caveat recorded in RECON-3, now demonstrated.
#   * BUT the module trace CONTRADICTS that as the source here. A non-finite
#     prob_lc/prob_pac gradient would reach `adaptive` first (it produces
#     prob), then lc_head/pac, then lc. All four are finite. Only encoder and
#     local_head are nan, and the only loss terms touching local_head are
#     focal_loss_logits and smooth_l1_reg_loss on the local branch.
#   * The LOGIT branch measures safe at the observed range: gradient amax is
#     0.750 at logits 12, 15 and 20 -- flat and finite, as
#     binary_cross_entropy_with_logits is designed to be.
#
# So the prime suspect is contradicted and this run exists to NAME the op
# rather than to confirm a hypothesis. torch.autograd.set_detect_anomaly(True)
# raises at the first backward op producing a non-finite value and gives a
# traceback to the forward line that created it.
#
# IDENTICAL to 549373 apart from trainer.detect_anomaly=true.
#
# NO STEP CAP, deliberately. A 20-step run would not reach the defect: it
# first appears at batch 31. Anomaly mode self-terminates at the first
# non-finite gradient, so the run stops on its own at ~batch 31; at 2-5x the
# ~1 s/step baseline that is roughly 1-3 minutes.
#
# READ THE STDERR, NOT THE BUNDLE. The failing step's TrainRecord is written
# AFTER backward, so the step that raises will NOT appear in metrics.csv --
# the traceback is its record. Steps 0..30 do survive: explog.close() runs in
# train.py's outer finally.
#SBATCH --job-name=cora-anom
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
#SBATCH --exclude=hpc-node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=anom-%j.out
#SBATCH --error=anom-%j.err

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
    trainer.detect_anomaly=true \
    model.cssm.dt_init=mamba \
    taps=stats \
    results_dir="$HOME/cora-anom"
