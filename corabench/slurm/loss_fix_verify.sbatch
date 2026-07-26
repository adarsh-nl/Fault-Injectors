#!/bin/bash
# Verify the Bug 1 loss fix and observe whether Bug 2 is self-limiting.
#   sbatch corabench/slurm/loss_fix_verify.sbatch
#
# Predecessor 547612 aborted at batch 45 with cls=inf while every one of the
# 41 forward taps stayed finite: head/cls_logits reached 205.5, fp16 sigmoid
# saturated to exactly 1.0, and clamp(p, 1e-6, 1-1e-6) was a no-op because
# 1.0-1e-6 rounds to 1.0 in half precision, so log(1-p) = log(0) = -inf.
#
# IDENTICAL to 547612 in every training-relevant respect -- same seed (2026),
# scenarios, lr (1e-3), grad_clip (10), grid, amp=true, log_every=1,
# taps=stats. ONE variable moved: the focal loss fix
#   site 1 (local head)  -> binary_cross_entropy_with_logits
#   sites 2/3 (LC, PAC)  -> float32 island, uncertainty gradient preserved
#
# COMPLETION IS NOT EVIDENCE OF HEALTH. The float32 island removes the inf,
# which also means the trainer's non-finite guard will NOT fire. If the head
# still saturates, this run finishes cleanly, writes a checkpoint that passes
# the finiteness check, and reports an AP -- while being just as broken as
# 547612. In fp16 sigmoid saturates to exactly 1.0 above a logit of ~8.3;
# 547612 reached 205.5. The trainer now logs head_logit_amax every step and
# WARNS above 20, so a saturating run announces itself in the .err file
# rather than waiting to be found in taps.csv. Grep the log for "saturating".
#
# THE DISCRIMINATING OBSERVATION is not whether this completes. It is what
# lc/output does once the loss is finite and optimizer steps keep landing.
# In 547612 it grew 0.331 -> 47.03 (142x) across the only three steps that
# ever landed, while encoder/bev_features stayed flat. Two outcomes:
#   stabilises -> the growth was a transient of the first few steps
#   keeps growing -> the LC branch has a genuine gain problem and lr, warmup
#                    or normalisation is the next variable to move
# Read it from taps.csv, location lc/output, column amax, against step.
#
# grad_norm_by_module (new column in metrics.csv) answers the ownership
# question the global norm cannot: whether AdaptiveFusion drives the growth
# THROUGH lc or lc originates it. Expect adaptive to dominate -- on synthetic
# it holds 401 of a 417 global norm at step 0 -- because PAC's classifier
# (fusion/pac.py:112, a plain Conv2d) never receives the -4.59 focal prior
# that DetectionHead applies, so PAC starts at p_bg ~ 0.25 and contributes
# ~47.7 of the ~50.1 step-0 cls loss while local and lc contribute ~1.13 each.
# That missing prior is a SEPARATE, UNFIXED defect; this run does not address
# it and should not be read as if it did.
#
# Expect ~3.6 min of training (210 steps at ~1.02 s) plus ~40 s validation.
#SBATCH --job-name=cora-lossfix
#SBATCH --partition=ps,main-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=lossfix-%j.out
#SBATCH --error=lossfix-%j.err

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

# Same scenario selection as 546515 and 547612: one name per split, quoted
# (YAML 1.1 reads bare underscores as digit separators).
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
    taps=stats \
    results_dir="$HOME/cora-lossfix"
