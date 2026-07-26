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
# FALSIFIABLE PREDICTION -- check this FIRST, at step 0, before anything else.
#
# PAC's classifier (fusion/pac.py fuse_cls) now carries the -4.59 focal prior
# it never had. With near-zero default weights cls_out ~ -4.59, so
# sigmoid(cls) = 0.01, and the focal loss consumes the recalibrated product
# sigmoid(cls) * sigmoid(-U) = 0.01 * 0.437 = 0.00437 (U at its own default
# init, observed mean +0.255 in 547612). Then per positive anchor
#   0.25 * (1 - 0.00437)^2 * (-log 0.00437) = 1.347
# and all 70356 negatives together contribute ~1e-4 after dividing by
# n_pos = 44.2 (MEASURED from the data, not assumed).
#
#   cls_pac       should be 1.34 - 1.36   (was ~47.6)
#   cls  (logged) should be ~3.6          (1.13 + 1.13 + 1.35; was 50.11)
#   pac  (logged) should be ~1.55 - 1.65  (NOT 1.35 -- the logged `pac`
#                 column is cls_pac + reg_weight*reg_pac with reg_weight=2,
#                 and reg_pac ~ 0.115 at step 0, so it sits ~0.23 above
#                 cls_pac. Reading 1.58 as a miss would be a false alarm.)
#
# Anything outside those bands means the chain is not what the taps imply and
# the analysis behind this fix is wrong somewhere.
#
# grad_norm_by_module (new column in metrics.csv) answers the ownership
# question the global norm cannot: whether AdaptiveFusion drives the LC growth
# THROUGH lc or lc originates it. Expect adaptive to dominate -- on synthetic
# it holds 401 of a 417 global norm at step 0 -- though that proxy predates
# the PAC prior, which should shrink adaptive's share substantially since much
# of that gradient was the uncertainty net straining to suppress an
# over-confident PAC branch (uncertainty_pac mean went +0.25 -> +11.4).
#
# The bias does NOT fix the underlying defect: PAC multiplies LOGITS by
# sigmoid gates, so a closing gate drives the logit to 0, i.e. p -> 0.5, the
# opposite of its semantic role. See the RECON note "gate composition in
# logit space" in docs/corabench_design.md. Recorded, deliberately unfixed.
#
# Expect ~3.6 min of training (210 steps at ~1.02 s) plus ~40 s validation.
#SBATCH --job-name=cora-lossfix
#SBATCH --partition=ps,main-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
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
