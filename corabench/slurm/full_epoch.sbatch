#!/bin/bash
# First attempt at a COMPLETE CoRA epoch on real OPV2V.
#   sbatch corabench/slurm/full_epoch.sbatch
#
# This run exists to prove the chain holds end to end -- LC -> loss -> decode
# -> backward -> validation -- across all 210 steps plus validation, which has
# never happened. Every predecessor died partway:
#
#   546515  loss NaN mid-epoch, 226/227 non-finite tensors, NaN "best.pt"
#   547612  cls=inf at batch 45, fp16 focal clamp a no-op above logit ~8.3
#   549175  Turing node, torch cu130 does not target sm_75
#   549332/3  torch.quantile refuses >2**24 elements, died in the first forward
#   549373  BOUNDED for 47 steps (dt_init works) then AsinBackward0 at batch 31
#   549416  anomaly mode named it: asin'(x) singular at x=+-1
#
# Fixed since 549373: the yaw residual is clamped strictly inside asin's
# singularity (96fbb2a). Nothing else changed.
#
# IDENTICAL to 549373 except:
#   trainer.scale_floor  1.0e-8 -> 1.0e-4
#   results_dir          -> $HOME/cora-fullepoch
#   no trainer.detect_anomaly
#
# WHY THE HIGHER FLOOR. 1e-8 is the diagnostic setting: it lets the run
# continue through a dead scaler, and in 549227 that meant 59 steps with
# gradients underflowed to exactly zero passing the finite check while Adam's
# moments decayed and the LR schedule advanced -- 101 of 116 steps doing
# nothing. For a run that is meant to STAND as the first clean epoch, 1e-4 is
# the honest floor, and aborting on it is a result rather than a failure.
#
# 1.0e-4, not 1e-4: YAML 1.1 parses `1e-4` as a STRING (the mantissa needs a
# decimal point). trainer.py calls float() so both work, but the same family
# of trap already cost a run via the scenario-name integer parse.
#
# THE AP FROM THIS RUN IS A THROWAWAY. RECON-5: yaw is encoded as a single
# sine channel, so arcsin covers only [-pi/2, pi/2] and sin(D) = sin(pi-D)
# makes heading 180-degree ambiguous. cpbench/data/postprocessing.py decodes
# with the same arcsin, in numpy outside autograd, so VALIDATION AP IS
# COMPUTED ON SATURATED, DIRECTION-AMBIGUOUS YAW. The decode fix is Nreg 7->8
# across shared cpbench code touching all five paper packages, and it
# deliberately does NOT ride into this run. Quote any AP from here as
# pre-decode-fix or not at all.
#
# READ THE RESULT IN THIS ORDER:
#   1. does the epoch complete (210 steps)
#   2. does validation run
#   3. peak GPU memory and epoch wall clock -- these size every later run
#   4. AP, and ONLY as a pipeline-completes signal
#SBATCH --job-name=cora-fullep
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40      # see CLAUDE.md: main-gpu spans Turing/Ampere/Lovelace
#SBATCH --exclude=hpc-node10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=fullep-%j.out
#SBATCH --error=fullep-%j.err

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
    taps=stats \
    results_dir="$HOME/cora-fullepoch"
