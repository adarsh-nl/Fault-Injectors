#!/bin/bash
# Diagnostic re-run of job 546515: localise the non-finite quantity.
#   sbatch corabench/slurm/nan_diag.sbatch
#
# 546515 ended with GradScaler scale == 0.0 after ~165 consecutive backoffs
# inside one 210-step epoch, 226/227 non-finite model tensors and 268/402
# non-finite Adam states. Backing the scale off never fixed it, so the
# non-finite quantity does NOT depend on the loss scale: it originates in the
# forward pass or the loss value, not in fp16 gradient overflow.
#
# This run is IDENTICAL to 546515 -- same seed, scenarios, lr, grad_clip, grid
# and amp=true -- except for three diagnostic changes:
#   trainer.log_every=1   per-step onset, not a 50-step window
#   taps=stats            StatsTap n_nan/n_inf at all 41 CoRA locations
#   results_dir           separate, so 546515's bundle is not overwritten
#SBATCH --job-name=cora-nan
#SBATCH --partition=ps,main-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=nan-diag-%j.out
#SBATCH --error=nan-diag-%j.err

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

# This is the first real use of the path resolver (df2d6bc). Ask the resolver
# itself what it resolves to and fail here, in the first second, rather than
# discovering a wrong root inside the diagnostic.
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

# Same scenario selection as 546515: one name per split, quoted (YAML 1.1
# reads bare underscores as digit separators).
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
    results_dir="$HOME/cora-nan-diag"
