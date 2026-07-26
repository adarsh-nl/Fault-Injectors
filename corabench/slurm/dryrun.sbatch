#!/bin/bash
# One-scenario, one-epoch dry run. Detached, so an SSH drop cannot kill it.
#   sbatch corabench/slurm/dryrun.sbatch
#SBATCH --job-name=cora-dry
#SBATCH --partition=ps,main-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=dryrun-%j.out
#SBATCH --error=dryrun-%j.err

set -euo pipefail

REPO="$HOME/Fault-Injectors"

# One resolution point for data: CPBENCH_DATA_ROOT. Set it in your shell (or
# here) for the machine; everything below and every config derives from it.
# On the UT EEMCS cluster the shared CV tree is /datasets/eemcs/ps/cv.
export CPBENCH_DATA_ROOT="${CPBENCH_DATA_ROOT:-/datasets/eemcs/ps/cv}"
DATA_ROOT="$CPBENCH_DATA_ROOT/opencood/opv2v"

module purge
unset PYTHONHOME PYTHONSTARTUP

cd "$REPO"
export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# dataset.scenarios is ONE key applied to every split, and build_adapters
# filters by name membership. So pass one name from each split: train/ keeps
# the train scenario, validate/ keeps the validate one. Passing only a train
# name leaves the validation adapter empty and the job dies at train.py:58.
#
# Names must be QUOTED inside the brackets: YAML 1.1 reads underscores as
# digit separators, so [2021_08_16_22_26_54] becomes the integer
# 20210816222654 and matches no directory.
SCEN_TRAIN=$(ls "$DATA_ROOT/train"    | head -1)
SCEN_VAL=$(ls   "$DATA_ROOT/validate" | head -1)
echo "train scenario:    $SCEN_TRAIN  ($(ls "$DATA_ROOT/train/$SCEN_TRAIN" | wc -l) agents)"
echo "validate scenario: $SCEN_VAL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

srun .venv-hpc/bin/python -m corabench.scripts.train \
    dataset=opv2v \
    dataset.scenarios="[\"$SCEN_TRAIN\",\"$SCEN_VAL\"]" \
    trainer.epochs=1 trainer.num_workers=7 \
    results_dir="$HOME/cora-dryrun"
