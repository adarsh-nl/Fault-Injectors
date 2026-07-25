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
DATA_ROOT="${DATA_ROOT:-/datasets/eemcs/ps/cv/opencood/opv2v}"

module purge
unset PYTHONHOME PYTHONSTARTUP

cd "$REPO"
export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# A 3-agent scenario is closer to the 2.8 train mean than the first
# alphabetically, which is 2-agent and would understate per-step cost.
SCEN=$(ls "$DATA_ROOT/train" | head -1)
echo "scenario: $SCEN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun .venv-hpc/bin/python -m corabench.scripts.train \
    dataset=opv2v dataset.root="$DATA_ROOT" \
    dataset.scenarios="[\"$SCEN\"]" \
    trainer.epochs=1 trainer.num_workers=7 \
    results_dir="$HOME/cora-dryrun"
