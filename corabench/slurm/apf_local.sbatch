#!/bin/bash
#SBATCH --job-name=apf-local
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
cd "$HOME/Fault-Injectors"
.venv-hpc/bin/python -u tools/apf/gate0/displacement.py \
    --root /datasets/eemcs/ps/cv/opencood/v2xset/test \
    --out-dir results/apf/gate0 --local --skip-selftest
echo "APF LOCAL EXIT $?"
