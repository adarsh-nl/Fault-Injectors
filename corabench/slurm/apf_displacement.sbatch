#!/bin/bash
#SBATCH --job-name=apf-disp
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# APF Gate 0 file 3. Dataset only, no model, no GPU, no opencood import.
# Off the login node because a full pass is about 2.5 minutes of yaml reads,
# past the login node budget.
set -uo pipefail
cd "$HOME/Fault-Injectors"
.venv-hpc/bin/python -u tools/apf/gate0/displacement.py \
    --root /datasets/eemcs/ps/cv/opencood/v2xset/test \
    --out-dir results/apf/gate0
echo "APF DISP EXIT $?"
