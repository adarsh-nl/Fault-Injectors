#!/bin/bash
#SBATCH --job-name=cssm-smoke
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
echo "=== cssm smoke | node $(hostname) | job $SLURM_JOB_ID ==="
source activate opencood-official
export PYTHONPATH="$HOME/Fault-Injectors:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
cd "$HOME/Fault-Injectors"
python -u tools/cssm_verify.py --phase smoke
echo "SMOKE EXIT $?"
