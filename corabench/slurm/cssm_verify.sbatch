#!/bin/bash
#SBATCH --job-name=cssm-verify
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
echo "=== cssm verify | node $(hostname) | job $SLURM_JOB_ID ==="
source activate opencood-official
export PYTHONPATH="$HOME/Fault-Injectors:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
cd "$HOME/Fault-Injectors"
python -u tools/cssm_verify.py --phase fwd,bwd,ref,perm,dyn
echo "VERIFY EXIT $?"
