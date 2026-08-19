#!/bin/bash
#SBATCH --job-name=adam-probe
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
source activate opencood-official
export PYTHONPATH="$HOME/Fault-Injectors:${PYTHONPATH:-}"
cd "$HOME/Fault-Injectors"
python -u tools/adam_moment_probe.py \
  results/cora_full/561546/ckpt_step2000.pt \
  results/cora_full/561546/ckpt_step4000.pt \
  results/cora_full/561546/ckpt_step6000.pt \
  results/cora_full/561546/abort_step7667.pt \
  results/cora_full/561545/abort_step382.pt
echo "ADAM PROBE EXIT $?"
