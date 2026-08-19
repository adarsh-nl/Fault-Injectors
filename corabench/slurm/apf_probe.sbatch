#!/bin/bash
#SBATCH --job-name=apf-probe
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --exclude=hpc-node14,hpc-node10
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
REPO="$HOME/Fault-Injectors"; OCOOD="$HOME/opencood-official"
SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
echo "=== apf probe | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) ==="
DATA="$SCRATCH/v2xset/test"; mkdir -p "$DATA"
rsync -a --exclude='*.png' /datasets/eemcs/ps/cv/opencood/v2xset/test/ "$DATA/"
cp -r "$HOME/opencood-eval/cobevt" "$SCRATCH/model"
for k in root_dir validate_dir test_dir; do sed -i "s|^$k:.*|$k: '$DATA'|" "$SCRATCH/model/config.yaml"; done
source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -u tools/apf/gate0/run_gate0.py --mode probe --model-dir "$SCRATCH/model"
echo "APF PROBE EXIT $?"
