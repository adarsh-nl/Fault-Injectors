#!/bin/bash
#SBATCH --job-name=y-stats
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
REPO="$HOME/Fault-Injectors"; OCOOD="$HOME/opencood-official"
SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
cp -r "$HOME/opencood-eval/cobevt" "$SCRATCH/model"
D=/datasets/eemcs/ps/cv/opencood/v2xset/test
for k in root_dir validate_dir test_dir; do sed -i "s|^$k:.*|$k: '$D'|" "$SCRATCH/model/config.yaml"; done
source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -u /home/nanjaiyalathaa/.claude/jobs/21b05871/tmp/y_stats.py 400 "$SCRATCH/model/config.yaml"
echo "Y STATS EXIT $?"
