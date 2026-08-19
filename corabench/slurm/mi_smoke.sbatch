#!/bin/bash
#SBATCH --job-name=mi-smoke
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# 30 frames, 2 conditions. Proves taps fire, ego assertion holds, Y is
# non-degenerate and the estimators run -- BEFORE the full 3x2834 run.
set -uo pipefail
REPO="$HOME/Fault-Injectors"; OCOOD="$HOME/opencood-official"
SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
echo "=== mi smoke | $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) ==="
DATA="$SCRATCH/v2xset/test"; mkdir -p "$DATA"
rsync -a --exclude='*.png' /datasets/eemcs/ps/cv/opencood/v2xset/test/ "$DATA/"
cp -r "$HOME/opencood-eval/cobevt" "$SCRATCH/model"
for k in root_dir validate_dir test_dir; do
  sed -i "s|^$k:.*|$k: '$DATA'|" "$SCRATCH/model/config.yaml"; done
source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"
OUT="$SCRATCH/mi"; mkdir -p "$OUT"
for C in clean agent_drop; do
  echo ""; echo "######## $C ########"
  python -u tools/mi_collect_opencood.py --model_dir "$SCRATCH/model" \
     --condition $C --out "$OUT/features_$C.npz" --n_samples 30 || exit 1
done
echo ""; echo "######## analyse (30 frames, tiny) ########"
python -u tools/mi_analyse.py --dir "$OUT" --conditions clean agent_drop \
   --pca_dims 8 --epochs_infonce 20 --epochs_smile 50 --batch_size 16
mkdir -p "$REPO/results/mi/smoke_$SLURM_JOB_ID"
rsync -a "$OUT/" "$REPO/results/mi/smoke_$SLURM_JOB_ID/"
echo "MI SMOKE EXIT $?"
