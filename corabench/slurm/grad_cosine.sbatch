#!/bin/bash
#SBATCH --job-name=grad-cosine
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40|l40s
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# sec 7.15b gradient-cosine experiment. ONE batch, two checkpoints, no
# training. Staging dominates the runtime, not the compute.
set -uo pipefail
REPO="$HOME/Fault-Injectors"; OCOOD="$HOME/opencood-official"
SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
echo "=== grad-cosine | node $(hostname) | gpu $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) ==="
mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' /datasets/eemcs/ps/cv/opencood/opv2v/train/ "$SCRATCH/data/"
cp "$OCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml" "$SCRATCH/hypes.yaml"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/hypes.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/hypes.yaml"
source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -u tools/grad_cosine.py --hypes "$SCRATCH/hypes.yaml" --seed 2029 \
  --ckpt "$REPO/results/cora_full/561546/ckpt_step2000.pt" \
  --ckpt "$REPO/results/cora_full/561546/ckpt_step6000.pt"
echo "GRAD-COSINE EXIT $?"
