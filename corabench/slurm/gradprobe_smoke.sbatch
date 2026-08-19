#!/bin/bash
#SBATCH --job-name=gprobe-smoke
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# sec 7.9 discipline: 3 steps on CPU, reading a handful of frames straight
# from the shared FS (no staging -- 3 steps touch ~6 samples), to prove the
# --grad_probe columns populate BEFORE any GPU job is submitted.
set -uo pipefail
REPO="$HOME/Fault-Injectors"; OCOOD="$HOME/opencood-official"
SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$OCOOD/opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml" "$SCRATCH/hypes.yaml"
sed -i "s|^root_dir:.*|root_dir: '/datasets/eemcs/ps/cv/opencood/opv2v/train'|" "$SCRATCH/hypes.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '/datasets/eemcs/ps/cv/opencood/opv2v/train'|" "$SCRATCH/hypes.yaml"
source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -u -m corabench.scripts.train_opencood \
    --hypes "$SCRATCH/hypes.yaml" --out "$SCRATCH/out" \
    --max_steps 3 --batch_size 2 --lr 1e-3 --num_workers 4 --seed 2029 \
    --dt_bound 0.2 --gate_interval 100 --ckpt_every 0 --grad_probe
RC=$?
echo "--- CSV header ---"
head -1 "$SCRATCH/out/smoke_loss.csv" | tr ',' '\n' | grep -nE "g_|r_|z_fused|y_pre|y_post|loss_trend"
echo "--- rows ---"
sed -n '2,4p' "$SCRATCH/out/smoke_loss.csv"
echo "GPROBE SMOKE EXIT $RC"
