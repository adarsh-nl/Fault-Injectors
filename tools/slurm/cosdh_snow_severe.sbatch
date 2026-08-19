#!/bin/bash
#SBATCH --job-name=cosdh-snow-sev
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# ONE cell: cosdh/lidar_snow/severe. Part c timed out at 36h with mild and
# moderate consuming ~12h each; re-running part c would repeat ~24h of
# already-landed work. 24h alone is ~1.5x the observed per-cell cost.
set -uo pipefail
REPO="$HOME/Fault-Injectors"; SEED=1234
CKPT=/datasets/eemcs/ps/cv/opencood/cosdh_checkpoints/opv2v_cosdh
DEST="$REPO/results/sweep/cosdh"; mkdir -p "$DEST"
SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
echo "=== cosdh lidar_snow/severe node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) ==="
DATA="$SCRATCH/OPV2V/test"; mkdir -p "$DATA"
rsync -a --exclude='*.png' /datasets/eemcs/ps/cv/opencood/opv2v/test/ "$DATA/"
cp -r "$CKPT" "$SCRATCH/model"
for k in root_dir validate_dir test_dir; do sed -i "s|^$k:.*|$k: '$DATA'|" "$SCRATCH/model/config.yaml"; done
source $(conda info --base)/etc/profile.d/conda.sh; conda activate cosdh
export PYTHONPATH="$HOME/cosdh-official:$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -c "import opencood; assert 'cosdh-official' in opencood.__file__" || exit 1
SPEC='{"lidar_snow": {"mount_height": 1.9, "severity": 3}}'
python tools/cosdh_inference.py --model_dir "$SCRATCH/model" \
    --fi_spec "$SPEC" --fi_seed "$SEED" --fi_out "$SCRATCH/out/lidar_snow/severe"
RC=$?
mkdir -p "$DEST/lidar_snow/severe"
rsync -a "$SCRATCH/out/lidar_snow/severe/" "$DEST/lidar_snow/severe/"
echo "exit $RC; cells now: $(find $DEST -name fi_result.json | wc -l)/26"
exit $RC
