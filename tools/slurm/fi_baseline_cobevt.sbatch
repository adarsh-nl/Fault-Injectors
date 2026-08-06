#!/bin/bash
#SBATCH --job-name=fi-base-cobevt
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Post-pcd-fix CoBEVT baseline confirmation + full-cloud fog/snow verification.
#
# Two things, one job:
#   1. OFFICIAL CoBEVT inference (unwrapped, via fi_inference.py `official`)
#      AFTER the src/datasets/pcd.py intensity fix. Expected: AP 0.85 / 0.66
#      at 2 d.p. The baseline path provably never calls load_pcd, so any shift
#      means the fix leaked somewhere unmapped -> STOP.
#   2. Full-cloud LidarFog + LidarSnow verification on real V2XSet clouds in
#      the same env the wrapper runs in (snow at mount_height=1.9, the CARLA
#      LiDAR mount; login-node runs used a 1/16 subsample for time).

set -euo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
MODEL_DIR="$HOME/opencood-eval/cobevt"
SEED=1234

echo "=== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' \
      /datasets/eemcs/ps/cv/opencood/v2xset/test/ "$SCRATCH/data/"
echo "staged $(find "$SCRATCH/data" -name '*.pcd' | wc -l) pcd files"

cp -r "$MODEL_DIR" "$SCRATCH/model"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/model/config.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/model/config.yaml"
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/model/config.yaml" \
    | wc -l | grep -qx 2 \
    || { echo "FATAL: config data paths not rewritten to scratch"; exit 1; }

source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$OCOOD"

echo ""
echo "################ full-cloud fog/snow verification (CPU) ################"
python - "$SCRATCH/data" <<'PY'
import glob, sys, time
import numpy as np
sys.path.insert(0, '/home/nanjaiyalathaa/Fault-Injectors')
from src.datasets.pcd import load_pcd
from src.fault_injectors.lidar_fog import LidarFogInjector
from src.fault_injectors.lidar_snow import LidarSnowInjector

files = sorted(glob.glob(sys.argv[1] + '/*/*/*.pcd'))[:2]
for f in files:
    v = load_pcd(f).astype(np.float64)
    print('cloud %s: %s  I[%.4f,%.4f] mean %.4f'
          % ('/'.join(f.split('/')[-3:]), v.shape, v[:,3].min(), v[:,3].max(), v[:,3].mean()))
    for sev in (1, 2, 3):
        out = LidarFogInjector(severity=sev, seed=1000)(v.copy())
        n = min(len(v), len(out))
        print('  fog  sev%d: pts %d->%d | I mean %.4f->%.4f range [%.4f,%.4f] | '
              'xyz_moved=%d I_changed=%d'
              % (sev, len(v), len(out), v[:,3].mean(), out[:,3].mean(),
                 out[:,3].min(), out[:,3].max(),
                 int((np.abs(out[:n,:3]-v[:n,:3]).max(axis=1)>1e-6).sum()),
                 int((np.abs(out[:n,3]-v[:n,3])>1e-6).sum())))
    for sev in (1, 2, 3):
        t0 = time.time()
        inj = LidarSnowInjector(severity=sev, seed=1000, mount_height=1.9,
                                verbose=False)
        out = inj(v.copy())
        fl = inj.last_flags
        print('  snow sev%d: pts %d->%d (%.0fs) | I mean %.4f->%.4f | '
              'unchanged=%d attenuated=%d scatter=%d removed=%d'
              % (sev, len(v), len(out), time.time()-t0,
                 v[:,3].mean(), out[:,3].mean(),
                 int((fl==0).sum()), int((fl==1).sum()), int((fl==2).sum()),
                 len(v)-len(out)))
    assert len(out) != len(v) or not np.array_equal(out[:,3], v[:,3]), \
        'snow was a silent no-op'
print('FOG/SNOW FULL-CLOUD VERIFICATION DONE')
PY

echo ""
echo "################ official CoBEVT baseline (post pcd fix) ################"
python "$REPO/tools/fi_inference.py" \
    --model_dir "$SCRATCH/model" \
    --fusion_method intermediate \
    --fi_condition official \
    --fi_seed "$SEED" \
    --fi_out "$SCRATCH/out/official"

python - <<'PY'
import json
r = json.load(open('/local/'.__add__(__import__('os').environ['SLURM_JOB_ID'])
                   + '/out/official/fi_result.json'))
ok = round(r['ap_50'], 2) == 0.85 and round(r['ap_70'], 2) == 0.66
print('BASELINE ap_50=%.6f ap_70=%.6f -> %s'
      % (r['ap_50'], r['ap_70'],
         'UNCHANGED at 2 d.p. (0.85/0.66)' if ok
         else 'SHIFTED -- STOP, the pcd fix leaked into the baseline path'))
PY

mkdir -p "$HOME/results/$SLURM_JOB_ID"
rsync -a "$SCRATCH/out/" "$HOME/results/$SLURM_JOB_ID/"
echo "results -> $HOME/results/$SLURM_JOB_ID"
