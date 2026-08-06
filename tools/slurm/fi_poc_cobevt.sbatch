#!/bin/bash
#SBATCH --job-name=fi-poc-cobevt
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# PoseError sev-2 PoC on CoBEVT / V2XSet.
#
# Four inference passes in ONE job, so all four share a session, a seed and a
# worker count -- which is what makes the comparison valid:
#
#   official    no wrapper at all          -> the fresh clean control
#   clean       wrapper, NULL pipeline     -> Gate 2: must match `official`
#   clean_rep   identical repeat of clean  -> determinism check
#   pose_sev2   sigma_xy=0.2, sigma_yaw=0.2 deg, non-ego only
#
# Every pass runs through tools/fi_inference.py, so the numpy seeding,
# wild_setting forcing and full-precision AP capture are identical across all
# four; only the dataset wrapper differs.

set -euo pipefail

SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
MODEL_DIR="$HOME/opencood-eval/cobevt"
SEED=1234

echo "=== node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

# ── stage: LiDAR + labels only (18 GB of the 42 GB split; PNGs unused) ──
mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' \
      /datasets/eemcs/ps/cv/opencood/v2xset/test/ "$SCRATCH/data/"
echo "staged $(find "$SCRATCH/data" -name '*.pcd' | wc -l) pcd files"

# checkpoint + config to scratch so we never write into ~/opencood-eval
cp -r "$MODEL_DIR" "$SCRATCH/model"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/model/config.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/model/config.yaml"
# Hard-fail rather than run four passes against /datasets over NFS.
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/model/config.yaml" \
    | tee /dev/stderr | wc -l | grep -qx 2 \
    || { echo "FATAL: config data paths not rewritten to scratch"; exit 1; }

source activate opencood-official
export PYTHONPATH="$OCOOD:$REPO:${PYTHONPATH:-}"
cd "$OCOOD"

for COND in official clean clean_rep pose_sev2; do
    SPEC="$COND"
    [ "$COND" = "clean_rep" ] && SPEC="clean"
    echo ""
    echo "################ $COND ################"
    python "$REPO/tools/fi_inference.py" \
        --model_dir "$SCRATCH/model" \
        --fusion_method intermediate \
        --fi_condition "$SPEC" \
        --fi_seed "$SEED" \
        --fi_out "$SCRATCH/out/$COND"
done

# merge the per-worker injection logs (16 workers each append their own file)
for COND in clean clean_rep pose_sev2; do
    D="$SCRATCH/out/$COND/injection"
    if compgen -G "$D/injection_summary.*.csv" > /dev/null; then
        head -1 "$(ls "$D"/injection_summary.*.csv | head -1)" \
            > "$SCRATCH/out/$COND/injection_summary.csv"
        tail -q -n +2 "$D"/injection_summary.*.csv \
            >> "$SCRATCH/out/$COND/injection_summary.csv"
        echo "$COND: $(($(wc -l < "$SCRATCH/out/$COND/injection_summary.csv") - 1)) injection rows"
    fi
done

# Gate 3 must be able to FAIL the job, not merely print: a silently empty log
# means the fault never fired, and every AP below it would be meaningless.
PROWS=0
[ -f "$SCRATCH/out/pose_sev2/injection_summary.csv" ] && \
    PROWS=$(($(wc -l < "$SCRATCH/out/pose_sev2/injection_summary.csv") - 1))
[ "$PROWS" -ge 1 ] || { echo "FATAL: pose_sev2 injection log is empty"; exit 1; }
[ ! -f "$SCRATCH/out/clean/injection_summary.csv" ] || \
    { echo "FATAL: null pipeline logged an injection"; exit 1; }
echo "GATE3: pose_sev2 logged $PROWS injections; clean logged none"

echo ""
echo "=== SUMMARY ==="
export SCRATCH_OUT="$SCRATCH/out"
python - <<'PY'
import glob, json, os
rows = {}
for f in sorted(glob.glob(os.environ['SCRATCH_OUT'] + '/*/fi_result.json')):
    r = json.load(open(f)); rows[os.path.basename(os.path.dirname(f))] = r
for k in ('official', 'clean', 'clean_rep', 'pose_sev2'):
    if k in rows:
        print('%-10s ap_50=%.6f  ap_70=%.6f' % (k, rows[k]['ap_50'], rows[k]['ap_70']))
if 'official' in rows and 'clean' in rows:
    print('GATE2 wrapper-null vs official: d50=%.3e d70=%.3e'
          % (abs(rows['clean']['ap_50'] - rows['official']['ap_50']),
             abs(rows['clean']['ap_70'] - rows['official']['ap_70'])))
if 'clean' in rows and 'clean_rep' in rows:
    print('DETERMINISM clean vs repeat:    d50=%.3e d70=%.3e'
          % (abs(rows['clean']['ap_50'] - rows['clean_rep']['ap_50']),
             abs(rows['clean']['ap_70'] - rows['clean_rep']['ap_70'])))
if 'official' in rows and 'pose_sev2' in rows:
    print('DELTA sev2 - clean control:     d50=%+.6f d70=%+.6f'
          % (rows['pose_sev2']['ap_50'] - rows['official']['ap_50'],
             rows['pose_sev2']['ap_70'] - rows['official']['ap_70']))
PY

# results back before the trap wipes scratch
mkdir -p "$HOME/results/$SLURM_JOB_ID"
rsync -a --exclude='injection/' "$SCRATCH/out/" "$HOME/results/$SLURM_JOB_ID/"
echo "results -> $HOME/results/$SLURM_JOB_ID"
