#!/bin/bash
#SBATCH --job-name=fi-snow-resume
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Resume the snow cells a walled part-c did not reach.
# 558721 (v2xvit snow) hit its 24 h limit inside lidar_snow/moderate: mild
# landed, moderate and severe did not. Snow is ~10 h/cell on V2XSet, so the
# 3-cell part needs >24 h; this runs only the missing cells, with 48 h.
#
#   sbatch tools/slurm/fi_sweep_snow_resume.sbatch <model> <tier> [<tier>...]

set -uo pipefail
MODEL="$1"; shift; TIERS="$*"
REPO="$HOME/Fault-Injectors"; SEED=1234
NS=opencood; CODE_ROOT="$HOME/opencood-official"
case "$MODEL" in
  cobevt)     MD="$HOME/opencood-eval/cobevt";     DS="/datasets/eemcs/ps/cv/opencood/v2xset/test" ;;
  where2comm) MD="$HOME/opencood-eval/where2comm"; DS="/datasets/eemcs/ps/cv/opencood/v2xset/test" ;;
  v2xvit)     MD="/datasets/eemcs/ps/cv/opencood/v2xset_checkpoints/v2x-vit"
              DS="/datasets/eemcs/ps/cv/opencood/v2xset/test"
              NS=v2xvit; CODE_ROOT="$HOME/v2xvit-official" ;;
  *) echo "FATAL: unknown model $MODEL"; exit 1 ;;
esac
DEST="$REPO/results/sweep/$MODEL"; mkdir -p "$DEST"

SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
echo "=== $MODEL snow resume [$TIERS] on $(hostname) ==="
mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' "$DS/" "$SCRATCH/data/"
cp -r "$MD" "$SCRATCH/model"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/model/config.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/model/config.yaml"
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/model/config.yaml" \
  | wc -l | grep -qx 2 || { echo "FATAL: paths not rewritten"; exit 1; }

source activate opencood-official
export PYTHONPATH="$CODE_ROOT:$REPO:${PYTHONPATH:-}"
cd "$CODE_ROOT"

FAILED=0
for T in $TIERS; do
  case "$T" in
    mild)     SEV=1 ;;
    moderate) SEV=2 ;;
    severe)   SEV=3 ;;
    *) echo "FATAL: unknown tier $T"; exit 1 ;;
  esac
  SPEC="{\"lidar_snow\": {\"mount_height\": 1.9, \"severity\": $SEV}}"
  echo ""; echo "################ $MODEL/lidar_snow/$T ################"
  if python "$REPO/tools/fi_inference.py" --model_dir "$SCRATCH/model" \
      --fusion_method intermediate --fi_spec "$SPEC" --fi_ns "$NS" \
      --fi_seed "$SEED" --fi_out "$SCRATCH/out/lidar_snow/$T"; then
      mkdir -p "$DEST/lidar_snow/$T"
      rsync -a "$SCRATCH/out/lidar_snow/$T/" "$DEST/lidar_snow/$T/"
      echo "cell lidar_snow/$T done -> $DEST/lidar_snow/$T"
  else
      echo "CELL FAILED: $MODEL/lidar_snow/$T"; FAILED=1
  fi
done
echo ""; echo "=== SNOW RESUME DONE model=$MODEL tiers=[$TIERS] failed=$FAILED ==="
exit $FAILED
