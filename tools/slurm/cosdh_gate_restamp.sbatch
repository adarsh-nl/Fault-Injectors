#!/bin/bash
#SBATCH --job-name=cosdh-restamp
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --nodelist=hpc-node12   # 562333 produced the original pair here (L40S); determinism floor is PER-NODE
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Re-run ONLY none/clean and none/wrapped_null, in ONE job so they are
# contemporaneous BY CONSTRUCTION -- that is exactly what the gate requires.
# The other 18 CoSDH cells are NOT re-run: their AP is correct and unaffected,
# and a re-run would cost ~30 h to change nothing but a provenance stamp.
set -uo pipefail
REPO="$HOME/Fault-Injectors"; SEED=1234
CKPT=/datasets/eemcs/ps/cv/opencood/cosdh_checkpoints/opv2v_cosdh
DEST="$REPO/results/sweep/cosdh"
SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
echo "=== cosdh gate re-stamp | node $(hostname) | job $SLURM_JOB_ID ==="
DATA="$SCRATCH/OPV2V/test"; mkdir -p "$DATA"
rsync -a --exclude='*.png' /datasets/eemcs/ps/cv/opencood/opv2v/test/ "$DATA/"
cp -r "$CKPT" "$SCRATCH/model"
for k in root_dir validate_dir test_dir; do sed -i "s|^$k:.*|$k: '$DATA'|" "$SCRATCH/model/config.yaml"; done
source $(conda info --base)/etc/profile.d/conda.sh; conda activate cosdh
export PYTHONPATH="$HOME/cosdh-official:$REPO:${PYTHONPATH:-}"
cd "$REPO"
python -c "import opencood; assert 'cosdh-official' in opencood.__file__" || exit 1
RC=0
for CELL in "none/clean:null" "none/wrapped_null:{}"; do
  OUT="${CELL%%:*}"; SPEC="${CELL##*:}"
  echo ""; echo "######## cosdh/$OUT ########"
  python tools/cosdh_inference.py --model_dir "$SCRATCH/model" \
      --fi_spec "$SPEC" --fi_seed "$SEED" --fi_out "$SCRATCH/out/$OUT" || RC=1
  mkdir -p "$DEST/$OUT"
  rsync -a "$SCRATCH/out/$OUT/" "$DEST/$OUT/"
done
echo ""; echo "=== stamps written ==="
for OUT in none/clean none/wrapped_null; do
  python -c "
import json;d=json.load(open('$DEST/$OUT/fi_result.json'))
print('  %-18s job_id=%s injector=%s ap50=%.9f ap70=%.9f'%('$OUT',d.get('job_id'),d.get('injector_sha256'),d['ap_50'],d['ap_70']))"
done
exit $RC
