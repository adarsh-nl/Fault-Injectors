#!/bin/bash
#SBATCH --job-name=fi-null-gate
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --exclude=hpc-node14
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ASSERT (b): wrapped-null AP must equal the UNWRAPPED SHIPPED-SETTING AP.
#
# Both passes run in ONE session (same node, same seed, same worker count) so
# the reference is contemporaneous -- a remembered number cannot serve, GPU
# nondeterminism is real (CoBEVT's floor is ~1e-5).
#
# The reference is the model AT ITS SHIPPED wild_setting (loc_err/async ON for
# the Noisy models). Comparing against a forced-clean run would encode the very
# bug this gate exists to catch.
#
#   sbatch tools/slurm/fi_wrapped_null_gate.sbatch <model>

set -uo pipefail
MODEL="${1:-where2comm}"
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

SCRATCH="/local/$SLURM_JOB_ID"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
echo "=== gate: $MODEL on $(hostname) ==="
mkdir -p "$SCRATCH/data"
rsync -a --exclude='*.png' "$DS/" "$SCRATCH/data/"
cp -r "$MD" "$SCRATCH/model"
sed -i "s|^root_dir:.*|root_dir: '$SCRATCH/data'|"         "$SCRATCH/model/config.yaml"
sed -i "s|^validate_dir:.*|validate_dir: '$SCRATCH/data'|" "$SCRATCH/model/config.yaml"
grep -E "^(root_dir|validate_dir): '$SCRATCH/data'" "$SCRATCH/model/config.yaml" | wc -l \
  | grep -qx 2 || { echo "FATAL: paths not rewritten"; exit 1; }

source activate opencood-official
export PYTHONPATH="$CODE_ROOT:$REPO:${PYTHONPATH:-}"
cd "$CODE_ROOT"

# pass 1: UNWRAPPED at shipped wild_setting (the reference)
echo ""; echo "######## $MODEL unwrapped-shipped (reference) ########"
python "$REPO/tools/fi_inference.py" --model_dir "$SCRATCH/model" \
    --fusion_method intermediate --fi_spec 'null' --fi_ns "$NS" \
    --fi_seed "$SEED" --fi_out "$SCRATCH/out/unwrapped"

# pass 2: WRAPPED, null pipeline, same shipped wild_setting
echo ""; echo "######## $MODEL wrapped-null ########"
python "$REPO/tools/fi_inference.py" --model_dir "$SCRATCH/model" \
    --fusion_method intermediate --fi_spec '{}' --fi_ns "$NS" \
    --fi_seed "$SEED" --fi_out "$SCRATCH/out/wrapped_null"

DEST="$REPO/results/null_gate/$MODEL"; mkdir -p "$DEST"
rsync -a "$SCRATCH/out/" "$DEST/"

# ALSO deposit the wrapped-null bundle into the SWEEP tree, so the aggregator's
# mandatory gate lookup (<root>/<model>/none/wrapped_null) can find it. Without
# this the cell exists but the aggregator still reports GATE-MISSING, which is
# how cobevt sat at 25/26 with a gate that had never run.
CELL="$REPO/results/sweep/$MODEL/none/wrapped_null"
mkdir -p "$CELL"
rsync -a "$SCRATCH/out/wrapped_null/" "$CELL/"
echo "gate cell -> $CELL"

python - "$DEST" "$MODEL" <<'PY'
import json, sys, os
d, model = sys.argv[1], sys.argv[2]
u = json.load(open(os.path.join(d, 'unwrapped', 'fi_result.json')))
w = json.load(open(os.path.join(d, 'wrapped_null', 'fi_result.json')))
d50 = abs(w['ap_50'] - u['ap_50']); d70 = abs(w['ap_70'] - u['ap_70'])
print('ASSERT (b) %s' % model)
print('  unwrapped-shipped ap_50=%.9f ap_70=%.9f  wild=%s'
      % (u['ap_50'], u['ap_70'], u.get('wild_setting_shipped')))
print('  wrapped-null      ap_50=%.9f ap_70=%.9f' % (w['ap_50'], w['ap_70']))
print('  |delta|           d50=%.3e d70=%.3e' % (d50, d70))
# floor: CoBEVT ~1e-5 measured; the two Noisy models measured bit-exact.
FLOOR = 1e-4
print('  VERDICT:', 'PASS (within %.0e)' % FLOOR if (d50 < FLOOR and d70 < FLOOR)
      else 'FAIL -- wrapper still alters the input')
PY
