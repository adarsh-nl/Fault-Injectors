#!/bin/bash
#SBATCH --job-name=v2xvit-full
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/nanjaiyalathaa/v2xvit-results/slurm-%x-%j.out
#SBATCH --error=/home/nanjaiyalathaa/v2xvit-results/slurm-%x-%j.err

# FULL V2X-ViT training run on V2XSet, Perfect (clean) setting.
#
#   60 epochs, batch 2, Adam 1e-3, compression=0, lr_steps [15,50],
#   backbone_fix false, num_workers=8, no faults. These are the shipped
#   Perfect-setting values from v2x-vit hypes_yaml/point_pillar_v2xvit.yaml
#   and are already the package defaults -- no protocol override is passed.
#   Graded later against the paper's clean AP@0.7 0.712 / AP@0.5 0.882.
#
# Budget: 0.750 s/step steady-state, 1.047 s/step in-training wall (job 554537,
# full-epoch measurement). 60 x 3347 steps ~ 58 h, plus ~30 min staging and
# ~15-20 min import. 72 h gives headroom for validation every 5 epochs.
#
# --constraint=l40 keeps this off the Turing nodes whose sm_75 the cu130 build
# does not target (job 549175, dead in 3 s). It no longer pins the exact SKU:
# hpc-node12 and hpc-node14 both advertise `l40` but report L40S and L40. The
# GPU actually obtained is recorded below.
#
# PERSISTENCE: /local is wiped when the job ends, so it holds ONLY the staged
# input. Every output -- checkpoints, logs, progress -- goes to $RESULTS on
# /home, which survives.

set -uo pipefail

REPO="$HOME/Fault-Injectors"
RESULTS="$HOME/v2xvit-results/$SLURM_JOB_ID"
SCRATCH="/local/$SLURM_JOB_ID"
DATA="$SCRATCH/v2xset"
PROGRESS="$RESULTS/progress.txt"

mkdir -p "$RESULTS" "$DATA/train" "$DATA/validate"
# Deliberately NOT trapped for $RESULTS -- only the scratch input is disposable.
trap 'rm -rf "$SCRATCH"' EXIT

SRC=/datasets/eemcs/ps/cv/opencood/v2xset

cd "$REPO"
export PYTHONPATH="$REPO"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
HEAD_SHA=$(git rev-parse --short HEAD)

{
  echo "=== V2X-ViT full run ==="
  echo "job        : $SLURM_JOB_ID on $(hostname)"
  echo "gpu        : $GPU"
  echo "HEAD       : $HEAD_SHA"
  echo "submitted  : $(date -Is)"
  echo "results    : $RESULTS   (persistent)"
  echo "staging    : $DATA      (node-local, wiped on exit)"
  echo "protocol   : 60 ep, bs 2, Adam 1e-3, compression=0, lr_steps [15,50],"
  echo "             backbone_fix false, workers=8, clean/Perfect, faults=none,"
  echo "             eval every 5 ep, ckpt EVERY ep, log_every=1"
  echo "loss       : OpenCOOD-matched cls (per-sample norm, binary labels)"
  echo "oracle     : clean AP@0.7 0.712 / AP@0.5 0.882 (grading is a later step)"
} | tee "$PROGRESS"

# -- capacity check before committing 30 minutes to staging -----------------
AVAIL_GB=$(df -BG --output=avail /local | tail -1 | tr -dc '0-9')
echo "local_avail_gb=$AVAIL_GB" | tee -a "$PROGRESS"
if [ "${AVAIL_GB:-0}" -lt 80 ]; then
    echo "FATAL: /local has ${AVAIL_GB}G free, need ~70G for train+validate" \
        | tee -a "$PROGRESS"
    exit 2
fi

# -- LABEL PRE-FLIGHT -- runs BEFORE staging and before any GPU work --------
# Job 554552 died 44 minutes in, after a 17-minute stage, on the 12th train
# scenario: a label YAML carrying an embedded numpy pickle that yaml.safe_load
# refuses. 1471 files across v2xset and opv2v carry it. That failure was
# detectable in ~90 seconds without touching a GPU, and this gate is that check
# moved to the front of the job.
#
# It opens EVERY label file in the exact splits this run will read, and demands
# four things (results/diag/scan_labels.py):
#   ALL_PARSE          no file raises
#   NO_FIELD_BLANKED   lidar_pose / ego_speed / vehicles survive intact --
#                      the tolerant loader must not silently null a field the
#                      model needs, which would train on corrupt labels
#   SCANNED_SOMETHING  it actually matched files (a predicate bug once made
#                      this pass over zero files, a far more dangerous result)
#   SAW_KNOWN_PICKLES  it saw the pickles it exists to check
# The scanner exits nonzero unless all four hold, so the check below is simply
# its exit code.
echo "label preflight start $(date -Is)" | tee -a "$PROGRESS"
PREFLIGHT_T0=$(date +%s)
SCAN_DATASETS=v2xset \
SCAN_SPLITS=train,validate \
SCAN_JOBS="${SLURM_CPUS_PER_TASK:-8}" \
SCAN_ROOT=/datasets/eemcs/ps/cv/opencood \
SCAN_OUT="$RESULTS/preflight_labels.json" \
    "$REPO/.venv-hpc/bin/python" "$REPO/results/diag/scan_labels.py" \
    2>&1 | tee -a "$PROGRESS"
PREFLIGHT_RC=${PIPESTATUS[0]}
echo "label preflight done  $(date -Is) ($(( $(date +%s) - PREFLIGHT_T0 ))s) rc=$PREFLIGHT_RC" \
    | tee -a "$PROGRESS"

if [ "$PREFLIGHT_RC" -ne 0 ]; then
    {
      echo
      echo "FATAL: label pre-flight FAILED (rc=$PREFLIGHT_RC). Nothing staged,"
      echo "       no GPU work done, no training started."
      echo "       Details: $RESULTS/preflight_labels.json"
      echo "       Look at the 'failures' and 'blanked' arrays -- a non-empty"
      echo "       'blanked' means a label parsed but lost a field the model"
      echo "       needs, which is the dangerous case: it would NOT crash."
    } | tee -a "$PROGRESS"
    exit 3
fi

# -- stage full train + validate, LiDAR + YAML only -------------------------
# Camera PNGs are ~56% of the bytes (3.9 GB/scenario total vs 1.72 GB
# LiDAR+YAML) and the LiDAR track never opens them.
echo "staging start $(date -Is)" | tee -a "$PROGRESS"
STAGE_T0=$(date +%s)
for split in train validate; do
    rsync -a --include='*/' --include='*.pcd' --include='*.yaml' --exclude='*' \
        "$SRC/$split/" "$DATA/$split/"
done
STAGE_S=$(( $(date +%s) - STAGE_T0 ))
{
  echo "staging done  $(date -Is)  (${STAGE_S}s, $(du -sm "$DATA" | cut -f1) MB)"
  echo "train scenarios: $(ls "$DATA/train" | wc -l)   validate: $(ls "$DATA/validate" | wc -l)"
} | tee -a "$PROGRESS"

# -- background progress writer ---------------------------------------------
# metrics.csv is buffered in memory and only written when fit() finishes
# (cpbench/logbook/experiment.py: _CsvSink.add appends, flush rewrites;
# Trainer.fit flushes once, after the last epoch). So it cannot be tailed for
# progress. Per-epoch checkpoints appearing on disk are the reliable live
# signal; this loop surfaces them plus GPU state every 5 minutes.
(
  while true; do
      sleep 300
      LATEST=$(ls -1 "$RESULTS"/*/checkpoints/epoch_*.pt 2>/dev/null | sort | tail -1)
      NCKPT=$(ls -1 "$RESULTS"/*/checkpoints/epoch_*.pt 2>/dev/null | wc -l)
      UTIL=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
             --format=csv,noheader 2>/dev/null | head -1)
      echo "[$(date -Is)] epochs_done=${NCKPT} latest=$(basename "${LATEST:-none}") gpu=${UTIL}" \
          >> "$PROGRESS"
  done
) &
MON_PID=$!

# -- live loss log, tailable --------------------------------------------------
# The operator checks the loss at epochs 1, 5 and 10; the 600-step smoke only
# proved stability that far. metrics.csv cannot serve this (memory-buffered,
# written once at the end), so this converts the incrementally-written
# tensorboard events to text every 5 minutes, with an explicit NaN / DIVERGED
# marker. Checkpoints are written every epoch, so a divergence caught here
# always has recoverable weights behind it.
LOSSLOG="$RESULTS/loss_log.txt"
echo "# live loss log -- appended every 5 min from tensorboard events" > "$LOSSLOG"
(
  # wait for the bundle (and its tensorboard dir) to be created by the logger
  for _ in $(seq 1 120); do
      TB=$(ls -d "$RESULTS"/*/tensorboard 2>/dev/null | head -1)
      [ -n "$TB" ] && break
      sleep 10
  done
  [ -n "${TB:-}" ] && "$REPO/.venv-hpc/bin/python" \
      "$REPO/results/diag/loss_watch.py" "$TB" "$LOSSLOG" 300
) &
WATCH_PID=$!
trap 'kill $MON_PID $WATCH_PID 2>/dev/null; rm -rf "$SCRATCH"' EXIT

# -- train -------------------------------------------------------------------
# CORRECTED stage-1 protocol. Run 555051 diverged using compression=32 and
# lr_steps=[15,65], both of which were read off the released NOISY/timedelay
# checkpoint's config. The shipped Perfect-setting config
# (v2x-vit hypes_yaml/point_pillar_v2xvit.yaml) is compression: 0 (:84),
# backbone_fix: false (:85), step_size: [15, 50] (:161), 60 epochs, batch 2,
# Adam 1e-3 -- and trainer/default.yaml + model/v2xvit.yaml already carry
# exactly those values, so NO protocol override is passed here.
#
# The two overrides below are OPERATIONAL, not protocol:
#   checkpoint_every=1  default is 10; we want every epoch recoverable
#   log_every=1         default is 20; per-step granularity for the loss log
#
# No max_batches: full epochs. No dataset.scenarios: every staged scenario.
# faults defaults to `none` (clean/Perfect). backbone_fix is not implemented
# anywhere in the package, which matches the reference's `false`.
echo "training start $(date -Is)" | tee -a "$PROGRESS"
"$REPO/.venv-hpc/bin/python" -m v2xvitbench.scripts.train \
    dataset=v2xset \
    "dataset.root=$DATA" \
    trainer=default \
    trainer.checkpoint_every=1 \
    trainer.log_every=1 \
    "results_dir=$RESULTS"
RC=$?

{
  echo "training end  $(date -Is)  exit=$RC"
  echo "checkpoints: $(ls -1 "$RESULTS"/*/checkpoints/*.pt 2>/dev/null | wc -l)"
  du -sh "$RESULTS" 2>/dev/null
} | tee -a "$PROGRESS"

kill $MON_PID 2>/dev/null
exit $RC
