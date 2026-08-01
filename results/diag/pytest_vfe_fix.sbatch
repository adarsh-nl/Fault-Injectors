#!/bin/bash
#SBATCH --job-name=pytestvfe
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Test suite for the 9 -> 10 pillar-feature-decoration fix (debug_log DL-006).
#
# SCOPE, deliberately narrow: cpbench (where the fix lives) and v2xvitbench
# (the fastest-path bench). corabench, lgcpbench, cobevtbench and w2cbench are
# DEFERRED -- they carry pre-existing reconstruction/broken state per
# docs/repository_audit.md, so their failures would not be attributable to this
# change and would drown the signal.
#
# --doctest-modules is not optional here: several of the edits were doctests.
#
# CPU only. No GPU, no training, no data. The venv is imported straight from
# NFS -- do NOT rsync it to /local, that is slower (DL-001). Budget the
# ~15-20 min torch-import tax before pytest emits its first line.

set -uo pipefail          # NOTE: no -e; see the exit-code capture below

REPO="$HOME/Fault-Injectors"
OUT="$REPO/results/diag/pytest_vfe_fix.txt"
SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

cd "$REPO"

# Keep any test that writes a results bundle off /home.
export CPBENCH_RESULTS_DIR="$SCRATCH/results"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$CPBENCH_RESULTS_DIR" "$TMPDIR"

RAW="$SCRATCH/pytest_raw.txt"

{
  echo "# pytest run for the 9->10 pillar decoration fix (DL-006)"
  echo "# job      : ${SLURM_JOB_ID:-none} on $(hostname)"
  echo "# started  : $(date -Is)"
  echo "# scope    : cpbench v2xvitbench --doctest-modules"
  echo "# deferred : corabench lgcpbench cobevtbench w2cbench (pre-existing state, audit)"
  echo "# git HEAD : $(git rev-parse --short HEAD 2>/dev/null)  (uncommitted working tree)"
  echo
} > "$OUT"

echo "== launching pytest (expect ~15-20 min of silence: torch import from NFS)"

# `set -e` is OFF and the exit code is captured explicitly: a failing test suite
# is the *expected* outcome to record, not a reason to abort before writing the
# gate file. PIPESTATUS[0] is pytest's code, not tee's.
# No --timeout flag: pytest-timeout is not in requirements-hpc.lock.txt (the
# venv has pytest and nothing else), and an unrecognised flag would abort pytest
# before a single test ran. The job's --time is the timeout.
"$REPO/.venv-hpc/bin/python" -m pytest \
    cpbench v2xvitbench \
    --doctest-modules \
    -rf -q 2>&1 | tee "$RAW"
RC=${PIPESTATUS[0]}

{
  echo "PYTEST_EXIT_CODE=$RC"
  echo
  echo "# --- summary line(s) as pytest printed them ---"
  grep -E "^(=+ .*(passed|failed|error|no tests).*=+|[0-9]+ (passed|failed))" "$RAW" | tail -5
  echo
  echo "# --- short test summary (failures only) ---"
  sed -n '/short test summary info/,$p' "$RAW" | head -80
  echo
  echo "# --- last 40 lines of raw output ---"
  tail -40 "$RAW"
  echo
  echo "# finished : $(date -Is)"
} >> "$OUT"

cp "$RAW" "$REPO/results/diag/pytest_vfe_fix_raw.txt" 2>/dev/null || true

echo "== wrote $OUT (exit $RC)"
exit "$RC"
