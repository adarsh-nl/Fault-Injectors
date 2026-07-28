#!/bin/bash
# Phase 2 verification: run the test suites off the login node.
#   sbatch corabench/slurm/verify_regdim.sbatch <label>
#
# NOT a training run. CPU only, no GPU requested. This exists because the
# login node cannot carry the before/after discipline Phase 2 depends on:
# during the 2026-07-28 session it sat at load 22-113 with 44 users, the
# three-package suite was SIGTERM'd at the 900 s timeout, the tightened
# two-package suite was SIGTERM'd at 900 s as well, and finally `git status`
# itself timed out at 120 s. CLAUDE.md already requires anything over a
# minute to go through sbatch; this is that.
#
# The label argument tags the output so BEFORE and AFTER runs are separable:
#   sbatch corabench/slurm/verify_regdim.sbatch before_regdim7
#   sbatch corabench/slurm/verify_regdim.sbatch after_regdim7
#   sbatch corabench/slurm/verify_regdim.sbatch after_regdim8
#
# Results land in $HOME/verify-results/<label>-<jobid>.txt and the summary
# line is echoed to the job's own stdout, so `tail` on either works.
#SBATCH --job-name=cora-verify
#SBATCH --partition=ps,main-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=verify-%j.out
#SBATCH --error=verify-%j.err

set -euo pipefail

LABEL="${1:-unlabelled}"
REPO="$HOME/Fault-Injectors"
OUT="$HOME/verify-results"
mkdir -p "$OUT"
RESULT="$OUT/${LABEL}-${SLURM_JOB_ID}.txt"

module purge
unset PYTHONHOME PYTHONSTARTUP

cd "$REPO"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

{
    echo "label      : $LABEL"
    echo "job        : ${SLURM_JOB_ID}  node ${SLURMD_NODENAME:-?}"
    echo "commit     : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "date       : $(date -Is)"
    echo
    # The reg_dim fingerprint: makes it impossible to mistake a BEFORE run for
    # an AFTER one. A contaminated baseline already cost one cycle this
    # session -- the files were edited while the suite that imports them was
    # still running -- so the state is recorded IN the result file.
    echo "reg_dim occurrences per shared file:"
    for f in cpbench/models/heads.py cpbench/data/preprocessing.py \
             cpbench/data/postprocessing.py cpbench/training/losses.py; do
        printf "   %-40s %s\n" "$f" "$(grep -c reg_dim "$f" || true)"
    done
    echo
} | tee "$RESULT"

# Two scopes. The first is the pass/fail gate for the reg_dim refactor; the
# second is the layering canary that proves the four idle packages are
# untouched by a change to shared cpbench.
run_suite () {
    local name="$1"; shift
    echo "=== $name ===" | tee -a "$RESULT"
    set +e
    .venv-hpc/bin/python -m pytest "$@" -q 2>&1 | tail -25 | tee -a "$RESULT"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "--> $name exit=$rc" | tee -a "$RESULT"
}

run_suite "CHANGING (cpbench + corabench)" cpbench corabench/tests
run_suite "IDLE CANARY (lgcp, cobevt, w2c, v2xvit)" \
    lgcpbench cobevtbench w2cbench v2xvitbench
run_suite "LAYERING" cpbench/tests/test_layering.py

echo | tee -a "$RESULT"
echo "results written to $RESULT" | tee -a "$RESULT"
