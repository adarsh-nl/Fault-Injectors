#!/bin/bash
# Nreg 7->8 verification: the two shared-code guarantees, off the login node.
#   mkdir -p logs && sbatch corabench/slurm/verify_regdim_branch.sbatch
#
# NOT a training run. CPU only, no GPU requested -- nothing here needs one.
#
# GUARANTEE 1  the shared decode in cpbench/data/postprocessing.py BRANCHES on
#              reg_dim (7 -> asin, 8 -> atan2), never reads rg[:, 7] at
#              reg_dim 7 where that column does not exist, and its 7-branch is
#              BYTE-IDENTICAL to the pre-change code (diffed below against
#              20b6547^ -- that identity is what keeps cobevt / v2xvit /
#              where2comm / lgcp safe, since none of them call the assertion).
# GUARANTEE 2  a known yaw round-trips through DECODE at BOTH widths, and a
#              real mismatch fires the startup assertion.
#
#   Case A  reg_dim=7  legacy packages (cobevt, v2xvit, where2comm, lgcp)
#   Case B  reg_dim=8  corabench
#   Case C  mismatch   assertion must FIRE
# then the four idle packages run as a canary, because the whole point of the
# change being strictly additive is that they are untouched.
#
# ORDER IS DELIBERATE: the three cases you actually read run FIRST and cheap.
# The full regression and the four-package canary run LAST, each under its own
# `timeout`, so a stall in the slow suites still leaves the three-case verdict
# on disk instead of losing everything to the wall clock (job 551932 lost its
# entire run that way).
#
# NOTE logs/ is gitignored (a076f70) and therefore does NOT exist in a fresh
# clone. Slurm opens --output BEFORE the script body runs, so an in-script
# mkdir is too late: job 551913 died in 27 s with empty stderr for exactly
# that reason. Run `mkdir -p logs` once before the first submit.
#
# NOTE no heredocs, and nothing buffered. Job 551932 hung for its full 40 min
# wall-clock at a `python - <<'PY' ... PY` block nested in a `{ ... } | tee`
# group. Its successor piped every case through `tail -12`, which withholds ALL
# output until the command exits -- a slow case is then indistinguishable from
# a hang. Everything here streams line by line through a plain `tee -a`, and
# the report lives in a real file (_verify_regdim_report.py), not a heredoc.
#SBATCH --job-name=cora-regdim
#SBATCH --partition=ps,main-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

REPO="$HOME/Fault-Injectors"
OUT="$HOME/verify-results"
mkdir -p "$OUT"
RESULT="$OUT/regdim-branch-${SLURM_JOB_ID}.txt"

module purge
unset PYTHONHOME PYTHONSTARTUP

cd "$REPO" || exit 1
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PY=".venv-hpc/bin/python"

say () { echo "$@" | tee -a "$RESULT"; }

FAILED=""

# Every step is wrapped: streams live, dies at its own timeout rather than
# taking the whole job down with it, and records a verdict either way.
run () {
    local name="$1" limit="$2"; shift 2
    say "=== $name ==="
    say "    started $(date +%H:%M:%S), timeout ${limit}s"
    timeout "$limit" "$@" 2>&1 | tee -a "$RESULT"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 124 ]; then
        say "--> $name TIMED OUT after ${limit}s"
        FAILED="$FAILED
    $name (timeout)"
    elif [ "$rc" -ne 0 ]; then
        say "--> $name FAILED exit=$rc"
        FAILED="$FAILED
    $name (exit $rc)"
    else
        say "--> $name ok"
    fi
    say ""
}

say "job    : ${SLURM_JOB_ID}  node ${SLURMD_NODENAME:-?}"
say "commit : $(git rev-parse --short HEAD 2>/dev/null || echo '?')  (UNCOMMITTED working tree)"
say "date   : $(date -Is)"
say "python : $($PY -V 2>&1)"
say ""

# ---------------------------------------------------------------------------
# GUARANTEE 1, part 1: the branch itself, read straight out of the source.
# ---------------------------------------------------------------------------
say "=== GUARANTEE 1a: the shared decode branch (postprocessing.py) ==="
sed -n '/THE SHARED DECODE BRANCH/,/return boxes/p' \
    cpbench/data/postprocessing.py | tee -a "$RESULT"
say ""

# ---------------------------------------------------------------------------
# GUARANTEE 1, part 2: the 7-branch is unchanged from before the Nreg change.
# This is the load-bearing proof for the four packages that never call the
# assertion. An EMPTY diff on the yaw line is the pass condition.
# ---------------------------------------------------------------------------
say "=== GUARANTEE 1b: reg_dim-7 yaw math vs 20b6547^ (pre-Nreg) ==="
say "  before:"
git show 20b6547^:cpbench/data/postprocessing.py \
    | grep -n "arcsin" | sed 's/^/    /' | tee -a "$RESULT"
say "  after (the else-branch):"
grep -n "np.arcsin" cpbench/data/postprocessing.py | sed 's/^/    /' \
    | tee -a "$RESULT"
say ""
OLD_YAW=$(git show 20b6547^:cpbench/data/postprocessing.py \
    | grep "boxes\[:, 6\] = " | tr -d '[:space:]')
NEW_YAW=$(grep "boxes\[:, 6\] = an\[:, 6\] + np.arcsin" \
    cpbench/data/postprocessing.py | tr -d '[:space:]')
if [ "$OLD_YAW" = "$NEW_YAW" ] && [ -n "$OLD_YAW" ]; then
    say "--> reg_dim-7 yaw decode is BYTE-IDENTICAL to pre-change"
else
    say "--> reg_dim-7 yaw decode CHANGED -- the four legacy packages are at risk"
    say "    old: $OLD_YAW"
    say "    new: $NEW_YAW"
    FAILED="$FAILED
    GUARANTEE 1b (7-branch math changed)"
fi
say ""

say "=== the autograd twin (corabench/fusion/pac.py) ==="
grep -n "torch.atan2\|torch.asin" corabench/fusion/pac.py | tee -a "$RESULT"
say ""

say "=== the single source of truth (config) ==="
grep -n "reg_dim" corabench/configs/model/cora.yaml | tee -a "$RESULT"
say ""

# ---------------------------------------------------------------------------
# The three cases, as readable numbers first, then as assertions.
# ---------------------------------------------------------------------------
run "REPORT  A/B/C round-trip numbers + assertion behaviour" 600 \
    "$PY" corabench/slurm/_verify_regdim_report.py

run "CASE A  reg_dim=7 legacy: assertion passes + yaw round-trips (asin)" 600 \
    "$PY" -m pytest corabench/tests/test_reg_dim.py -k "case_a" -v --no-header

run "CASE B  reg_dim=8 corabench: assertion passes + yaw round-trips (atan2)" 600 \
    "$PY" -m pytest corabench/tests/test_reg_dim.py -k "case_b or decodes_agree" \
    -v --no-header

run "CASE C  mismatch: assertion FIRES" 600 \
    "$PY" -m pytest corabench/tests/test_reg_dim.py -k "case_c or agreement" \
    -v --no-header

run "NO DEFAULT MOVED (every shared constructor still 7)" 300 \
    "$PY" -m pytest corabench/tests/test_reg_dim.py -k "default_moved" -v --no-header

# ---------------------------------------------------------------------------
# The expensive suites, last on purpose.
# ---------------------------------------------------------------------------
run "LAYERING (the cross-package contract)" 600 \
    "$PY" -m pytest cpbench/tests/test_layering.py -q --no-header

run "REGRESSION (cpbench + corabench)" 1200 \
    "$PY" -m pytest cpbench corabench/tests -q --no-header

run "IDLE CANARY (lgcp, cobevt, w2c, v2xvit) -- must be untouched" 1200 \
    "$PY" -m pytest lgcpbench cobevtbench w2cbench v2xvitbench -q --no-header

# ---------------------------------------------------------------------------
say "================== SUMMARY =================="
if [ -n "$FAILED" ]; then
    say "VERDICT: FAIL"
    say "failed steps:$FAILED"
    say "results written to $RESULT"
    exit 1
fi
say "VERDICT: PASS -- both guarantees hold, no default moved,"
say "         four legacy packages unaffected"
say "results written to $RESULT"
