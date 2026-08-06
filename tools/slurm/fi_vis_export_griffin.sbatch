#!/bin/bash
#SBATCH --job-name=fi-vis-griffin
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Griffin half of the visualisation exports (CPU only).
#
# Runs .venv-hpc's interpreter DIRECTLY -- no `conda deactivate`, which is
# what killed job 558138: conda's activate.d scripts reference unbound
# variables (QT_XCB_GL_INTEGRATION) and `set -u` turns that into a fatal
# error. A venv interpreter needs no conda state torn down.
#
# --append-index adds to the existing INDEX.md rather than clobbering the
# 56 OpenCOOD entries already written.

set -uo pipefail
REPO="$HOME/Fault-Injectors"
cd "$REPO"

PYTHONPATH="$REPO" "$REPO/.venv-hpc/bin/python" -u \
    notebooks/fault_visualisation.py --export --datasets griffin-rest --append-index
RC=$?

N=$(find "$REPO/results/fault_injector_visualisation" -type f | wc -l)
echo "GRIFFIN EXPORTS DONE: rc=$RC, $N files total in results/fault_injector_visualisation"
exit $RC
