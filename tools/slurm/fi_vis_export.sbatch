#!/bin/bash
#SBATCH --job-name=fi-vis-export
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fault-visualisation exports (CPU ONLY -- never touches the GPU queue).
# Invocation 1: OpenCOOD datasets in the opencood-official env.
# Invocation 2: Griffin in .venv-hpc (plyfile). INDEX appended, not clobbered.

set -uo pipefail
REPO="$HOME/Fault-Injectors"
OCOOD="$HOME/opencood-official"
cd "$OCOOD"

source activate opencood-official
PYTHONPATH="$OCOOD:$REPO" python -u \
    "$REPO/notebooks/fault_visualisation.py" --export --datasets v2xset,opv2v
RC1=$?
conda deactivate

cd "$REPO"
PYTHONPATH="$REPO" "$REPO/.venv-hpc/bin/python" -u \
    notebooks/fault_visualisation.py --export --datasets griffin --append-index
RC2=$?

N=$(find "$REPO/results/fault_injector_visualisation" -type f | wc -l)
echo "EXPORTS DONE: rc=$RC1/$RC2, $N files in results/fault_injector_visualisation"
exit $(( RC1 || RC2 ))
