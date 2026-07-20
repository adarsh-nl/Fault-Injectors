#!/bin/bash
# Build the Python 3.7 OpenCOOD environment on the UT EEMCS HPC.
#   sbatch lgcpbench/slurm/opencood_env.sbatch
#
# Why a SEPARATE environment
#   OpenCOOD pins numba==0.49.0, which does not build on modern Python, so it
#   is effectively locked to Python 3.7 regardless of the torch version. It
#   also needs spconv (unpinned upstream; pinned here) and CUDA. lgcpbench's
#   core deliberately does not depend on any of that -- only
#   lgcpbench/perception/opencood/ does -- so the two environments coexist:
#
#     .venv-hpc        py3.9+  core, control plane, native backbone, CPU sweeps
#     ~/.conda/envs/opencood-py37   OpenCOOD models + pretrained weights
#
# This job is idempotent: rerunning it verifies rather than rebuilds.
#SBATCH --job-name=lgcp-ocenv
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO="${REPO:-$HOME/Fault-Injectors}"
ENV_NAME="${ENV_NAME:-opencood-py37}"
OPENCOOD_DIR="${OPENCOOD_DIR:-$HOME/OpenCOOD}"
# Pin the upstream commit so a reproduced Table II is attributable to a
# specific OpenCOOD revision rather than to whatever main happened to be.
OPENCOOD_COMMIT="${OPENCOOD_COMMIT:-main}"
# spconv is NOT pinned upstream; pin it here. cu113 matches the torch build
# below. Change both together or neither.
SPCONV_PKG="${SPCONV_PKG:-spconv-cu113}"

module purge
module load nvidia/cuda-11.8 2>/dev/null || module load nvidia/cuda-12.4 || true
module load miniconda3 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Load a conda module (see" \
         "https://hpc.wiki.utwente.nl/eemcs-hpc:software) or install miniconda" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "== creating conda env ${ENV_NAME} (python 3.7)"
    conda create -y -n "$ENV_NAME" python=3.7
fi
conda activate "$ENV_NAME"

echo "== python: $(python --version 2>&1)"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 7):
    raise SystemExit(
        f"expected Python 3.7, got {sys.version_info[:2]}. OpenCOOD's "
        f"numba==0.49.0 pin will not build on anything newer."
    )
PY

echo "== torch (OpenCOOD is tested on 1.8-1.12)"
pip install --upgrade pip
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

echo "== spconv: ${SPCONV_PKG}"
pip install "$SPCONV_PKG"

if [ ! -d "$OPENCOOD_DIR" ]; then
    echo "== cloning OpenCOOD"
    git clone https://github.com/DerrickXuNu/OpenCOOD.git "$OPENCOOD_DIR"
fi
cd "$OPENCOOD_DIR"
git fetch --all --quiet
git checkout "$OPENCOOD_COMMIT"
RESOLVED_SHA="$(git rev-parse HEAD)"
echo "== OpenCOOD at ${RESOLVED_SHA}"

pip install -r requirements.txt
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace || \
    echo "WARNING: pcdet_utils build failed; needed only by some models"

# Record the exact revision so a results directory can be traced back to it.
mkdir -p "$REPO/results"
cat > "$REPO/results/opencood_env.json" <<JSON
{
  "conda_env": "${ENV_NAME}",
  "opencood_dir": "${OPENCOOD_DIR}",
  "opencood_commit": "${RESOLVED_SHA}",
  "spconv": "${SPCONV_PKG}",
  "python": "$(python --version 2>&1)",
  "torch": "$(python -c 'import torch; print(torch.__version__)')"
}
JSON

echo "== verifying the adapter can import OpenCOOD"
cd "$REPO"
PYTHONPATH="$REPO" python - <<'PY'
from opencood.tools import train_utils
from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
assert hasattr(train_utils, "create_model")
print("OK: OpenCOOD importable, create_model and SpVoxelPreprocessor present")
PY

echo
echo "Done. Run the paper models with, e.g.:"
echo "  sbatch lgcpbench/slurm/benchmark_array.sbatch \\"
echo "     model=where2comm dataset=opv2v \\"
echo "     model.hypes_yaml=$OPENCOOD_DIR/opencood/hypes_yaml/point_pillar_where2comm.yaml \\"
echo "     model.checkpoint=/path/to/net_epoch_bestval_at30.pth"
