#!/bin/bash
#SBATCH --job-name=keymatch2
#SBATCH --partition=ps,main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# CPU-only, read-only diagnostic: dump official-checkpoint and reimpl
# state_dict keys+shapes for all four benches and compare them.
# No GPU, no training, no data loading, no fault injection.
#
# The venv is copied to node-local NVMe first: importing torch straight from
# the NFS home stalls for minutes on a loaded login node (D-state on
# rpc_wait_bit_killable). /local is dual-NVMe and node-local.

set -euo pipefail

REPO="$HOME/Fault-Injectors"
SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "node=$(hostname) jobid=$SLURM_JOB_ID"

# -- stage the interpreter ---------------------------------------------------
# no venv copy: import straight from NFS on the compute node
# a relocated venv needs its home repointed; the interpreter itself is a
# symlink into /software, which is shared and fine to follow
:
PY="$REPO/.venv-hpc/bin/python"

# -- stage the checkpoints ---------------------------------------------------
CKPT="$SCRATCH/ckpt"
mkdir -p "$CKPT"
unzip -o -q /datasets/eemcs/ps/cv/opencood/v2xset_checkpoints/cobevt_lidar.zip \
    -d "$CKPT/cobevt"
unzip -o -q /datasets/eemcs/ps/cv/opencood/v2xset_checkpoints/point_pillar_where2comm_v2xset.zip \
    -d "$CKPT/where2comm"
mkdir -p "$CKPT/v2xvit"
cp /datasets/eemcs/ps/cv/opencood/v2xset_checkpoints/v2x-vit/* "$CKPT/v2xvit/"
# the OPV2V zoo: CoBEVT's own OPV2V instantiations (camera seg + lidar det)
mkdir -p "$CKPT/opv2v_zoo"
OPV2V_MODELS=/datasets/eemcs/ps/cv/opencood/opv2v
unzip -o -q "$OPV2V_MODELS/CoBEVT_Models-20240806T195322Z-001/CoBEVT_Models/cobevt.zip" \
    -d "$CKPT/opv2v_zoo/cobevt_camera_dynamic"
unzip -o -q "$OPV2V_MODELS/CoBEVT_Models-20240806T195322Z-001/CoBEVT_Models/cobevt_static.zip" \
    -d "$CKPT/opv2v_zoo/cobevt_camera_static"
unzip -o -q "$OPV2V_MODELS/Models-20240806T195353Z-001/Models/pointpillar_cobevt/pointpillar_CoBEVT_nocompression.zip" \
    -d "$CKPT/opv2v_zoo/cobevt_lidar_opv2v_nocomp"

find "$CKPT" -name "*.pth" -printf "%p\t%s\n"

# -- run ---------------------------------------------------------------------
cd "$REPO"
OMP_NUM_THREADS=4 CKPT_ROOT="$CKPT" "$PY" "$REPO/results/diag/keymatch_diag.py"

echo "done"
