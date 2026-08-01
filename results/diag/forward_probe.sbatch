#!/bin/bash
#SBATCH --job-name=fwdprobe
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ONE single-sample forward pass on ONE real V2XSet test scene.
#
# This is a wiring check, not an evaluation. It loads one frame, collates a
# batch of one, runs one forward under no_grad, and reports shapes,
# magnitudes, NaN/Inf counts, the agent count and the agent mask occupancy.
# No metric is computed. No checkpoint is loaded. No fault is injected.
#
# --constraint=l40 is mandatory: ps,main-gpu spans three GPU generations and
# .venv-hpc's torch 2.13.0+cu130 does not target sm_75. An unpinned job that
# lands on a Turing node aborts the CUDA context with an empty stderr
# (job 549175, dead in 3 s).
#
# The venv is imported straight from NFS. Do NOT rsync it to /local first --
# measured at >14 min on 2026-07-31 versus under a minute for the direct
# import (docs/debug_log.md DL-001).

set -euo pipefail

REPO="$HOME/Fault-Injectors"
SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

PKG="${PROBE_PKG:?set PROBE_PKG, e.g. v2xvitbench}"

# One V2XSet test scenario, staged node-local. 2021_10_26_22_12_49 is chosen
# deliberately: 5 agents (the configured max_cav) AND an infrastructure agent
# (-1), so the agent axis and the vehicle/infra typing are both exercised.
SCENARIO="${PROBE_SCENARIO:-2021_10_26_22_12_49}"
SRC=/datasets/eemcs/ps/cv/opencood/v2xset/test/"$SCENARIO"

mkdir -p "$SCRATCH/v2xset/test/$SCENARIO"
# LiDAR + labels only; the camera PNGs are ~10x the bytes and unused here.
rsync -a --include='*/' --include='*.pcd' --include='*.yaml' --exclude='*' \
    "$SRC/" "$SCRATCH/v2xset/test/$SCENARIO/"

echo "node=$(hostname) jobid=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "scenario=$SCENARIO agents=$(ls "$SCRATCH/v2xset/test/$SCENARIO" | wc -l)"
du -sh "$SCRATCH/v2xset"

cd "$REPO"

# NOTE the quoting on scenarios: YAML 1.1 reads 2021_10_26_22_12_49 as the
# integer 20211026221249 and it then matches no directory.
PROBE_PKG="$PKG" \
PROBE_OVERRIDES="${PROBE_OVERRIDES:-dataset=v2xset} dataset.root=$SCRATCH/v2xset dataset.scenarios=[\"$SCENARIO\"]" \
PROBE_OUT="$REPO/results/diag/${PKG}_forward_probe.json" \
    "$REPO/.venv-hpc/bin/python" "$REPO/results/diag/forward_probe.py"

echo "done"
