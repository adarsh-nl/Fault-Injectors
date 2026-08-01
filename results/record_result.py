#!/usr/bin/env python3
"""Write ONE evaluation cell as an atomic JSON file. Source of truth is cells/.

A finished eval job calls this once. Filename is derived from the cell key so
distinct cells never collide and a rerun overwrites only its own cell. Write is
tmp+rename on the same filesystem so a half-written file is never observed.

Clean baseline is recorded as its own cell with --fault clean (no severity).
Faulty cells store only their own AP; retention is computed at render time
against the run's clean cell, so the clean number lives in exactly one place.
"""
import argparse, json, os, subprocess, tempfile, time

def git_commit(cwd):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells-dir", default=os.path.join(os.path.dirname(__file__), "cells"))
    p.add_argument("--operator", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--fault", required=True, help="pose_error|comm_latency|bandwidth_limit|fog|snow|agentdrop|clean")
    p.add_argument("--severity", default="clean", help="e.g. 0.6m, 400ms, dense, or an agent-count stratum for agentdrop")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--ap50", type=float, required=True)
    p.add_argument("--ap70", type=float, required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--faultinj-repo", default=".", help="repo dir to stamp git commit from")
    p.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    p.add_argument("--node", default=os.environ.get("SLURMD_NODENAME", ""))
    p.add_argument("--guards-ok", action="store_true", help="set only if pre-flight guards passed for this run")
    a = p.parse_args()

    os.makedirs(a.cells_dir, exist_ok=True)
    sev = "clean" if a.fault == "clean" else a.severity
    key = f"{a.operator}__{a.dataset}__{a.fault}__{sev}__seed{a.seed}"
    rec = {
        "operator": a.operator, "dataset": a.dataset, "fault": a.fault,
        "severity": sev, "seed": a.seed, "checkpoint": a.checkpoint,
        "metric": {"ap_50": a.ap50, "ap_70": a.ap70},
        "faultinj_commit": git_commit(a.faultinj_repo),
        "job_id": a.job_id, "node": a.node, "guards_ok": bool(a.guards_ok),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    dst = os.path.join(a.cells_dir, key + ".json")
    fd, tmp = tempfile.mkstemp(dir=a.cells_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    os.replace(tmp, dst)
    print("wrote", dst)

if __name__ == "__main__":
    main()
