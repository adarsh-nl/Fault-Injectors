"""Single-sample forward-pass sanity check on one real V2XSet scene.

Deliberately NOT an evaluation: it builds the test-split dataset, takes ONE
sample, collates a batch of one, runs one forward under ``no_grad`` and
reports shapes and magnitudes. No metric is computed, no checkpoint is
loaded (the model is randomly initialised -- the question is whether the
pipeline is wired, not whether it is accurate), no fault is injected.

What it is checking, in order of what would actually be wrong:

  1. the adapter reads the real scenario directory at all
  2. the agent count survives collation (record_len vs. the agents on disk)
  3. infrastructure agents are typed as infrastructure, not as vehicles
  4. every tensor entering the model has the shape the config implies
  5. the forward produces finite values of a plausible magnitude
  6. the agent mask is not all-zero and not all-one

(6) matters because both degenerate masks produce a forward pass that looks
healthy. An all-zero mask silently discards every collaborator; an all-one
mask silently treats padding as real. Neither raises.

Configured entirely by environment:
    PROBE_PKG        v2xvitbench | cobevtbench | w2cbench | corabench
    PROBE_OVERRIDES  space-separated config overrides
    PROBE_OUT        json output path
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

PKG = os.environ["PROBE_PKG"]
OVERRIDES = os.environ.get("PROBE_OVERRIDES", "").split()
OUT = Path(os.environ.get("PROBE_OUT", REPO / "results" / "diag"
                          / f"{PKG}_forward_probe.json"))


def stats(t):
    """Magnitude summary that survives an all-NaN tensor."""
    if not torch.is_tensor(t):
        return {"not_a_tensor": str(type(t))}
    f = t.detach().float()
    finite = torch.isfinite(f)
    d = {"shape": list(t.shape), "dtype": str(t.dtype),
         "n_nan": int(torch.isnan(f).sum()), "n_inf": int(torch.isinf(f).sum()),
         "n_finite": int(finite.sum()), "numel": int(f.numel())}
    if finite.any():
        g = f[finite]
        d.update(min=round(float(g.min()), 6), max=round(float(g.max()), 6),
                 mean=round(float(g.mean()), 6), std=round(float(g.std()), 6)
                 if g.numel() > 1 else 0.0,
                 frac_zero=round(float((g == 0).float().mean()), 6))
    return d


def main():
    report = {"pkg": PKG, "overrides": OVERRIDES,
              "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(),
              "device_name": (torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else None),
              "slurm_job": os.environ.get("SLURM_JOB_ID"),
              "node": os.environ.get("SLURMD_NODENAME")}
    try:
        common = __import__(f"{PKG}.scripts.common", fromlist=["common"])
        cfg = common.load(OVERRIDES)
        report["grid"] = {
            "voxel_size": cfg["dataset"]["grid"]["voxel_size"],
            "point_range": cfg["dataset"]["grid"]["point_range"],
            "downsample": cfg["dataset"]["grid"].get("downsample"),
            "max_cav": cfg["dataset"].get("max_cav"),
            "root": str(cfg["dataset"].get("root")),
            "scenarios": cfg["dataset"].get("scenarios")}

        # -- 1. dataset ---------------------------------------------------
        dataset = common.build_dataset(cfg, bridge=None, split="test")
        report["dataset"] = {"class": type(dataset).__name__,
                             "frames": len(dataset)}

        sample = dataset[0]
        report["sample_keys"] = sorted(
            k for k in sample if not k.startswith("_"))
        report["sample"] = {k: stats(v) for k, v in sample.items()
                            if torch.is_tensor(v)}
        for k, v in sample.items():
            if not torch.is_tensor(v) and k in (
                    "record_len", "agent_types", "n_agents", "is_infra",
                    "gt_boxes", "delays", "ego_index"):
                report["sample"][k] = repr(v)[:400]

        # -- 2. collate ---------------------------------------------------
        batch = common.build_collator(cfg)([sample])
        report["batch"] = {k: stats(v) for k, v in batch.items()
                           if torch.is_tensor(v)}
        report["batch_nontensor"] = {
            k: repr(v)[:400] for k, v in batch.items()
            if not torch.is_tensor(v)}

        # -- 3. model -----------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = common.build_model(cfg).to(device).eval()
        report["model"] = {
            "class": type(model).__name__,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_tensors": len(model.state_dict())}

        moved = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            out = model(moved)
        report["output"] = {k: stats(v) for k, v in out.items()
                            if torch.is_tensor(v)}
        report["output_nontensor"] = {
            k: repr(v)[:400] for k, v in out.items()
            if not torch.is_tensor(v)}

        # -- 4. the checks that do not raise on their own ------------------
        checks = {}
        bad = [k for k, s in report["output"].items()
               if s.get("n_nan") or s.get("n_inf")]
        checks["output_all_finite"] = not bad
        checks["output_nonfinite_keys"] = bad

        for name in ("agent_mask", "mask", "spatial_mask", "com_mask"):
            if name in out and torch.is_tensor(out[name]):
                m = out[name].detach().float()
                frac = float(m.mean())
                checks[f"{name}_frac_on"] = round(frac, 6)
                checks[f"{name}_degenerate"] = frac in (0.0, 1.0)
        checks["cls_not_constant"] = bool(
            "cls" in out and torch.is_tensor(out["cls"])
            and float(out["cls"].detach().float().std()) > 0)
        report["checks"] = checks
        report["status"] = "OK"

    except Exception:
        report["status"] = "ERROR"
        report["traceback"] = traceback.format_exc()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
