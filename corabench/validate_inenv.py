"""In-env validation gate (A1): the synthetic forward/backward re-run inside
the ACTUAL training environment (opencood-official, torch 1.12) after the
compat port -- a port can reintroduce issues, so the gate must pass where
training will run, not only in .venv-hpc.

Feeds a synthetic batch in OpenCOOD's EXACT loader format (raw (P,32,4)
voxels, (z,y,x) coords, VoxelPostprocessor-style labels, 7-dim raw-delta-yaw
targets per B1) through CoRABatchAdapter.collate -> CoRAModel(reg_dim=7) ->
CoRALoss(reg_dim=7) -> backward under detect_anomaly.

    python -m corabench.validate_inenv [--small]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data.opencood_adapter import CoRABatchAdapter, translate_label
from .models import CoRAModel
from .selfcheck import static_source_checks
from .training import CoRALoss

VOXEL = (0.4, 0.4)
FULL_RANGE = (-140.8, -40.0, -3.0, 140.8, 40.0, 1.0)
SMALL_RANGE = (-19.2, -19.2, -3.0, 19.2, 19.2, 1.0)


def synth_agent(rng: np.random.RandomState, point_range, n_pillars: int):
    """One agent's voxel dict in OpenCOOD's exact preprocessor format."""
    gx = int(round((point_range[3] - point_range[0]) / VOXEL[0]))
    gy = int(round((point_range[4] - point_range[1]) / VOXEL[1]))
    coords = np.stack([np.zeros(n_pillars, np.int64),
                       rng.randint(0, gy, n_pillars),
                       rng.randint(0, gx, n_pillars)], axis=1)   # (z, y, x)
    npts = rng.randint(1, 33, n_pillars)
    feats = np.zeros((n_pillars, 32, 4), np.float32)
    for i in range(n_pillars):
        cx = point_range[0] + (coords[i, 2] + 0.5) * VOXEL[0]
        cy = point_range[1] + (coords[i, 1] + 0.5) * VOXEL[1]
        k = npts[i]
        feats[i, :k, 0] = cx + rng.uniform(-0.2, 0.2, k)
        feats[i, :k, 1] = cy + rng.uniform(-0.2, 0.2, k)
        feats[i, :k, 2] = rng.uniform(-1.0, 1.0, k)
        feats[i, :k, 3] = rng.uniform(0.3, 1.0, k)
    return {"features": torch.from_numpy(feats),
            "coords": torch.from_numpy(coords),
            "num_points": torch.from_numpy(npts.astype(np.int64))}


def synth_label(rng: np.random.RandomState, h: int, w: int, a: int = 2):
    pos = np.zeros((h, w, a), np.float32)
    neg = np.ones((h, w, a), np.float32)
    for _ in range(12):                          # a dozen positive anchors
        y, x, an = rng.randint(0, h), rng.randint(0, w), rng.randint(0, a)
        pos[y, x, an] = 1.0
        neg[y, x, an] = 0.0
    tgt = np.zeros((h, w, a * 7), np.float32)
    tgt[pos.repeat(7, axis=2) > 0] = rng.uniform(-1, 1)
    return {"pos_equal_one": pos, "neg_equal_one": neg, "targets": tgt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(4)
    rng = np.random.RandomState(7)

    print("torch %s" % torch.__version__)
    static_source_checks()
    print("ok   static source checks")

    point_range = SMALL_RANGE if args.small else FULL_RANGE
    gx = int(round((point_range[3] - point_range[0]) / VOXEL[0]))
    gy = int(round((point_range[4] - point_range[1]) / VOXEL[1]))
    h, w = gy // 2, gx // 2                      # feature map (stride 2)
    channels = 64 if args.small else 256

    adapter = CoRABatchAdapter(opencood_dataset=None, voxel_size=VOXEL,
                               point_range=point_range)
    items = []
    for counts in (2, 5):                        # a 2-agent and a 5-agent scene
        agents = [synth_agent(rng, point_range, rng.randint(400, 1200))
                  for _ in range(counts)]
        items.append({"agents": agents,
                      "targets": translate_label(synth_label(rng, h, w)),
                      "index": len(items)})
    batch = adapter.collate(items)
    print("agent_counts %s | voxel_features %s (10-dim decorated)"
          % (batch["agent_counts"], tuple(batch["voxel_features"].shape)))

    model = CoRAModel((gy, gx), channels=channels, reg_dim=7)
    model.train()
    dt0 = F.softplus(model.lc.cssm.scan.dt_bias.detach())
    print("delta init [%.2e, %.2e] in [1e-3,1e-1]: %s"
          % (dt0.min(), dt0.max(),
             bool((dt0 >= 0.999e-3).all() and (dt0 <= 1.001e-1).all())))
    g = model.lc.gate
    print("gate single-channel: %s" % (g.pre.out_channels == 1))

    t0 = time.time()
    with torch.autograd.set_detect_anomaly(True):
        out = model(batch)
        print("forward %.0fs" % (time.time() - t0))
        keys = ("local_cls", "local_reg", "cls_lc", "reg_lc", "cls_pac",
                "reg_pac", "cls_lc_recal", "cls_pac_recal", "u_lc", "u_pac")
        fin = all(bool(torch.isfinite(out[k]).all()) for k in keys)
        print("all outputs finite:", fin)
        for k in ("local_cls", "cls_lc", "cls_pac"):
            print("  init logit mean %-10s %.3f"
                  % (k, float(out[k].detach().mean())))
        parts = CoRALoss(reg_dim=7)(out, batch)
        for k in sorted(parts):
            print("loss %-16s %.4f" % (k, float(parts[k].detach())))
        t0 = time.time()
        parts["loss"].backward()
        print("backward %.0fs" % (time.time() - t0))

    nan_p = [n for n, p in model.named_parameters()
             if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    no_g = [n for n, p in model.named_parameters() if p.grad is None]
    print("NaN/inf-grad params:", len(nan_p), nan_p[:4])
    print("no-grad params     :", len(no_g), no_g[:4])
    gmax = max(float(p.grad.abs().max()) for p in model.parameters()
               if p.grad is not None)
    print("grad |max|: %.3e" % gmax)
    ok = fin and not nan_p and not no_g and bool(
        torch.isfinite(parts["loss"]))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
