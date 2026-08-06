"""CoRA training on OpenCOOD's actual loader (A1+B1 path).

Smoke mode (--max_steps N) runs N steps with per-step loss logging -- the
gate between "won't NaN" (synthetic pass) and "learns" (this): the loss must
descend past step 14 (the historical divergence point of the previous
implementation) without diverging. Full mode (--epochs) is the 30-epoch run,
gated separately.

    python -m corabench.scripts.train_opencood \
        --hypes <config.yaml> --out <dir> --max_steps 400
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

from ..compat import autocast, grad_scaler
from ..data.opencood_adapter import CoRABatchAdapter
from ..models import CoRAModel
from ..selfcheck import grad_step_guard, static_source_checks
from ..training import CoRALoss


def build(hypes_path: str, train: bool):
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import IntermediateFusionDataset
    hypes = yaml_utils.load_yaml(hypes_path)
    ds = IntermediateFusionDataset(params=hypes, visualize=False, train=train)
    pre = hypes["preprocess"]
    adapter = CoRABatchAdapter(ds, pre["args"]["voxel_size"][:2],
                               pre["cav_lidar_range"])
    rng = pre["cav_lidar_range"]
    vx, vy = pre["args"]["voxel_size"][:2]
    grid_hw = (int(round((rng[4] - rng[1]) / vy)),
               int(round((rng[3] - rng[0]) / vx)))
    return adapter, grid_hw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="> 0 = smoke mode: stop after N optimiser steps")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--ckpt_every", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    static_source_checks()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter, grid_hw = build(args.hypes, train=True)
    # OpenCOOD's anchor/target grid, probed from a real item -- the canvas
    # and the anchor grid differ by stride 4, and assuming their relation
    # cost the first smoke attempt (job 557631).
    head_hw = tuple(adapter[0]["targets"]["cls_target"].shape[:2])
    print("dataset %d samples | canvas %s | head/anchor grid %s | device %s"
          % (len(adapter), grid_hw, head_hw, device), flush=True)

    model = CoRAModel(grid_hw, channels=args.channels, reg_dim=7,
                      head_hw=head_hw).to(device)
    loss_fn = CoRALoss(reg_dim=7).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [15, 25], gamma=0.1)
    amp = args.amp and device == "cuda"
    scaler = grad_scaler(enabled=amp)

    loader = torch.utils.data.DataLoader(
        adapter, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=adapter.collate,
        drop_last=True)

    log_path = os.path.join(args.out, "smoke_loss.csv" if args.max_steps
                            else "train_loss.csv")
    fh = open(log_path, "w", newline="")
    writer = None

    step = 0
    skipped = 0
    t0 = time.time()
    # feature-map/target-shape agreement asserted on the first batch below.
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with autocast(device, enabled=amp):
                out = model(batch)
                if step == 0:
                    assert out["cls_lc"].shape[-2:] == \
                        batch["cls_target"].shape[1:3], \
                        "feature map does not match the hypes' target grid"
                parts = loss_fn(out, batch)
            scaler.scale(parts["loss"]).backward()
            scaler.unscale_(opt)
            stepped, gnorm = grad_step_guard(model, opt, max_norm=10.0)
            scaler.update()
            if stepped:
                model.update_teacher()
            else:
                skipped += 1

            row = {"step": step, "epoch": epoch,
                   "stepped": int(stepped), "grad_norm": round(gnorm, 3)
                   if gnorm == gnorm else "nan",
                   "lr": opt.param_groups[0]["lr"],
                   "sec": round(time.time() - t0, 1)}
            row.update({k: round(float(v.detach()), 5)
                        for k, v in parts.items()})
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            if step % 10 == 0 or step < 20:
                print("step %4d  loss %8.4f  cls(l/lc/p) %.3f/%.3f/%.3f  "
                      "reg(l/lc/p) %.3f/%.3f/%.3f  align %.4f  gnorm %s"
                      % (step, float(parts["loss"]),
                         float(parts["loss_local_cls"]),
                         float(parts["loss_lc_cls"]),
                         float(parts.get("loss_pac_cls", float("nan"))),
                         float(parts["loss_local_reg"]),
                         float(parts["loss_lc_reg"]),
                         float(parts.get("loss_pac_reg", float("nan"))),
                         float(parts.get("loss_align", 0.0)),
                         row["grad_norm"]), flush=True)
            step += 1
            if args.ckpt_every and step % args.ckpt_every == 0:
                torch.save({"step": step, "model": model.state_dict(),
                            "opt": opt.state_dict()},
                           os.path.join(args.out, "ckpt_step%d.pt" % step))
            if args.max_steps and step >= args.max_steps:
                break
        sched.step()
        if args.max_steps and step >= args.max_steps:
            break

    torch.save({"step": step, "model": model.state_dict()},
               os.path.join(args.out, "last.pt"))
    print("done: %d steps, %d skipped (non-finite grad), %.0fs"
          % (step, skipped, time.time() - t0))
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
