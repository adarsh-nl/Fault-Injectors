"""The one-shot validation gate (spec §6).

Synthetic multi-agent batch -> full train-mode forward (teacher on) -> the
five-term loss -> backward under torch.autograd.set_detect_anomaly(True).
Checks, in order, the failure classes that each cost a debugging round in the
previous implementation:

    1. static source checks (no hard clamps / asin on grad paths)
    2. construction asserts fired (focal bias, dt init) -- implicit in build
    3. forward: finite outputs, documented shapes
    4. init loss components in sane ranges (cls ~1, not ~50)
    5. backward: finite gradient for EVERY parameter
    6. PAC prior preservation probe (init cls logits still near -4.6)
    7. one guarded optimizer step executes (and would skip on non-finite)

    .venv-hpc/bin/python -m corabench.validate
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from cpbench.data import GridSpec
from cpbench.data.synthetic import SyntheticCooperativeDataset
from cpbench.logbook.env import seed_everything

from .data import CoRADataset
from .models import CoRAModel
from .selfcheck import FOCAL_BIAS, grad_step_guard, static_source_checks
from .training import CoRALoss

PASS, FAIL = "ok  ", "FAIL"
failures = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print("%s %s%s" % (PASS if cond else FAIL, name,
                       ("  -- " + extra) if extra else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    seed_everything(2026, deterministic=False)   # CPU validation: fast path
    torch.set_num_threads(4)

    print("== 1. static source checks ==")
    static_source_checks()
    print("ok   no hard clamps / asin on differentiable paths")

    # small grid for a CPU-speed pass; same code path as the full grid
    grid = GridSpec((0.4, 0.4), (-19.2, -19.2, -3.0, 19.2, 19.2, 1.0))
    ds = CoRADataset(
        SyntheticCooperativeDataset(n_frames=2, n_agents=3, n_objects=4),
        grid, reg_dim=8)
    batch = CoRADataset.collate([ds[0], ds[1]])

    print("== 2/3. build + forward (train mode, teacher on) ==")
    model = CoRAModel(grid.grid_hw, channels=64, reg_dim=8,
                      checkpoint_chunks=False)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print("built: %.2fM params, grid %s, feature %s"
          % (n_params / 1e6, grid.grid_hw, grid.feature_hw))

    dt0 = F.softplus(model.lc.cssm.scan.dt_bias.detach())
    check("delta init in [1e-3, 1e-1]",
          bool((dt0 >= 1e-3 * 0.999).all() and (dt0 <= 1e-1 * 1.001).all()),
          "min %.2e max %.2e" % (dt0.min(), dt0.max()))

    with torch.autograd.set_detect_anomaly(True):
        out = model(batch)
        h, w = grid.feature_hw
        na = sum(batch["agent_counts"])
        bsz = len(batch["agent_counts"])
        shape_expect = {
            "local_cls": (na, 2, h, w), "local_reg": (na, 16, h, w),
            "cls_lc": (bsz, 2, h, w), "reg_lc": (bsz, 16, h, w),
            "cls_pac": (bsz, 2, h, w), "reg_pac": (bsz, 16, h, w),
            "cls_lc_recal": (bsz, 2, h, w), "cls_pac_recal": (bsz, 2, h, w),
            "u_lc": (bsz, 2, h, w), "u_pac": (bsz, 2, h, w),
        }
        for k, s in shape_expect.items():
            check("shape %s == %s" % (k, s), tuple(out[k].shape) == s,
                  str(tuple(out[k].shape)))
            check("finite %s" % k, bool(torch.isfinite(out[k]).all()))

        # 6. prior preservation: every cls path near the focal prior at init
        for k in ("local_cls", "cls_lc", "cls_pac", "cls_lc_recal",
                  "cls_pac_recal"):
            m = float(out[k].mean())
            check("init prior %s in [-6, -3]" % k, -6.0 < m < -3.0,
                  "mean %.3f (focal prior %.3f)" % (m, FOCAL_BIAS))

        print("== 4. loss components at init ==")
        loss_fn = CoRALoss(reg_dim=8)
        parts = loss_fn(out, batch)
        for k in sorted(parts):
            print("   %-16s %.4f" % (k, float(parts[k])))
        for k in ("loss_local_cls", "loss_lc_cls", "loss_pac_cls"):
            v = float(parts[k])
            check("%s ~ O(1) (in [0.05, 5], not ~50)" % k, 0.05 < v < 5.0,
                  "%.3f" % v)
        for k in ("loss_local_reg", "loss_lc_reg", "loss_pac_reg"):
            v = float(parts[k])
            check("%s sane (in [1e-4, 20])" % k, 1e-4 < v < 20.0, "%.3f" % v)
        check("loss_align finite and small at init",
              0.0 <= float(parts.get("loss_align", 0.0)) < 10.0,
              "%.5f" % float(parts.get("loss_align", 0.0)))
        check("total finite", bool(torch.isfinite(parts["loss"])),
              "%.4f" % float(parts["loss"]))

        print("== 5. backward ==")
        parts["loss"].backward()

    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    none_grad = [n for n, p in model.named_parameters() if p.grad is None]
    check("every gradient finite (0 NaN/inf tensors)", not bad,
          "; ".join(bad[:5]))
    # parameters legitimately without grad this pass: none expected --
    # discrete CIT masks pass gradient through features, teacher is not
    # registered. Anything else is a wiring hole.
    check("every parameter received a gradient", not none_grad,
          "; ".join(none_grad[:8]))
    gmax = max(float(p.grad.abs().max()) for p in model.parameters()
               if p.grad is not None)
    print("   grad |max| over all params: %.3e" % gmax)

    print("== 7. guarded optimizer step ==")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stepped, gnorm = grad_step_guard(model, opt, max_norm=10.0)
    check("guarded step executed on finite grads", stepped,
          "norm %.3f" % gnorm)
    model.update_teacher()
    print("   teacher EMA update ran")

    print("\n%d failures" % len(failures))
    if failures:
        print("FAILED:", failures)
        return 1
    print("CORA VALIDATION PASS -- ready to train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
