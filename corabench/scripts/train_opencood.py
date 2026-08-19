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
import random
import time

import numpy as np
import torch

from ..compat import autocast, grad_scaler, load as torch_load
from ..data.opencood_adapter import CoRABatchAdapter
from ..models import CoRAModel
from ..selfcheck import grad_step_guard, static_source_checks
from ..runtime_gate import GateAbort, RuntimeGate
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


RESUME_CONTRACT_VERSION = 1


def rng_state():
    """Every RNG stream that can affect a step."""
    s = {"torch": torch.get_rng_state(),
         "numpy": np.random.get_state(),
         "python": random.getstate()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def load_rng(s):
    """Returns a list of streams that could NOT be restored, so a partial
    restore is reported rather than assumed."""
    missing = []
    if "torch" in s:
        torch.set_rng_state(s["torch"].cpu() if hasattr(s["torch"], "cpu")
                            else s["torch"])
    else:
        missing.append("torch")
    if "numpy" in s:
        np.random.set_state(s["numpy"])
    else:
        missing.append("numpy")
    if "python" in s:
        random.setstate(s["python"])
    else:
        missing.append("python")
    if torch.cuda.is_available():
        if "cuda" in s:
            torch.cuda.set_rng_state_all([x.cpu() if hasattr(x, "cpu") else x
                                          for x in s["cuda"]])
        else:
            missing.append("cuda")
    return missing


def save_ckpt(path, *, step, epoch, batch_in_epoch, model, opt, sched, scaler,
              args):
    """Everything a correct resume needs.

    The pre-2026-08-10 checkpoints held only {step, model, opt}. Resuming from
    those cannot restore the scheduler, the GradScaler or any RNG stream --
    and the SCHEDULER is a correctness requirement, not a nicety:
    MultiStepLR(opt, [15, 25], gamma=0.1) decays at epochs 15 and 25, both
    inside the resume window, so a restarted scheduler would hold lr=1e-3
    through epochs that should be at 1e-4 while the log still says 30 epochs.
    """
    torch.save({"resume_contract_version": RESUME_CONTRACT_VERSION,
                "step": step, "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng_state(),
                "seed": args.seed, "batch_size": args.batch_size}, path)


def epoch_loader(adapter, args, epoch):
    """DataLoader whose shuffle is a pure function of (seed, epoch).

    Saving a sampler's internal position is impractical across worker
    processes; making the permutation reproducible from (seed, epoch) is
    equivalent for resume and far more robust -- the order is recomputable
    rather than restored, so it cannot drift.
    """
    g = torch.Generator()
    g.manual_seed(args.seed * 1000003 + epoch)
    return torch.utils.data.DataLoader(
        adapter, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=adapter.collate,
        drop_last=True, generator=g)


def assert_geometry(grid_hw, head_hw, want_canvas, want_head):
    """Hard-assert the DERIVED geometry, not the YAML that was meant to
    produce it.

    Twice now a geometry mismatch has cost a run. Job 557631 died because the
    encoder strode the canvas by 2 while OpenCOOD anchored at canvas/4. Jobs
    559595/559596 were launched on CoBEVT's config, whose range is +-38.4 and
    whose feature_stride is 4, giving canvas (192,704) / head (48,176) -- the
    paper's OpenCOOD OPV2V basis is (200,704) / (100,352). Both times the YAML
    looked plausible and the geometry it implied was wrong, so the check has
    to be on the derived numbers.

    >>> assert_geometry((200, 704), (100, 352), '200x704', '100x352')
    >>> assert_geometry((192, 704), (48, 176), '200x704', '100x352')
    Traceback (most recent call last):
        ...
    SystemExit: GEOMETRY ASSERTION FAILED: canvas is (192, 704), expected (200, 704); head/anchor grid is (48, 176), expected (100, 352)
    """
    bad = []
    for name, got, want in (("canvas", tuple(grid_hw), want_canvas),
                            ("head/anchor grid", tuple(head_hw), want_head)):
        if not want:
            continue
        exp = tuple(int(x) for x in want.lower().split("x"))
        if got != exp:
            bad.append("%s is %s, expected %s" % (name, got, exp))
    if bad:
        raise SystemExit("GEOMETRY ASSERTION FAILED: " + "; ".join(bad))


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
    ap.add_argument("--dt_bound", type=float, default=0.2)
    ap.add_argument("--gate_interval", type=int, default=100)
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--lambda_align", type=float, default=1.0,
                    help="sec 7.7 arms A/B. NOTE parts['loss_align'] logs the "
                         "RAW term, not the weighted contribution, so the "
                         "term stays observable at lambda=0.")
    ap.add_argument("--ema_momentum", type=float, default=0.999,
                    help="sec 7.7 arm C. tau = 1/(1-m).")
    ap.add_argument("--no_pac", action="store_true",
                    help="sec 7.11 rung 2: disable the PAC branch. NOTE PAC "
                         "consumes the LOCAL branch outputs, not LC's, so "
                         "this does not remove anything downstream of LC.")
    ap.add_argument("--no_teacher", action="store_true",
                    help="sec 7.11 rungs 2 and 3: disable the EMA teacher. "
                         "This ALSO zeroes align -- align_terms is only "
                         "appended when self.training and _teacher_enabled -- "
                         "so --lambda_align is redundant alongside it.")
    ap.add_argument("--assert_canvas", default="",
                    help="HxW the pillar canvas MUST have, e.g. 200x704. "
                         "Abort if not. See assert_geometry().")
    ap.add_argument("--assert_head", default="",
                    help="HxW the head/anchor grid MUST have, e.g. 100x352.")
    ap.add_argument("--grad_probe", action="store_true",
                    help="sec 7.14 backward-side probe: per-step gradient "
                         "norms and ||d_theta||/||theta|| for the selective "
                         "parameter groups, plus activation/gradient norms "
                         "at the z_fused -> y_pre_ln -> y_post_ln "
                         "boundaries and the gate's loss-trend statistic. "
                         "EVERY step, one shared timeline.")
    ap.add_argument("--act_probe", action="store_true",
                    help="record LC/CSSM boundary activation magnitudes: "
                         "every step for the first 50, then every 10")
    ap.add_argument("--init_scale", type=float, default=0.0,
                    help="GradScaler init_scale override (0 = torch default "
                         "65536). The seed diagnostic tests whether early "
                         "backoffs select the basin; a lower init_scale is "
                         "the mitigation that touches no model code.")
    ap.add_argument("--resume", default="",
                    help="checkpoint to resume from; restores model, "
                         "optimiser, scheduler, GradScaler, epoch, RNG and "
                         "sampler position. See save_ckpt().")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    static_source_checks()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter, grid_hw = build(args.hypes, train=True)
    # OpenCOOD's anchor/target grid, probed from a real item -- the canvas
    # and the anchor grid differ by stride 4, and assuming their relation
    # cost the first smoke attempt (job 557631).
    head_hw = tuple(adapter[0]["targets"]["cls_target"].shape[:2])
    print("dataset %d samples | canvas %s | head/anchor grid %s | device %s"
          % (len(adapter), grid_hw, head_hw, device), flush=True)
    # BEFORE the model is built and before a single step is taken.
    assert_geometry(grid_hw, head_hw, args.assert_canvas, args.assert_head)
    if args.assert_canvas or args.assert_head:
        print("geometry assertion PASSED (canvas=%s head=%s)"
              % (grid_hw, head_hw), flush=True)

    model = CoRAModel(grid_hw, channels=args.channels, reg_dim=7,
                      head_hw=head_hw, dt_bound=args.dt_bound,
                      fp32_island=True,
                      ema_momentum=args.ema_momentum,
                      pac_enabled=not args.no_pac,
                      teacher_enabled=not args.no_teacher).to(device)
    gate = None if args.no_gate else RuntimeGate(
        dt_bound=args.dt_bound, interval=args.gate_interval)
    if args.act_probe:
        model.lc.record_acts = True
    # sec 7.14 probe groups. Named separately on purpose: the precedence
    # prediction is about WHICH group moves first, so pooling them into one
    # norm would erase exactly the information the experiment needs.
    probe_groups = None
    if args.grad_probe:
        model.lc.cssm.record_flow = True
        c_ = model.lc.cssm
        probe_groups = [
            ("a_log", [c_.scan.a_log]),
            ("dt_bias", [c_.scan.dt_bias]),
            ("dt_proj", list(c_.dt_proj.parameters())),
            ("b_proj", list(c_.b_proj.parameters())),
            ("c_proj", list(c_.c_proj.parameters())),
            ("out_norm", list(c_.out_norm.parameters())),
            # LC's output projection (GatingUnit.out) -- the last trainable
            # map before F_out, i.e. the downstream comparator for "did the
            # selective params move FIRST".
            ("gate_out", list(model.lc.gate.out.parameters())),
        ]
        print("[grad_probe] ON: groups %s + flow taps z_fused/y_pre_ln/"
              "y_post_ln" % [g for g, _ in probe_groups], flush=True)
    loss_fn = CoRALoss(reg_dim=7,
                       lambda_align=args.lambda_align).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [15, 25], gamma=0.1)
    amp = args.amp and device == "cuda"
    scaler = grad_scaler(enabled=amp, init_scale=args.init_scale)
    if args.init_scale and amp:
        print("GradScaler init_scale -> %g" % args.init_scale, flush=True)

    log_path = os.path.join(args.out, "smoke_loss.csv" if args.max_steps
                            else "train_loss.csv")
    fh = open(log_path, "w", newline="")
    writer = None

    step = 0
    skipped = 0
    start_epoch, skip_batches = 0, 0
    if args.resume:
        ck = torch_load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        step = int(ck.get("step", 0))
        missing = []
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            missing.append("optimiser")
        ver = ck.get("resume_contract_version")
        if ver is None:
            # PRE-FIX CHECKPOINT ({step, model, opt} only). The scheduler IS
            # reconstructable: sched.step() runs once per epoch and
            # steps_per_epoch is deterministic (len(adapter)//batch_size with
            # drop_last), so epoch = step // steps_per_epoch exactly. Nothing
            # else is: scaler, RNG and sampler position are gone.
            spe = len(adapter) // args.batch_size
            start_epoch = step // spe
            skip_batches = step % spe
            for _ in range(start_epoch):
                sched.step()
            missing += ["GradScaler state", "RNG (torch/cuda/numpy/python)",
                        "exact sampler order for the resumed epoch"]
            print("[resume] PRE-FIX checkpoint (no resume_contract_version). "
                  "step=%d -> epoch=%d reconstructed from step count "
                  "(steps/epoch=%d); scheduler advanced %d epochs."
                  % (step, start_epoch, spe, start_epoch), flush=True)
        else:
            start_epoch = int(ck["epoch"])
            skip_batches = int(ck.get("batch_in_epoch", 0))
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            missing += load_rng(ck.get("rng", {}))
            if int(ck.get("seed", args.seed)) != args.seed:
                print("[resume] WARNING seed differs: ckpt=%s cli=%s -- the "
                      "sampler permutation will NOT match"
                      % (ck.get("seed"), args.seed), flush=True)
        print("[resume] from %s | step=%d epoch=%d skip_batches=%d lr=%.3g "
              "scaler_scale=%s"
              % (args.resume, step, start_epoch, skip_batches,
                 opt.param_groups[0]["lr"],
                 (scaler.get_scale() if amp else "n/a")), flush=True)
        if missing:
            print("[resume] NOT RESTORED: %s" % ", ".join(missing), flush=True)

    t0 = time.time()
    # feature-map/target-shape agreement asserted on the first batch below.
    aborted = None
    try:
      for epoch in range(start_epoch, args.epochs):
        model.train()
        loader = epoch_loader(adapter, args, epoch)
        n_skip = skip_batches if epoch == start_epoch else 0
        for bi, batch in enumerate(loader):
            if bi < n_skip:                 # already consumed pre-checkpoint
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if args.act_probe:
                probed = (step < 50) or (step % 10 == 0)
                model.lc.record_acts = probed
            with autocast(device, enabled=amp):
                out = model(batch)
                if step == 0:
                    assert out["cls_lc"].shape[-2:] == \
                        batch["cls_target"].shape[1:3], \
                        "feature map does not match the hypes' target grid"
                parts = loss_fn(out, batch)
            scaler.scale(parts["loss"]).backward()
            scaler.unscale_(opt)
            # ── sec 7.14 probe, part 1: RAW grad norms (post-unscale,
            # PRE-clip -- grad_step_guard clips at 10.0 and the probe must
            # see what the optimizer pressure is, not its clipped shadow),
            # and a parameter snapshot for the post-step ||dtheta||/||theta||.
            probe_row = {}
            if probe_groups is not None:
                cur_scale = float(scaler.get_scale()) if amp else 1.0
                snap = {}
                for gname, plist in probe_groups:
                    gsq, wsq = 0.0, 0.0
                    for p in plist:
                        if p.grad is not None:
                            gsq += float(p.grad.detach().float().norm()) ** 2
                        wsq += float(p.detach().float().norm()) ** 2
                    probe_row["g_" + gname] = gsq ** 0.5
                    snap[gname] = ([p.detach().clone() for p in plist],
                                   wsq ** 0.5)
            stepped, gnorm = grad_step_guard(model, opt, max_norm=10.0)
            scaler.update()
            # ── sec 7.14 probe, part 2: realized parameter step per group
            # (0 on skipped steps -- the `stepped` column disambiguates),
            # flow-boundary norms (grad norms rescaled from the
            # GradScaler-scaled backward), and the gate trend statistic.
            if probe_groups is not None:
                for gname, plist in probe_groups:
                    olds, wnorm = snap[gname]
                    dsq = sum(float((p.detach() - o).float().norm()) ** 2
                              for p, o in zip(plist, olds))
                    probe_row["r_" + gname] = (dsq ** 0.5) / (wnorm + 1e-12)
                for k, v in dict(model.lc.cssm.last_flow).items():
                    probe_row[k] = (v / cur_scale if k.endswith("_gradnorm")
                                    else v)
            if stepped:
                model.update_teacher()
            else:
                skipped += 1

            fin = bool(torch.isfinite(parts["loss"]).item())
            if gate is not None:
                gate.observe_step(
                    fin, float(parts['loss'].detach()) if fin else None,
                    scale=(scaler.get_scale() if amp else None),
                    grad_norm=gnorm, stepped=stepped)
                g = gate.check(step, model)
                if g is not None:
                    print("[gate] " + "  ".join(
                        "%s=%s" % (k, (round(v, 5) if isinstance(v, float) else v))
                        for k, v in g.items()), flush=True)
            row = {"step": step, "epoch": epoch,
                   "stepped": int(stepped), "grad_norm": round(gnorm, 3)
                   if gnorm == gnorm else "nan",
                   # scale is read AFTER scaler.update(), i.e. it is the value
                   # that will be used for the NEXT step; a backoff at step k
                   # therefore shows as a halving between row k-1 and row k.
                   "scale": (scaler.get_scale() if amp else 0.0),
                   "seed": args.seed,
                   "lr": opt.param_groups[0]["lr"],
                   "sec": round(time.time() - t0, 1)}
            row.update({k: round(float(v.detach()), 5)
                        for k, v in parts.items()})
            # PERMANENT INSTRUMENT PANEL (job 567755 post-mortem). These six
            # go into EVERY run, not just --grad_probe runs: 567755 stepped
            # 1,600 times on zero gradients and it was invisible because
            # nothing logged fraction_nonzero_grad or median_grad_norm.
            # 'scale' is DELIBERATELY skipped: the row already carries it,
            # read AFTER scaler.update() (i.e. the value the NEXT step uses,
            # so a backoff shows as a halving between rows k-1 and k). The
            # panel's copy is the PRE-update value. Overwriting would
            # silently redefine an existing column and break comparability
            # with every CSV already on disk.
            if gate is not None:
                for k, v in gate.panel().items():
                    if k == 'scale':
                        continue
                    row[k] = (round(float(v), 8) if v == v else '')
            sc = getattr(getattr(model.lc, "cssm", None), "scan", None)
            for k, v in (getattr(sc, "last_delta_stats", None) or {}).items():
                row[k] = round(float(v), 6)
            for k, v in (getattr(model, "last_teacher_stats", None) or {}).items():
                row["t_" + k] = round(float(v), 4)
            if args.act_probe:
                # last_act_stats PERSISTS between probes, so writing it
                # unconditionally would emit stale numbers as if they were
                # fresh. Keys stay in the header (step 0 is always probed, so
                # the DictWriter schema is fixed); values are blank when this
                # step did not measure.
                for k, v in (getattr(model.lc, "last_act_stats", None)
                             or {}).items():
                    row[k] = round(float(v), 6) if probed else ""
            if probe_groups is not None:
                # trend is read AFTER gate.observe_step above, so this row's
                # loss_trend includes this row's loss -- one shared timeline.
                probe_row["loss_trend"] = (gate.trend_ratio()
                                           if gate is not None
                                           else float("nan"))
                row.update({k: (round(v, 8) if isinstance(v, float)
                                and v == v else v)
                            for k, v in probe_row.items()})
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
                save_ckpt(os.path.join(args.out, "ckpt_step%d.pt" % step),
                          step=step, epoch=epoch, batch_in_epoch=bi + 1,
                          model=model, opt=opt, sched=sched, scaler=scaler,
                          args=args)
            if args.max_steps and step >= args.max_steps:
                break
        sched.step()
        if args.max_steps and step >= args.max_steps:
            break
    except GateAbort as exc:
        # An abort is a RESULT, not a crash: save state and the gate history
        # so the trip is diagnosable without re-running.
        aborted = str(exc)
        print("\n[GATE ABORT] %s" % aborted, flush=True)
        save_ckpt(os.path.join(args.out, "abort_step%d.pt" % step),
                  step=step, epoch=epoch, batch_in_epoch=bi + 1, model=model,
                  opt=opt, sched=sched, scaler=scaler, args=args)

    save_ckpt(os.path.join(args.out, "last.pt"), step=step,
              epoch=locals().get("epoch", start_epoch),
              batch_in_epoch=locals().get("bi", -1) + 1, model=model, opt=opt,
              sched=sched, scaler=scaler, args=args)
    if gate is not None:
        import json
        with open(os.path.join(args.out, "gate_history.json"), "w") as gh:
            json.dump({"aborted": aborted, "history": gate.history}, gh, indent=1)
    print("done: %d steps, %d skipped (non-finite grad), %.0fs"
          % (step, skipped, time.time() - t0))
    fh.close()
    # A gate abort must exit NON-ZERO: --dependency=afterok chains
    # off this code, and an abort that reported success would let a
    # dependent full run start on a model the gate just rejected.
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
