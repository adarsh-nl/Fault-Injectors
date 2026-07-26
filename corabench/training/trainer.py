"""
trainer.py
----------
Trainer: AMP training loop with checkpointing, resume and full logging.

Paper settings (configs/trainer/default.yaml): Adam lr 1e-3, 30 epochs,
batch size 2, single GPU. Fault-aware training (pose noise on the training
bridge) is configured on the DATASET's DataFaultBridge, not here -- the
trainer never touches corruption.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..data.cooperative import collate_cooperative
from cpbench.logbook.experiment import ExperimentLogger
from cpbench.logbook.schema import TrainRecord
from .losses import CoRALoss
from .validator import Validator, _to_device

logger = logging.getLogger(__name__)


class Trainer:
    """Train a CoRAModel.

    Parameters
    ----------
    model        CoRAModel (already on `device`).
    train_set    CoRADataset (its bridge may carry training noise).
    val_set      CoRADataset with a CLEAN bridge, or None.
    loss_fn      CoRALoss.
    exp_logger   ExperimentLogger (checkpoints + records).
    cfg          trainer config dict: epochs, batch_size, optimizer{lr,
                 weight_decay}, scheduler{milestones, gamma}, amp,
                 grad_clip, val_every, num_workers, log_every.

    Example
    -------
    >>> Trainer(model, train_set, val_set, CoRALoss(), explog,
    ...         {"epochs": 30, "batch_size": 2}).fit()        # doctest: +SKIP
    """

    def __init__(self, model, train_set, val_set, loss_fn: CoRALoss,
                 exp_logger: ExperimentLogger, cfg: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 taps: Optional[Any] = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.explog = exp_logger
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loader = DataLoader(
            train_set, batch_size=int(cfg.get("batch_size", 2)), shuffle=True,
            num_workers=int(cfg.get("num_workers", 0)),
            collate_fn=collate_cooperative, drop_last=True)
        self.validator = Validator(
            val_set, self.device, batch_size=int(cfg.get("batch_size", 2)),
            num_workers=int(cfg.get("num_workers", 0))) if val_set else None

        opt_cfg = cfg.get("optimizer", {})
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=float(opt_cfg.get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-4)))
        sch_cfg = cfg.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=sch_cfg.get("milestones", [15, 25]),
            gamma=float(sch_cfg.get("gamma", 0.1)))
        self.amp = bool(cfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.grad_clip = float(cfg.get("grad_clip", 10.0))
        self.epochs = int(cfg.get("epochs", 30))
        self.val_every = int(cfg.get("val_every", 1))
        self.log_every = int(cfg.get("log_every", 50))
        self.start_epoch = 0
        self.best_metric = -1.0
        self._nonfinite_steps = 0
        self.taps = taps
        self._scale_floor = float(cfg.get("scale_floor", 1e-8))
        self._logit_warn_threshold = float(cfg.get("logit_warn_threshold", 20.0))
        self._saturating_steps = 0
        self._initial_scale = float(self.scaler.get_scale()) if self.amp else 1.0

    # -- checkpointing ------------------------------------------------------

    def _optimizer_state_amax(self) -> float:
        """max|exp_avg_sq| over Adam's state; inf/nan once the moments rot.

        Adam's moments are the one piece of training state GradScaler does not
        protect: ``found_inf`` guards the CURRENT step's gradients, but a
        moment that has already gone non-finite makes every subsequent step --
        including ones the scaler considers healthy -- write NaN into the
        parameters. Job 546515 finished with 268/402 non-finite Adam tensors
        by a route that could not be reconstructed from the code, so it is
        measured rather than assumed.
        """
        worst = 0.0
        for state in self.optimizer.state.values():
            v = state.get("exp_avg_sq")
            if torch.is_tensor(v):
                m = float(v.abs().max())
                if not (m <= worst):        # False for NaN, so NaN propagates
                    worst = m
        return worst

    def _grad_norm_by_module(self) -> str:
        """Per-top-level-submodule gradient norms as "name=norm;name=norm".

        The global norm says how big the gradient is, never which part of the
        model owns it. In job 547612 the only three steps that ever landed had
        pre-clip norms of 1006 / 317 / 204, and lc/output grew 142x across
        them while encoder/bev_features stayed flat -- but with one global
        scalar there is no way to tell whether the LC branch originated that
        or merely transmitted it.

        Format is ';'-separated and comma-free so it survives a CSV cell.
        """
        totals: Dict[str, float] = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            top = name.split(".")[0]
            totals[top] = totals.get(top, 0.0) + float(
                param.grad.detach().pow(2).sum())
        return ";".join(f"{k}={v ** 0.5:.6g}"
                        for k, v in sorted(totals.items(),
                                           key=lambda kv: -kv[1]))

    def _warn_if_head_saturating(self, out: Dict[str, Any], epoch: int,
                                 i: int) -> float:
        """max|cls logits| across branches; warns past the saturation knee.

        A focal loss made finite by a float32 island no longer produces a NaN
        when the head saturates, so the trainer's non-finite guard goes quiet
        and the run COMPLETES while being just as broken. In half precision
        sigmoid saturates to exactly 1.0 above a logit of about 8.3; job
        547612 reached 205.5. Warning at 20 puts the alarm well past normal
        variation and well below the point where the loss stops carrying
        information, so a saturating run announces itself in the log instead
        of waiting to be found in taps.csv.
        """
        worst = 0.0
        for branch in ("local", "lc", "pac"):
            block = out.get(branch)
            if isinstance(block, dict) and torch.is_tensor(block.get("cls")):
                worst = max(worst, float(block["cls"].detach().abs().max()))
        if worst > self._logit_warn_threshold:
            self._saturating_steps += 1
            logger.warning(
                "epoch %d batch %d: max|cls logit| = %.4g exceeds %.4g. In "
                "fp16 sigmoid saturates to exactly 1.0 above ~8.3, so the "
                "classification loss is losing its gradient signal. The loss "
                "is FINITE (float32 island) and will not trip the non-finite "
                "guard -- completion is not evidence of health. %d saturating "
                "steps so far; check taps.csv lc/output and head/cls_logits.",
                epoch, i, worst, self._logit_warn_threshold,
                self._saturating_steps)
        return worst

    def _assert_finite_state(self, epoch: int, tag: str) -> bool:
        """True if every model tensor is finite; logs the offenders if not."""
        bad = [k for k, v in self.model.state_dict().items()
               if torch.is_floating_point(v) and not torch.isfinite(v).all()]
        if bad:
            total = sum(1 for v in self.model.state_dict().values()
                        if torch.is_floating_point(v))
            logger.error(
                "epoch %d: REFUSING to write checkpoint %r -- %d/%d float "
                "tensors are non-finite (first: %s). A checkpoint that loads "
                "cleanly and is entirely NaN is worse than no checkpoint: it "
                "was written as 'best' in job 546515 and looked valid. Fix "
                "the run, do not resume from this state.",
                epoch, tag, len(bad), total, ", ".join(bad[:3]))
        return not bad

    def save_checkpoint(self, epoch: int, tag: str) -> Path:
        path = self.explog.checkpoints_dir / f"{tag}.pt"
        if not self._assert_finite_state(epoch, tag):
            return path
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
        }, path)
        return path

    def resume(self, path: "str | Path") -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.best_metric = float(ckpt.get("best_metric", -1.0))
        logger.info("resumed from %s at epoch %d", path, self.start_epoch)

    # -- training -----------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        for i, batch in enumerate(self.loader):
            t0 = time.perf_counter()
            batch = _to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type,
                                enabled=self.amp):
                out = self.model(batch, taps=self.taps, return_teacher=True)
                losses = self.loss_fn(out, batch)
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            # Measured after unscale_ and BEFORE the clip, so it is the real
            # gradient rather than a uniformly rescaled one. Only on logging
            # steps: each entry costs a reduction and a device sync.
            logging_step = (i % self.log_every == 0)
            by_module = self._grad_norm_by_module() if logging_step else ""
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip)

            # clip_grad_norm_ defaults to error_if_nonfinite=False: ONE NaN
            # gradient makes total_norm NaN, so clip_coef = max_norm/(NaN+eps)
            # is NaN and every gradient -- including every finite one -- is
            # multiplied by it. GradScaler.step() then skips on found_inf, but
            # found_inf was computed by unscale_ BEFORE the clip, and it does
            # not protect Adam's already-accumulated moments: a single step
            # that gets through poisons exp_avg/exp_avg_sq permanently, and
            # from then on every step the scaler considers healthy still
            # writes NaN into the parameters. Job 546515 ended with 226/227
            # non-finite tensors and 268/402 non-finite Adam states this way.
            #
            # Skipping here turns silent total corruption into a logged skip.
            # update() is still called so the scale backs off exactly as it
            # would have; only the optimizer step is withheld.
            if not bool(torch.isfinite(grad_norm)):
                self._nonfinite_steps += 1
                logger.warning(
                    "epoch %d batch %d: non-finite grad_norm (%s) -- skipping "
                    "optimizer step (scaler scale %.4g, %d skipped so far). "
                    "Gradients zeroed; Adam state left untouched.",
                    epoch, i, grad_norm, self.scaler.get_scale(),
                    self._nonfinite_steps)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            running += float(losses["total"].item())

            # Scale-floor abort. A GradScaler that has backed off this far has
            # not been fixed by backing off, so the non-finite quantity is not
            # fp16 gradient overflow -- it does not depend on the loss scale,
            # and no further step can recover. Job 546515 ran to the end of its
            # epoch with scale exactly 0.0, learning nothing and writing a
            # 226/227-NaN checkpoint. Stopping here stops at the informative
            # moment instead of burning the wall clock.
            scale = self.scaler.get_scale()
            if scale < self._scale_floor:
                raise RuntimeError(
                    f"GradScaler scale collapsed to {scale:.4g} (floor "
                    f"{self._scale_floor:.4g}) at epoch {epoch} batch {i} "
                    f"after {self._nonfinite_steps} non-finite steps. The "
                    f"scale has halved from {self._initial_scale:.4g} without "
                    f"the gradients becoming finite, so the source is "
                    f"scale-independent: a non-finite forward activation or "
                    f"loss, not fp16 gradient overflow. Last losses: "
                    f"total={float(losses['total'].item()):.4g} "
                    f"cls={float(losses['cls'].item()):.4g} "
                    f"reg={float(losses['reg'].item()):.4g} "
                    f"align={float(losses['align'].item()):.4g}. Inspect "
                    f"taps.csv (n_nan/n_inf per location) for the first "
                    f"location that goes non-finite.")

            self.explog.log_fault_records(batch["fault_records"])
            if logging_step:
                mem = torch.cuda.max_memory_allocated() / 2 ** 20 \
                    if self.device.type == "cuda" else 0.0
                rec = TrainRecord(
                    epoch=epoch, batch=i,
                    loss_total=float(losses["total"].item()),
                    loss_cls=float(losses["cls"].item()),
                    loss_reg=float(losses["reg"].item()),
                    loss_align=float(losses["align"].item()),
                    loss_pac=float(losses["pac"].item()),
                    lr=self.optimizer.param_groups[0]["lr"],
                    grad_norm=float(grad_norm),
                    scaler_scale=float(self.scaler.get_scale()),
                    n_skipped_steps=float(self._nonfinite_steps),
                    opt_state_amax=self._optimizer_state_amax(),
                    grad_norm_by_module=by_module,
                    head_logit_amax=self._warn_if_head_saturating(out, epoch, i),
                    batch_time_s=time.perf_counter() - t0,
                    gpu_mem_mb=mem)
                self.explog.log_train(rec)
                logger.info(
                    "epoch %d batch %d loss %.4g (cls %.4g reg %.4g "
                    "align %.4g pac %.4g) | grad_norm %.4g scale %.4g "
                    "skipped %d opt_amax %.4g logit_amax %.4g | %s",
                    epoch, i, rec.loss_total, rec.loss_cls, rec.loss_reg,
                    rec.loss_align, rec.loss_pac, rec.grad_norm,
                    rec.scaler_scale, int(rec.n_skipped_steps),
                    rec.opt_state_amax, rec.head_logit_amax,
                    rec.grad_norm_by_module)
        return running / max(len(self.loader), 1)

    def fit(self) -> Dict[str, float]:
        """Full training run; returns the last validation metrics."""
        t_start = time.perf_counter()
        last_val: Dict[str, float] = {}
        for epoch in range(self.start_epoch, self.epochs):
            avg = self.train_epoch(epoch)
            self.scheduler.step()
            logger.info("epoch %d done, mean loss %.4f", epoch, avg)
            self.save_checkpoint(epoch, "last")
            if self.validator and (epoch + 1) % self.val_every == 0:
                last_val = self.validator.run(self.model)
                from cpbench.logbook.schema import EvalRecord
                self.explog.log_eval(EvalRecord(
                    epoch=epoch, dataset=type(self.model).__name__,
                    split="val", condition={"fault": "clean"},
                    detection=last_val,
                    n_frames=int(last_val.get("n_frames", 0))))
                metric = last_val.get("ap70", 0.0)
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.save_checkpoint(epoch, "best")
                    logger.info("new best ap70=%.4f at epoch %d", metric, epoch)
            self.explog.flush()
        total_s = time.perf_counter() - t_start
        self.explog.scalar("train/total_time_s", total_s, self.epochs)
        logger.info("training finished in %.1f s", total_s)
        return last_val
