"""
trainer.py
----------
The training loop: AMP, gradient clipping, checkpointing, resume, logging.

Track-agnostic by construction. The trainer never names CoBEVTCamera or
CoBEVTLidar -- it is handed a model, a loss and a function that extracts the
loss arguments from a batch. Both tracks then share one loop, which is what
keeps their optimiser schedule, their gradient clipping and their checkpoint
format identical and therefore their results comparable.

Training is never tapped
------------------------
``taps`` is not threaded into the training forward pass. Observation costs a
detach and a stats pass per emit site, and there are ~200 of them per step;
paying that on every training batch to record numbers nobody reads would be
a straight waste. Taps belong in evaluation, where the tensors are the
product rather than a by-product.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from cpbench.logbook import ExperimentLogger, TrainRecord

logger = logging.getLogger(__name__)

# A batch and the model's output -> the loss dict.
LossFn = Callable[[Dict[str, Any], Dict[str, torch.Tensor]],
                  Dict[str, torch.Tensor]]


class Trainer:
    """Train either CoBEVT track.

    Purpose
        One loop, one checkpoint format, one set of logged columns for both
        tracks.

    Inputs
    ------
    model         CoBEVTCamera or CoBEVTLidar
    loss_fn       ``(batch, output) -> {"loss": ..., ...}``. Extra keys are
                  logged verbatim, so a track can report its own components.
    optimizer     any torch optimizer
    device        where to run
    scheduler     stepped per epoch, after validation
    logger_       ExperimentLogger; None disables persistence (tests)
    amp           mixed precision. On by default for CUDA and forced off on
                  CPU, where autocast is slower rather than faster.
    grad_clip     max gradient norm; 0 disables
    log_every     batches between TrainRecord rows

    Example
    -------
    >>> # see tests/test_train_smoke.py for a runnable end-to-end example
    """

    def __init__(self, model: nn.Module, loss_fn: LossFn,
                 optimizer: torch.optim.Optimizer,
                 device: Optional[torch.device] = None,
                 scheduler: Optional[Any] = None,
                 logger_: Optional[ExperimentLogger] = None,
                 amp: Optional[bool] = None, grad_clip: float = 35.0,
                 log_every: int = 10) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logbook = logger_
        self.grad_clip = float(grad_clip)
        self.log_every = int(log_every)
        self.amp = (self.device.type == "cuda") if amp is None else bool(amp)
        if self.amp and self.device.type != "cuda":
            logger.warning("AMP requested on %s; disabling (autocast is not a "
                           "speedup off CUDA)", self.device.type)
            self.amp = False
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.epoch = 0
        self.best_metric = float("-inf")

    # -- batch plumbing -----------------------------------------------------

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device, non_blocking=True)
                    if torch.is_tensor(v) else v) for k, v in batch.items()}

    # -- one epoch ----------------------------------------------------------

    def train_epoch(self, loader, max_batches: Optional[int] = None
                    ) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        n_batches = 0

        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            started = time.perf_counter()
            batch = self._to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                output = self.model(batch)
                losses = self.loss_fn(batch, output)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                # Skipping beats propagating: one bad batch should not
                # poison every parameter through an inf gradient, and a
                # silently-NaN model is very expensive to diagnose later.
                logger.warning("non-finite loss at epoch %d batch %d; "
                               "skipping the step", self.epoch, index)
                continue

            self.scaler.scale(loss).backward()
            grad_norm = 0.0
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = float(nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1

            if self.logbook is not None and index % self.log_every == 0:
                self.logbook.log_train(TrainRecord(
                    epoch=self.epoch, batch=index,
                    loss_total=float(loss),
                    loss_cls=float(losses.get("loss_cls",
                                              losses.get("loss_seg", 0.0))),
                    loss_reg=float(losses.get("loss_reg", 0.0)),
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    grad_norm=grad_norm,
                    batch_time_s=time.perf_counter() - started,
                    gpu_mem_mb=self._gpu_mb()))

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _gpu_mb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    # -- full run -----------------------------------------------------------

    def fit(self, train_loader, epochs: int, validator=None,
            max_batches: Optional[int] = None) -> Dict[str, float]:
        """Train for ``epochs``, validating and checkpointing each epoch."""
        summary: Dict[str, float] = {}
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            summary = self.train_epoch(train_loader, max_batches)
            logger.info("epoch %d: %s", epoch,
                        ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))

            if validator is not None:
                metrics = validator.run(self.model, epoch=epoch)
                score = validator.score(metrics)
                summary["val_score"] = score
                if score > self.best_metric:
                    self.best_metric = score
                    self.save("best.pt")
            if self.scheduler is not None:
                self.scheduler.step()
            self.save("last.pt")
        return summary

    # -- checkpoints --------------------------------------------------------

    def save(self, name: str) -> Optional[Path]:
        if self.logbook is None:
            return None
        path = self.logbook.checkpoints_dir / name
        torch.save({"epoch": self.epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "best_metric": self.best_metric}, path)
        return path

    def load(self, path: "str | Path", strict: bool = True) -> None:
        """Resume. ``strict=False`` is how an ablation checkpoint loads into a
        full model -- the parameter names match by design, so only genuinely
        absent modules are skipped."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch = int(state["epoch"]) + 1
        self.best_metric = float(state.get("best_metric", float("-inf")))
        logger.info("resumed from %s at epoch %d", path, self.epoch)
