"""
trainer.py
----------
The training loop, and nothing else.

The trainer takes a model, a loader and a ``loss_fn(batch, output) -> dict``.
It does not know what the model is, what the loss means, or which paper it
belongs to -- it steps, clips, logs and checkpoints. Everything paper-specific
is in the closure it was handed, which is what makes it testable with a
two-parameter stub and reusable when the camera track lands.

Taps are off during training, deliberately
------------------------------------------
Their cost is small but non-zero, and they produce nothing a training run
consumes. The intermediate tensors are the *product* only at evaluation time,
where the tester threads them through every forward so a fault run's dumps
line up frame-for-frame with the clean run's.

Communication volume is not logged during training either, and that one is not
a preference: the selector keeps a random fraction of the map in train mode
(A17), so a measured volume would be a draw from the curriculum rather than a
model decision. The accountant refuses outright.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from cpbench.logbook import ExperimentLogger, TrainRecord

logger = logging.getLogger(__name__)

LossFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, torch.Tensor]]


@dataclass
class TrainerConfig:
    """Everything the loop needs that is not an object.

    Attributes
    ----------
    epochs        passes over the training set.
    grad_clip     max gradient norm; None disables clipping.
    amp           mixed precision. Applied to the forward and backward only;
                  the confidence pathway's threshold comparison sits near the
                  fp16 precision floor at the released 0.01, so a bandwidth
                  number that shifted with the AMP setting would not be a
                  bandwidth number. Evaluation runs in fp32 regardless.
    log_every     batches between TrainRecord rows.
    eval_every    epochs between validation passes; 0 disables.
    checkpoint_every  epochs between checkpoints; the best model is always
                  written separately.
    """

    epochs: int = 1
    grad_clip: Optional[float] = 35.0
    amp: bool = False
    log_every: int = 20
    eval_every: int = 1
    checkpoint_every: int = 0


class Trainer:
    """Drive training, logging every step through the shared logbook.

    Purpose
        One place that owns the optimisation loop, so a paper package's
        contribution is not entangled with gradient clipping.

    Inputs
    ------
    model      any ``nn.Module`` whose forward takes ``(batch, taps=None)``.
    loss_fn    ``(batch, output) -> {"loss": Tensor, ...}``. Extra keys are
               logged; only ``loss`` is back-propagated.
    optimizer, scheduler   built by ``scripts/common.py`` from config.
    config     :class:`TrainerConfig`.
    logbook    ``ExperimentLogger``; None disables persistence (tests).
    validator  optional object with ``run(model) -> Dict[str, float]``;
               its ``score`` key, if present, selects the best checkpoint.
    device     where batches are moved.

    Outputs
    -------
    ``fit(loader)`` returns the training history as a list of
    :class:`TrainRecord`.

    Example
    -------
    >>> # see tests/test_training.py
    """

    def __init__(self, model: nn.Module, loss_fn: LossFn,
                 optimizer: torch.optim.Optimizer,
                 config: Optional[TrainerConfig] = None,
                 scheduler: Optional[Any] = None,
                 logbook: Optional[ExperimentLogger] = None,
                 validator: Optional[Any] = None,
                 device: Optional[torch.device] = None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config or TrainerConfig()
        self.scheduler = scheduler
        self.logbook = logbook
        self.validator = validator
        self.device = device or torch.device("cpu")
        self.scaler = (torch.cuda.amp.GradScaler()
                       if self.config.amp and self.device.type == "cuda"
                       else None)
        self.history: list = []
        self.best_score: Optional[float] = None

    # -- helpers ------------------------------------------------------------

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                terms = self.loss_fn(batch, self.model(batch))
            self.scaler.scale(terms["loss"]).backward()
            if self.config.grad_clip:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     self.config.grad_clip)
            else:
                grad_norm = torch.zeros(())
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            terms = self.loss_fn(batch, self.model(batch))
            terms["loss"].backward()
            grad_norm = (nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  self.config.grad_clip)
                         if self.config.grad_clip else torch.zeros(()))
            self.optimizer.step()
        terms["grad_norm"] = grad_norm
        return terms

    def _record(self, epoch: int, batch_index: int,
                terms: Dict[str, torch.Tensor],
                elapsed: float) -> TrainRecord:
        return TrainRecord(
            epoch=epoch, batch=batch_index,
            loss_total=float(terms["loss"]),
            loss_cls=float(terms.get("loss_cls", 0.0)),
            loss_reg=float(terms.get("loss_reg", 0.0)),
            # loss_align carries the pre-fusion single-agent term (A11).
            # Renaming the field would break a CSV schema four packages share
            # for cosmetic gain; the mapping is documented here instead.
            loss_align=float(terms.get("loss_single", 0.0)),
            lr=float(self.optimizer.param_groups[0]["lr"]),
            grad_norm=float(terms.get("grad_norm", 0.0)),
            batch_time_s=elapsed,
            gpu_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                        if self.device.type == "cuda" else 0.0))

    def _checkpoint(self, name: str, epoch: int) -> Optional[Path]:
        if self.logbook is None:
            return None
        path = self.logbook.checkpoints_dir / name
        torch.save({"epoch": epoch, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)
        return path

    # -- the loop -----------------------------------------------------------

    def fit(self, loader) -> list:
        """Train for ``config.epochs`` passes over `loader`."""
        for epoch in range(self.config.epochs):
            self.model.train()
            for batch_index, raw in enumerate(loader):
                started = time.perf_counter()
                terms = self._step(self._to_device(raw))
                record = self._record(epoch, batch_index, terms,
                                      time.perf_counter() - started)
                self.history.append(record)
                if (self.logbook is not None
                        and batch_index % max(self.config.log_every, 1) == 0):
                    self.logbook.log_train(record)
            if self.scheduler is not None:
                self.scheduler.step()
            self._end_of_epoch(epoch)
        if self.logbook is not None:
            self.logbook.flush()
        return self.history

    def _end_of_epoch(self, epoch: int) -> None:
        if (self.config.eval_every and self.validator is not None
                and (epoch + 1) % self.config.eval_every == 0):
            metrics = self.validator.run(self.model)
            score = metrics.get("score")
            logger.info("epoch %d: validation %s", epoch, metrics)
            if score is not None and (self.best_score is None
                                      or score > self.best_score):
                self.best_score = float(score)
                self._checkpoint("best.pt", epoch)
        if (self.config.checkpoint_every
                and (epoch + 1) % self.config.checkpoint_every == 0):
            self._checkpoint(f"epoch_{epoch:03d}.pt", epoch)
