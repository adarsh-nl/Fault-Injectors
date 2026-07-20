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
                 device: Optional[torch.device] = None) -> None:
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

    # -- checkpointing ------------------------------------------------------

    def save_checkpoint(self, epoch: int, tag: str) -> Path:
        path = self.explog.checkpoints_dir / f"{tag}.pt"
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
                out = self.model(batch, return_teacher=True)
                losses = self.loss_fn(out, batch)
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running += float(losses["total"].item())

            self.explog.log_fault_records(batch["fault_records"])
            if i % self.log_every == 0:
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
                    batch_time_s=time.perf_counter() - t0,
                    gpu_mem_mb=mem)
                self.explog.log_train(rec)
                logger.info("epoch %d batch %d loss %.4f (cls %.4f reg %.4f "
                            "align %.4f)", epoch, i, rec.loss_total,
                            rec.loss_cls, rec.loss_reg, rec.loss_align)
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
