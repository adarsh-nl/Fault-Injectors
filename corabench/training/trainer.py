"""Training loop (spec §4) with the §5.6 non-finite-gradient guard."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..compat import autocast, grad_scaler

from ..selfcheck import grad_step_guard
from .losses import CoRALoss


class Trainer:
    def __init__(self, model, dataset, lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 2,
                 milestones=(15, 25), gamma: float = 0.1,
                 grad_clip: float = 10.0, amp: bool = True,
                 num_workers: int = 4, device: str = "cuda",
                 loss: Optional[CoRALoss] = None, logger=None) -> None:
        self.model = model.to(device)
        self.device = device
        self.loss_fn = (loss or CoRALoss()).to(device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=list(milestones), gamma=gamma)
        self.scaler = grad_scaler(enabled=amp and device == "cuda")
        self.amp = amp and device == "cuda"
        self.grad_clip = grad_clip
        self.logger = logger
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=dataset.collate,   # bound: works for both
                                 # CoRADataset (static) and CoRABatchAdapter
                                 drop_last=True)
        self.skipped_steps = 0

    def _to_device(self, batch: Dict) -> Dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        agg: Dict[str, float] = {}
        n = 0
        for batch in self.loader:
            batch = self._to_device(batch)
            with autocast(self.device, enabled=self.amp):
                out = self.model(batch)
                parts = self.loss_fn(out, batch)
            self.scaler.scale(parts["loss"]).backward()
            self.scaler.unscale_(self.opt)
            # §5.6: never step on a non-finite gradient
            stepped, gnorm = grad_step_guard(self.model, self.opt,
                                             self.grad_clip)
            self.scaler.update()
            if not stepped:
                self.skipped_steps += 1
            else:
                self.model.update_teacher()
            n += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            if self.logger is not None:
                self.logger.log_train(epoch=epoch, step=n,
                                      **{k: float(v.detach())
                                         for k, v in parts.items()},
                                      grad_norm=gnorm, stepped=stepped)
        self.sched.step()
        return {k: v / max(n, 1) for k, v in agg.items()}
