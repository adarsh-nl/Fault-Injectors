"""
losses.py
---------
V2X-ViT's objective: focal classification + smooth-L1 regression on the
fused ego output.

Deliberately thin. The paper's loss (section 4.4: focal + smooth-L1,
cls weight 1.0, reg weight 2.0) is exactly the shared
``cpbench.training.DetectionLoss`` -- which is consolidated for a reason:
two packages once carried diverged copies that agreed at ``num_classes=1``
and disagreed above it, and nothing failed. What stays here is only the
paper-specific part: anchor target assignment from the batch's ground
truth, and the weighting.

Unlike Where2comm there is no per-round or pre-fusion supervision --
V2X-ViT supervises one output, the fused ego map -- so this module is the
whole objective.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch import nn

from cpbench.training.losses import DetectionLoss

logger = logging.getLogger(__name__)


class V2XViTLoss(nn.Module):
    """Detection loss on the fused ego output.

    Purpose
        Turn a model output dict and a batch into the scalar the trainer
        steps on, and the components the logbook records.

    Inputs
    ------
    assigner    a ``cpbench.data.TargetAssigner``; encodes ground-truth
                boxes onto the anchor grid.
    alpha, gamma  focal parameters (reference: 0.25, 2.0)
    cls_weight  weight on the classification term (reference: 1.0)
    reg_weight  weight on the regression term (reference: 2.0)
    num_classes classes per anchor (reference: 1, cars only)

    Outputs
    -------
    ``{"loss", "loss_cls", "loss_reg"}`` -- the first is what to
    back-propagate, the rest are what to log.

    Example
    -------
    >>> import numpy as np, torch
    >>> from cpbench.data import AnchorGenerator, GridSpec, TargetAssigner
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
    ...                 downsample=4)
    >>> loss_fn = V2XViTLoss(TargetAssigner(AnchorGenerator(spec)))
    >>> output = {"cls": torch.zeros(1, 2, 16, 16),
    ...           "reg": torch.zeros(1, 14, 16, 16)}
    >>> batch = {"gt_boxes": [np.zeros((0, 7), dtype=np.float32)]}
    >>> out = loss_fn(batch, output)
    >>> sorted(out), bool(torch.isfinite(out["loss"]))
    (['loss', 'loss_cls', 'loss_reg'], True)
    """

    def __init__(self, assigner, alpha: float = 0.25, gamma: float = 2.0,
                 cls_weight: float = 1.0, reg_weight: float = 2.0,
                 num_classes: int = 1) -> None:
        super().__init__()
        self.assigner = assigner
        self.cls_weight = float(cls_weight)
        # DetectionLoss composes loss = loss_cls + reg_weight * loss_reg; the
        # cls weight is applied here so the shared term stays untouched.
        self.detection = DetectionLoss(alpha, gamma,
                                       reg_weight / max(cls_weight, 1e-12),
                                       num_classes)

    def _targets(self, gt_boxes: Sequence[Any], device) -> tuple:
        cls_targets, reg_targets = [], []
        for boxes in gt_boxes:
            assigned = self.assigner(
                boxes if boxes is not None
                else np.zeros((0, 7), dtype=np.float32))
            cls_targets.append(assigned["cls_target"])
            reg_targets.append(assigned["reg_target"])
        return (torch.stack(cls_targets).to(device),
                torch.stack(reg_targets).to(device))

    def forward(self, batch: Dict[str, Any],
                output: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        cls_target, reg_target = self._targets(batch["gt_boxes"],
                                               output["cls"].device)
        terms = self.detection(output["cls"], output["reg"],
                               cls_target, reg_target)
        return {"loss": self.cls_weight * terms["loss"],
                "loss_cls": (self.cls_weight * terms["loss_cls"]).detach(),
                "loss_reg": terms["loss_reg"].detach()}
