"""
losses.py
---------
Anchor-based detection objectives, shared by every detection paper here.

Focal classification plus smooth-L1 box regression is not any paper's
contribution -- it is the PointPillars/RetinaNet objective every anchor-based
BEV detector in this repository trains against. It lived in ``cobevtbench``
while there was one consumer.

Extracted when ``w2cbench`` turned out to hold a *divergent* second copy rather
than a duplicate: that one collapsed the class axis, writing every positive
anchor's target into channel 0 regardless of its label. With ``num_classes: 1``
-- the default in every shipped config -- the two agree exactly. Above that,
the second copy trained the wrong class and nothing failed; the loss still fell,
on the wrong objective.

That is the argument for one implementation rather than two correct-looking
ones, and it is why the merge adopted cobevtbench's class handling wholesale.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0
                       ) -> torch.Tensor:
    """Element-wise focal loss on raw logits.

    Computed from logits rather than probabilities: ``log(sigmoid(x))`` via
    ``binary_cross_entropy_with_logits`` is numerically stable where
    ``log(sigmoid(x))`` computed in two steps saturates and returns -inf for
    confident negatives, which is most anchors.

    >>> import torch
    >>> loss = sigmoid_focal_loss(torch.zeros(4), torch.ones(4))
    >>> bool(torch.isfinite(loss).all())
    True
    """
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * ce


class DetectionLoss(nn.Module):
    """Focal classification plus smooth-L1 box regression.

    Purpose
        The LiDAR track's objective (assumption A10 -- reconstructed from the
        paper's one-line description plus OpenCOOD convention).

    Inputs
    ------
    alpha, gamma  focal parameters
    reg_weight    weight on the regression term
    num_classes   classes per anchor

    Outputs
    -------
    ``{"loss", "loss_cls", "loss_reg"}``.

    Shapes
    ------
    cls_map     (B, A*num_classes, H, W)
    reg_map     (B, A*reg_dim, H, W)
    cls_target  (B, H, W, A)      1 positive, 0 negative, -1 ignore
    reg_target  (B, H, W, A, reg_dim)

    Regression is computed on positive anchors only. Including negatives
    would train the box head on anchors that have no box, which dominates the
    term by count and drags every prediction toward the anchor mean.

    Example
    -------
    >>> import torch
    >>> loss_fn = DetectionLoss()
    >>> out = loss_fn(torch.zeros(1, 2, 8, 8), torch.zeros(1, 14, 8, 8),
    ...               torch.zeros(1, 8, 8, 2), torch.zeros(1, 8, 8, 2, 7))
    >>> sorted(out), bool(torch.isfinite(out["loss"]))
    (['loss', 'loss_cls', 'loss_reg'], True)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reg_weight: float = 2.0, num_classes: int = 1,
                 reg_dim: int = 7) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reg_weight = float(reg_weight)
        # See DetectionHead.reg_dim; must match the TargetAssigner.
        self.reg_dim = int(reg_dim)
        self.num_classes = int(num_classes)

    def forward(self, cls_map: torch.Tensor, reg_map: torch.Tensor,
                cls_target: torch.Tensor,
                reg_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, _, height, width = cls_map.shape
        n_anchors = cls_target.shape[-1]

        cls_pred = cls_map.reshape(batch, n_anchors, self.num_classes,
                                   height, width)
        cls_pred = cls_pred.permute(0, 3, 4, 1, 2).reshape(-1, self.num_classes)
        labels = cls_target.reshape(-1)

        valid = labels >= 0                       # -1 = ignore
        positive = labels > 0
        n_positive = int(positive.sum())

        one_hot = torch.zeros_like(cls_pred)
        if self.num_classes == 1:
            one_hot[:, 0] = (labels > 0).to(cls_pred.dtype)
        else:
            index = labels.clamp(min=0).long()
            one_hot.scatter_(1, index.unsqueeze(1), 1.0)
            one_hot[labels <= 0] = 0.0

        focal = sigmoid_focal_loss(cls_pred, one_hot, self.alpha, self.gamma)
        # Normalise by positives, not by anchor count: the latter makes the
        # loss scale with grid size, so changing the BEV resolution silently
        # rescales the learning rate.
        loss_cls = focal[valid].sum() / max(n_positive, 1)

        reg_pred = reg_map.reshape(batch, n_anchors, self.reg_dim,
                                   height, width)
        reg_pred = reg_pred.permute(0, 3, 4, 1, 2).reshape(-1, self.reg_dim)
        reg_true = reg_target.reshape(-1, self.reg_dim)
        if n_positive:
            loss_reg = F.smooth_l1_loss(reg_pred[positive], reg_true[positive],
                                        reduction="sum") / n_positive
        else:
            # Keeps the graph connected on an empty frame; a bare zero would
            # detach the head and silently skip its gradient for that step.
            loss_reg = reg_pred.sum() * 0.0

        total = loss_cls + self.reg_weight * loss_reg
        return {"loss": total, "loss_cls": loss_cls.detach(),
                "loss_reg": loss_reg.detach()}

    def extra_repr(self) -> str:
        return (f"alpha={self.alpha}, gamma={self.gamma}, "
                f"reg_weight={self.reg_weight}")
