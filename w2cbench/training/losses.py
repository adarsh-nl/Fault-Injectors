"""
losses.py
---------
The multi-round objective (assumption A11).

    L = sum_{k=0..K} sum_i L_det(O_i^(k), O_i)        (paper section 4.6)

Two distinct supervisions, and the second is load-bearing
----------------------------------------------------------
*The fused rounds.* Every round's decoded output is supervised, not just the
last. That is what makes an intermediate round a usable operating point rather
than merely a step toward one, and it is why a K=3 model can be evaluated at
K=1 without retraining.

*The pre-fusion output* (``psm_single`` / ``rm_single`` in the released code).
This is the only direct gradient the confidence head ever receives, and the
reason is structural rather than incidental: selection is a hard ``{0, 1}``
mask, so **no gradient flows from the fused loss back into the confidence
map**. Remove this term and the tensor that decides what gets transmitted
would be trained only through the path its gradient was supposed to come from.
The model would still converge -- the detection head is shared, so it is
trained by the fused loss -- but the confidence map would be a by-product
rather than a target, and the whole selection mechanism would be resting on an
untrained signal.

The detection term itself is shared
-----------------------------------
``DetectionLoss`` (focal + smooth-L1) comes from ``cpbench.training.losses``.
It used to be a local copy, and the copy was wrong above ``num_classes=1``: it
wrote every positive anchor's target into channel 0 regardless of its label, so
a three-class run trained class 0 and the loss fell on the wrong objective
without anything failing. What is paper-specific is the *schedule* -- which
rounds are supervised, and with what weight -- and that is what stays here.

Only the ego's pre-fusion output is supervised
-----------------------------------------------
Ground truth exists in the ego frame. A collaborator's pre-fusion prediction is
in *its own* frame, and the dataset has no labels there.

This is not a limitation, because of A2: the detection head is shared by every
agent. Training the ego's pre-fusion output trains exactly the parameters that
produce every collaborator's confidence map. The alternative -- warping labels
into each collaborator's frame -- would supervise the same weights with the
same objective through a noisier path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from cpbench.training.losses import DetectionLoss

logger = logging.getLogger(__name__)


class MultiRoundDetectionLoss(nn.Module):
    """Where2comm's objective: every round, plus the pre-fusion output (A11).

    Purpose
        Turn a model output dict and a batch into the scalar the trainer
        steps on, and the components the logbook records.

    Inputs
    ------
    assigner       a ``cpbench.data.TargetAssigner``; encodes ground-truth
                   boxes onto the anchor grid.
    single_weight  weight on the pre-fusion term (A11). Zero disables it,
                   which is the ablation that shows what an untrained
                   confidence map costs -- not a configuration to run.
    round_weights  per-round weights; None means uniform. A schedule that
                   ramps later rounds is a curriculum knob the paper hints at
                   but does not specify.

    Outputs
    -------
    ``{"loss", "loss_cls", "loss_reg", "loss_single", "loss_r{k}"}`` -- the
    first is what to back-propagate, the rest are what to log. The keys map
    onto ``cpbench.logbook.TrainRecord``'s fields in ``scripts/common.py``.

    Example
    -------
    >>> import numpy as np, torch
    >>> from cpbench.data import AnchorGenerator, GridSpec, TargetAssigner
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> loss_fn = MultiRoundDetectionLoss(TargetAssigner(AnchorGenerator(spec)))
    >>> output = {"rounds": [{"cls": torch.zeros(1, 2, 32, 32),
    ...                       "reg": torch.zeros(1, 14, 32, 32)}],
    ...           "single_cls": torch.zeros(1, 2, 32, 32),
    ...           "single_reg": torch.zeros(1, 14, 32, 32)}
    >>> batch = {"gt_boxes": [np.zeros((0, 7), dtype=np.float32)],
    ...          "record_len": [1]}
    >>> out = loss_fn(batch, output)
    >>> bool(torch.isfinite(out["loss"])), "loss_single" in out
    (True, True)
    """

    def __init__(self, assigner, alpha: float = 0.25, gamma: float = 2.0,
                 reg_weight: float = 2.0, num_classes: int = 1,
                 single_weight: float = 1.0,
                 round_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.assigner = assigner
        self.detection = DetectionLoss(alpha, gamma, reg_weight, num_classes)
        self.single_weight = float(single_weight)
        self.round_weights = (list(round_weights)
                              if round_weights is not None else None)

    # -- targets ------------------------------------------------------------

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

    @staticmethod
    def _ego_rows(record_len: Sequence[int]) -> List[int]:
        """Indices of each sample's ego within the flat agent axis.

        The pre-fusion output covers every agent, but only the ego's has
        ground truth in a matching frame -- see the module docstring.
        """
        rows, offset = [], 0
        for count in record_len:
            rows.append(offset)
            offset += int(count)
        return rows

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, Any],
                output: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        rounds = output["rounds"]
        device = rounds[0]["cls"].device
        cls_target, reg_target = self._targets(batch["gt_boxes"], device)

        weights = (self.round_weights if self.round_weights is not None
                   else [1.0] * len(rounds))
        if len(weights) != len(rounds):
            raise ValueError(
                f"round_weights has {len(weights)} entries but the model ran "
                f"{len(rounds)} rounds")

        total = torch.zeros((), device=device)
        parts: Dict[str, torch.Tensor] = {}
        loss_cls = torch.zeros((), device=device)
        loss_reg = torch.zeros((), device=device)
        for index, (prediction, weight) in enumerate(zip(rounds, weights)):
            terms = self.detection(prediction["cls"], prediction["reg"],
                                   cls_target, reg_target)
            total = total + weight * terms["loss"]
            loss_cls = loss_cls + weight * terms["loss_cls"]
            loss_reg = loss_reg + weight * terms["loss_reg"]
            parts[f"loss_r{index}"] = terms["loss"].detach()

        single = torch.zeros((), device=device)
        if self.single_weight > 0.0 and "single_cls" in output:
            rows = self._ego_rows(batch["record_len"])
            single_terms = self.detection(
                output["single_cls"][rows], output["single_reg"][rows],
                cls_target, reg_target)
            single = single_terms["loss"]
            total = total + self.single_weight * single

        return {"loss": total, "loss_cls": loss_cls.detach(),
                "loss_reg": loss_reg.detach(),
                "loss_single": (self.single_weight * single).detach(),
                **parts}
