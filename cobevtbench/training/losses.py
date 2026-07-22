"""
losses.py
---------
Training objectives for both tracks.

Camera: weighted cross-entropy
------------------------------
``VanillaSegLoss``, reproducing the released ``vanilla_seg_loss.py``. The
class weights are extreme and deliberately so: the released config uses
``d_weights: 75.0`` for the vehicle class. A BEV map is overwhelmingly
background -- vehicles occupy on the order of 1% of the 256x256 grid -- so
unweighted cross-entropy is minimised by predicting background everywhere,
which scores ~99% pixel accuracy and 0 IoU. The weight is what makes the
task learnable at all, not a tuning refinement.

LiDAR: focal + smooth-L1
------------------------
The paper states only "two 3x3 conv layers for classification and
regression" (Appendix C.3), so the objective is the standard OpenCOOD
PointPillars one: sigmoid focal loss over anchors, smooth L1 on the seven
box parameters of positive anchors only. Recorded under assumption A10 as
part of the LiDAR track being a reconstruction rather than a port.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class VanillaSegLoss(nn.Module):
    """Weighted cross-entropy for BEV semantic segmentation.

    Purpose
        The camera track's objective, matching the released implementation.

    Inputs
    ------
    target       ``"dynamic"`` (background, vehicle) or ``"static"``
                 (background, drivable area, lane)
    d_weights    vehicle class weight (released config: 75.0)
    s_weights    drivable-area weight (released config: 2.0)
    l_weights    lane weight (released config: 4.0)
    coefficient  overall scale on the loss (``d_coe``/``s_coe``: 2.0 / 1.0)
    ignore_index label excluded from the loss, e.g. outside the sensed area

    Outputs
    -------
    ``{"loss": scalar, "loss_seg": scalar}`` -- a dict, not a bare tensor, so
    the trainer logs the same keys whichever track it is running.

    Shapes
    ------
    logits  (B, K, H, W)
    target  (B, H, W) int64 in [0, K)

    Example
    -------
    >>> import torch
    >>> loss_fn = VanillaSegLoss(target="dynamic")
    >>> logits = torch.zeros(2, 2, 8, 8)
    >>> labels = torch.zeros(2, 8, 8, dtype=torch.long)
    >>> out = loss_fn(logits, labels)
    >>> bool(torch.isfinite(out["loss"])), sorted(out)
    (True, ['loss', 'loss_seg'])

    The weighting is what makes a rare class visible. Note that
    ``F.cross_entropy`` divides by the *sum of the target classes' weights*,
    so the effect only appears once the prediction is actually wrong about
    the rare class -- with uniform logits every pixel has the same
    cross-entropy and the weighting cancels exactly:

    >>> labels[:, 3:5, 3:5] = 1                     # a small vehicle region
    >>> flat = torch.zeros(2, 2, 8, 8)
    >>> a = VanillaSegLoss("dynamic", d_weights=1.0)(flat, labels)["loss"]
    >>> b = VanillaSegLoss("dynamic", d_weights=75.0)(flat, labels)["loss"]
    >>> bool(torch.isclose(a, b))
    True

    Predict background confidently and the two diverge sharply -- which is
    the regime training is actually in, and why 75.0 is not a tuning detail:

    >>> confident = torch.zeros(2, 2, 8, 8)
    >>> confident[:, 0] = 5.0
    >>> confident[:, 1] = -5.0
    >>> weak = VanillaSegLoss("dynamic", d_weights=1.0)(confident, labels)["loss"]
    >>> strong = VanillaSegLoss("dynamic", d_weights=75.0)(confident, labels)["loss"]
    >>> bool(strong > 5 * weak)
    True
    """

    def __init__(self, target: str = "dynamic", d_weights: float = 75.0,
                 s_weights: float = 2.0, l_weights: float = 4.0,
                 coefficient: Optional[float] = None,
                 ignore_index: int = -100) -> None:
        super().__init__()
        if target == "dynamic":
            weights = [1.0, float(d_weights)]
            default_coefficient = 2.0
        elif target == "static":
            weights = [1.0, float(s_weights), float(l_weights)]
            default_coefficient = 1.0
        else:
            raise ValueError(
                f"unknown target {target!r}; expected 'dynamic' or 'static'")
        self.target = target
        self.coefficient = float(default_coefficient if coefficient is None
                                 else coefficient)
        self.register_buffer("weights", torch.tensor(weights),
                             persistent=False)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        if logits.shape[0] != target.shape[0]:
            raise ValueError(
                f"batch mismatch: logits {tuple(logits.shape)} vs target "
                f"{tuple(target.shape)}")
        if logits.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"logits are {tuple(logits.shape[-2:])} but targets are "
                f"{tuple(target.shape[-2:])}; the decoder output and the "
                "rasterised BEV grid must be the same size")
        seg = F.cross_entropy(logits, target.long(),
                              weight=self.weights.to(logits.dtype),
                              ignore_index=self.ignore_index)
        return {"loss": self.coefficient * seg, "loss_seg": seg.detach()}

    def extra_repr(self) -> str:
        return (f"target={self.target}, weights={self.weights.tolist()}, "
                f"coefficient={self.coefficient}")


# ---------------------------------------------------------------------------
# The detection objective moved to :mod:`cpbench.training.losses` when
# ``w2cbench`` was found to hold a divergent second copy (it collapsed the
# class axis). Re-exported so every import path here keeps working.
# ---------------------------------------------------------------------------

from cpbench.training.losses import DetectionLoss, sigmoid_focal_loss  # noqa: E402,F401
