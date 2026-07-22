"""
confidence.py
-------------
Stage 2: the spatial confidence map -- the paper's contribution in one tensor.

    C_i^(k) = Phi_generator(F_i^(k))  in  [0, 1]^(H x W)

Concretely (assumption A2, matching the released implementation):

    cls = head(F)                       (L, A, H, W)   detection logits
    C   = sigmoid(cls).max(dim=1)       (L, 1, H, W)   strongest evidence
    C   = gaussian_smooth(C)            (L, 1, H, W)   A9

Why max and not mean
--------------------
The question a confidence map answers is "can this agent perceive *something*
here", which is a maximum over evidence. A mean is diluted by the anchors that
found nothing, so a cell holding one clearly-seen vehicle scores lower than an
ambiguous cell where every anchor is mildly unsure -- and the clearly-seen
vehicle is exactly the cell worth transmitting. The released code agrees:

    ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(
        dim=1, keepdim=True)

``lgcpbench.confidence.pooling`` reaches the same conclusion for the same
reason in a different protocol, and offers mean and top-k as ablations there.

One decoder, two call sites
---------------------------
The paper reuses the detection decoder's parameters for the generator, so
there is exactly ONE ``DetectionHead`` in the model. This module owns it, and
the orchestrator reaches through ``self.confidence.head`` to decode the fused
map at the end of the last round. That is deliberately visible rather than
tidied away: sharing the head means a fault or a gradient that changes
detection also changes *what gets transmitted*, and a reader should not have
to discover that coupling from a state-dict diff.

Why the head's own taps are suppressed here
-------------------------------------------
``cpbench``'s ``DetectionHead`` emits ``head/cls_logits`` and friends. A
pre-fusion invocation is a different observation point from the final decode:
one drives selection, the other is the model's answer. Letting both land on
``head/cls_logits`` would merge two semantically different tensors into one
location, so the generator calls the head without taps and emits under its own
``confidence/r{k}/*`` names. The final decode passes taps through and keeps
``head/*``.

Round 0 is the released code's ``psm_single``
---------------------------------------------
At k=0 the head is applied to the un-fused, single-agent feature map, which is
what A11 supervises separately. The names are round-templated rather than
carrying a distinct ``single`` prefix because with K > 1 the generator runs
every round, and a separate name would either duplicate the k=0 tensor or
silently cover only the first round.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import nn

from cpbench.models import DetectionHead
from cpbench.observation import TapProtocol, emit

from ..comm.smoothing import GaussianSmoother

logger = logging.getLogger(__name__)


class SpatialConfidenceGenerator(nn.Module):
    """Turn per-agent BEV features into a spatial confidence map.

    Purpose
        Produce ``C_i``, the tensor that decides where an agent communicates,
        how strongly its message is weighted in fusion, and -- via
        ``R = 1 - C`` -- where it asks others to look.

    Inputs
    ------
    in_channels   D, the encoder's output width.
    num_anchors   anchors per BEV cell (PointPillars: 2).
    num_classes   detection classes (paper: 1, vehicles).
    smoother      a :class:`GaussianSmoother`, or None to disable (A9).

    Outputs (from :meth:`forward`)
    ------------------------------
    dict with

    ==============  ====================  ==================================
    ``cls``         ``(L, A*n_cls, H, W)``  detection logits
    ``reg``         ``(L, A*7, H, W)``      box regression
    ``confidence``  ``(L, 1, H, W)``        C_i, smoothed, in [0, 1]
    ``raw``         ``(L, 1, H, W)``        C_i before smoothing
    ==============  ====================  ==================================

    ``reg`` is returned even though the generator does not use it: at k=0 it
    is the released ``rm_single`` that A11 supervises, and the loss needs it.

    Taps emitted
    ------------
    ``confidence/r{k}/cls_logits``, ``/reg_map``, ``/sigmoid``, ``/map``, and
    ``/smoothed`` when a smoother is configured.

    Example
    -------
    >>> import torch
    >>> gen = SpatialConfidenceGenerator(in_channels=16, num_anchors=2).eval()
    >>> out = gen(torch.randn(3, 16, 8, 8))
    >>> out["confidence"].shape, out["cls"].shape
    (torch.Size([3, 1, 8, 8]), torch.Size([3, 2, 8, 8]))
    >>> bool((out["confidence"] >= 0).all() and (out["confidence"] <= 1).all())
    True

    The head is the model's only decoder, exposed for the final decode:

    >>> isinstance(gen.head, DetectionHead)
    True
    """

    def __init__(self, in_channels: int, num_anchors: int = 2,
                 num_classes: int = 1,
                 smoother: Optional[GaussianSmoother] = None) -> None:
        super().__init__()
        self.head = DetectionHead(in_channels=in_channels,
                                  num_anchors=num_anchors,
                                  num_classes=num_classes)
        self.smoother = smoother
        logger.info("SpatialConfidenceGenerator(D=%d, A=%d, classes=%d, "
                    "smoothing=%s)", in_channels, num_anchors, num_classes,
                    "on" if smoother is not None else "off")

    def forward(self, features: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> Dict[str, torch.Tensor]:
        """Generate the confidence map for one round.

        Shapes
        ------
        in   ``features`` (L, D, H, W) -- one map per real agent
        out  see the class docstring

        Every step is a separate statement with its own tap, so a fault
        benchmark can watch the logits, the per-anchor objectness, the
        collapsed map and the smoothed map independently. Composing them
        would leave the reduction -- the step that decides what gets
        transmitted -- unobservable.
        """
        # No taps: this module owns the naming of a pre-fusion invocation.
        predictions = self.head(features)
        cls, reg = predictions["cls"], predictions["reg"]
        emit(taps, cls, module="SpatialConfidenceGenerator",
             location=f"confidence/r{round_index}/cls_logits")
        emit(taps, reg, module="SpatialConfidenceGenerator",
             location=f"confidence/r{round_index}/reg_map")

        objectness = torch.sigmoid(cls)
        emit(taps, objectness, module="SpatialConfidenceGenerator",
             location=f"confidence/r{round_index}/sigmoid")

        raw = objectness.max(dim=1, keepdim=True).values          # A2
        emit(taps, raw, module="SpatialConfidenceGenerator",
             location=f"confidence/r{round_index}/map")

        confidence = raw
        if self.smoother is not None:
            confidence = self.smoother(raw, taps=taps,
                                       round_index=round_index)
        return {"cls": cls, "reg": reg, "confidence": confidence, "raw": raw}

    def decode(self, features: torch.Tensor,
               taps: Optional[TapProtocol] = None,
               branch: str = "fused") -> Dict[str, torch.Tensor]:
        """Run the shared head as the model's final detection decoder.

        The same parameters as :meth:`forward` uses (A2), but taps are passed
        through so the result lands on ``head/cls_logits`` / ``head/reg_map``
        / ``head/cls_sigmoid`` -- the model's answer, not the selection
        signal.

        Shapes
        ------
        in   ``features`` (B, D, H, W) -- the fused ego map
        out  ``{"cls": (B, A*n_cls, H, W), "reg": (B, A*7, H, W)}``

        >>> import torch
        >>> gen = SpatialConfidenceGenerator(in_channels=16).eval()
        >>> gen.decode(torch.randn(1, 16, 8, 8))["cls"].shape
        torch.Size([1, 2, 8, 8])
        """
        return self.head(features, taps=taps, branch=branch)
