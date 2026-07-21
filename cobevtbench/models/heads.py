"""
heads.py
--------
BEV semantic segmentation head.

One convolution from the decoder's output width to per-class logits.

Two assumptions live here
-------------------------
**A3** -- the paper describes a 1x1 convolution; the released code uses 3x3
with padding 1. The code wins by default, ``kernel_size`` is exposed.

**A7** -- the reference's ``BevSegHead.__init__`` reads
``if target == 'dynamic' ... if target == 'static' ... else ...`` with a
second ``if`` rather than ``elif``, so the dynamic configuration also enters
the ``else`` branch and allocates a static head it never uses. Harmless to
the output, but it inflates the parameter count the paper reports and leaves
an optimizer state for weights that receive no gradient. Not reproduced: this
head builds exactly one convolution.

**A8** -- dynamic (vehicle) and static (road, lane) are two separately
trained models merged at inference, not one multi-head model. That is why
``target`` selects a single head rather than producing both.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit

TARGETS = {"dynamic": 2, "static": 3}


class BevSegHead(nn.Module):
    """Per-class BEV segmentation logits.

    Purpose
        Produce the tensor IoU is computed from.

    Inputs
    ------
    target       ``"dynamic"`` (background, vehicle) or ``"static"``
                 (background, drivable area, lane)
    input_dim    decoder output width (CoBEVT: 32)
    num_classes  override the class count implied by ``target``
    kernel_size  assumption A3; 3 in the released code, 1 in the paper

    Outputs
    -------
    ``{"logits": (B, K, H, W), "probs": (B, K, H, W), "labels": (B, H, W)}``

    ``labels`` is the argmax, returned rather than left to the caller because
    every consumer -- the IoU evaluator, the robustness pairing, the
    qualitative dump -- needs the same one, and an argmax taken twice over
    different axes is a silent way to disagree.

    Shapes
    ------
    x       (B, input_dim, 256, 256)
    logits  (B, K, 256, 256)   K = 2 dynamic, 3 static

    Example
    -------
    >>> import torch
    >>> head = BevSegHead(target="dynamic", input_dim=8)
    >>> out = head(torch.randn(2, 8, 16, 16))
    >>> out["logits"].shape, out["labels"].shape
    (torch.Size([2, 2, 16, 16]), torch.Size([2, 16, 16]))
    >>> bool(torch.allclose(out["probs"].sum(1), torch.ones(2, 16, 16)))
    True

    Exactly one head is built -- no dead parameters (assumption A7):

    >>> sum(1 for _ in head.children())
    1
    """

    def __init__(self, target: str = "dynamic", input_dim: int = 32,
                 num_classes: Optional[int] = None,
                 kernel_size: int = 3) -> None:
        super().__init__()
        if target not in TARGETS:
            raise ValueError(
                f"unknown target {target!r}; expected one of {sorted(TARGETS)}")
        self.target = target
        self.num_classes = int(num_classes if num_classes is not None
                               else TARGETS[target])
        self.head = nn.Conv2d(input_dim, self.num_classes, kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, taps: Optional[TapProtocol] = None,
                location_prefix: str = "head") -> Dict[str, torch.Tensor]:
        logits = self.head(x)
        emit(taps, logits, module="BevSegHead",
             location=f"{location_prefix}/seg_logits", target=self.target)

        probs = logits.softmax(dim=1)
        emit(taps, probs, module="BevSegHead",
             location=f"{location_prefix}/seg_softmax", target=self.target)

        labels = logits.argmax(dim=1)
        emit(taps, labels, module="BevSegHead",
             location=f"{location_prefix}/seg_argmax", target=self.target)

        return {"logits": logits, "probs": probs, "labels": labels}

    def extra_repr(self) -> str:
        return f"target={self.target}, num_classes={self.num_classes}"
