"""Paper-agnostic training pieces.

    losses.py  focal + smooth-L1 detection objective, shared by every
               anchor-based detector here

A paper's own objective -- a segmentation loss, a multi-round schedule, a
distillation term -- stays in its package. Only the parts with exactly one
correct implementation live here.
"""

from .losses import DetectionLoss, sigmoid_focal_loss

__all__ = ["DetectionLoss", "sigmoid_focal_loss"]
