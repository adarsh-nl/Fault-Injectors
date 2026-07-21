"""
Losses, Trainer, Validator.

Training is never tapped: `taps=None` throughout, so the measurement plane
costs one `is None` check per emit site and AMP autocast is unaffected.

Contents
--------
losses     VanillaSegLoss (camera) and DetectionLoss (lidar)
trainer    track-agnostic AMP loop with checkpointing and resume
validator  clean-condition validation; faults belong in evaluation/
"""

from .losses import DetectionLoss, VanillaSegLoss, sigmoid_focal_loss
from .trainer import Trainer
from .validator import DetectionValidator, SegmentationValidator

__all__ = ["VanillaSegLoss", "DetectionLoss", "sigmoid_focal_loss",
           "Trainer", "SegmentationValidator", "DetectionValidator"]
