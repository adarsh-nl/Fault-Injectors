"""Training: the multi-round objective, the loop, and mid-training scoring.

    losses.py     DetectionLoss + MultiRoundDetectionLoss (A11)
    trainer.py    the loop; paper-agnostic, driven by a loss closure
    validator.py  AP over a held-out split, for checkpoint selection
"""

from .losses import DetectionLoss, MultiRoundDetectionLoss
from .trainer import Trainer, TrainerConfig
from .validator import Validator

__all__ = ["DetectionLoss", "MultiRoundDetectionLoss", "Trainer",
           "TrainerConfig", "Validator"]
