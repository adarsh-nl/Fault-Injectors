"""
backbone.py
-----------
Multi-scale image feature extraction for the camera track.

**This now lives in** :mod:`cpbench.models.image`. It moved when ``w2cbench``
became the second package to need a ResNet feature pyramid: the backbone is not
CoBEVT's contribution, and two copies of it would be two places to fix the
next time torchvision changes a weights API.

This module re-exports, so every existing import path keeps working.
"""

from __future__ import annotations

from cpbench.models.image import IMAGENET_MEAN, IMAGENET_STD, ResnetEncoder

__all__ = ["ResnetEncoder", "IMAGENET_MEAN", "IMAGENET_STD"]
