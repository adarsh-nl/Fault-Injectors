"""Generic encoder building blocks.

These are standard components, not any paper's contribution: the pillar VFE,
the scatter and the multi-scale BEV backbone for LiDAR, a ResNet feature
pyramid for camera, and anchor-based detection / confidence heads. A paper's
own architecture lives in its package.

``image`` imports torchvision, which is why it is imported lazily below: the
LiDAR-only packages should not need it installed.
"""

from .encoder import (BEVBackbone, PillarVFE, PointPillarEncoder,
                      PointPillarScatter, validate_backbone_geometry)
from .heads import ConfidenceHead, DetectionHead

__all__ = ["PillarVFE", "PointPillarScatter", "BEVBackbone",
           "PointPillarEncoder", "ConfidenceHead", "DetectionHead",
           "ResnetEncoder", "validate_backbone_geometry"]


def __getattr__(name: str):
    """Lazily expose the camera backbone without importing torchvision."""
    if name == "ResnetEncoder":
        from .image import ResnetEncoder
        return ResnetEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
