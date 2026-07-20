"""Generic PointPillars building blocks.

These are standard components, not any paper's contribution: the pillar VFE,
the scatter, the multi-scale BEV backbone, and anchor-based detection /
confidence heads. Paper-specific architecture lives in the paper package.
"""

from .encoder import (BEVBackbone, PillarVFE, PointPillarEncoder,
                      PointPillarScatter)
from .heads import ConfidenceHead, DetectionHead

__all__ = ["PillarVFE", "PointPillarScatter", "BEVBackbone",
           "PointPillarEncoder", "ConfidenceHead", "DetectionHead"]
