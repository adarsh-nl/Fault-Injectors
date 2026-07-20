"""
lgcpbench.perception
====================
The seam between LGCP's control plane and the perception backbone it
orchestrates.

LGCP is backbone-agnostic by construction -- the paper itself says it "adopts
existing collaborative perception models". ``CollabPerceptionModel`` is that
contract; everything above it (grouping, leader election, scheduling) is
written against the protocol and never against a concrete model.

Example
-------
>>> import torch
>>> from lgcpbench.roi import AreaGrid
>>> from lgcpbench.perception import AreaFeatureMasker, NativeReferenceBackbone
>>> grid = AreaGrid((-140.8, -38.4, -3.0, 140.8, 38.4, 1.0))
>>> masker = AreaFeatureMasker(grid, feature_hw=(48, 176))
>>> model = NativeReferenceBackbone(grid_hw=(192, 704), feature_hw=(48, 176),
...                                 channels=8)
>>> feats = torch.zeros(3, 8, 48, 176)          # 3 CAVs, already encoded
>>> area_feats = [masker.extract(feats[i], area_id=200) for i in range(3)]
>>> tuple(model.fuse(area_feats[0], area_feats[1:]).shape)
(8, 4, 7)
"""

from .area_masking import DEFAULT_BITS_PER_ELEMENT, AreaFeatureMasker
from .native import NativeReferenceBackbone, PerPixelAttentionFusion
from .protocol import AgentInputs, CollabPerceptionModel, Detections

__all__ = [
    "AgentInputs",
    "Detections",
    "CollabPerceptionModel",
    "AreaFeatureMasker",
    "DEFAULT_BITS_PER_ELEMENT",
    "NativeReferenceBackbone",
    "PerPixelAttentionFusion",
]
