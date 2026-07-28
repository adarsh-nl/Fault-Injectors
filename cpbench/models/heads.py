"""
heads.py
--------
Prediction heads: confidence (H_conf, paper Eq. 2) and detection (cls+reg).

Both are 1x1-conv heads over BEV features and are shared across agents; the
detection head is reused by the ego LC branch and by every collaborator's
local branch, so all detection maps live in the same output space -- a
prerequisite for PAC's map-level correction.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..observation.taps import TapProtocol, emit


class ConfidenceHead(nn.Module):
    """Spatial confidence head H_conf.

    Purpose  produce the 1-channel confidence map used by CIT (message M1)
             and by LC's confidence weighting.
    Inputs   F (N, C, H, W).
    Outputs  logits (N, 1, H, W); use `probability()` for sigma(logits).
    """

    def __init__(self, in_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1))

    def forward(self, features: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        logits = self.net(features)
        emit(taps, logits, module="ConfidenceHead", location="confidence/logits")
        return logits

    def probability(self, features: torch.Tensor,
                    taps: Optional[TapProtocol] = None) -> torch.Tensor:
        prob = torch.sigmoid(self.forward(features, taps=taps))
        emit(taps, prob, module="ConfidenceHead", location="confidence/map")
        return prob


class DetectionHead(nn.Module):
    """Anchor-based detection head (PointPillars convention).

    Inputs   F (N, C, H, W).
    Outputs  dict:
        cls (N, A*num_classes, H, W) logits
        reg (N, A*reg_dim, H, W) box deltas (TargetAssigner encoding)
    `branch` labels the tap context ('local', 'lc', ...).
    """

    def __init__(self, in_channels: int, num_anchors: int = 2,
                 num_classes: int = 1, reg_dim: int = 7) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Width of the regression encoding. 7 = SECOND/VoxelNet deltas with a
        # single sin(yaw) channel; 8 adds the cos companion so yaw decodes with
        # atan2 over the full circle (RECON-5). DEFAULT 7 is load-bearing:
        # lgcpbench decodes released OpenCOOD checkpoints through the shared
        # BoxDecoder and those weights have num_anchors*7 regression heads.
        self.reg_dim = int(reg_dim)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * self.reg_dim, 1)
        # focal-loss-friendly prior: rare positives at init
        nn.init.constant_(self.cls_head.bias, -4.59)   # sigmoid ~= 0.01

    def forward(self, features: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                branch: str = "lc") -> Dict[str, torch.Tensor]:
        cls = self.cls_head(features)
        reg = self.reg_head(features)
        emit(taps, cls, module="DetectionHead", location="head/cls_logits",
             branch=branch)
        emit(taps, reg, module="DetectionHead", location="head/reg_map",
             branch=branch)
        emit(taps, torch.sigmoid(cls.detach()), module="DetectionHead",
             location="head/cls_sigmoid", branch=branch)
        return {"cls": cls, "reg": reg}
