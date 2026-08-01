"""
stub.py
-------
A faithful test double for OpenCOOD models.

Why this exists
    OpenCOOD is locked to Python 3.7 with ``numba==0.49.0``, ``spconv`` and
    CUDA. It cannot run in this project's environment or in CI, so the
    adapter's own logic -- submodule driving, the eval-mode guard, checkpoint
    verification, per-model fusion dispatch, area restriction -- would
    otherwise be entirely untested until someone ran a cluster job.

    These stubs mirror the STRUCTURE the adapter depends on, verified against
    OpenCOOD sources: the submodule names, the dict-in/dict-out convention of
    ``pillar_vfe``/``scatter``/``backbone``, the ``spatial_features_2d`` key,
    the ``fusion_net.fuse_modules`` layout, and each fusion module's calling
    signature.

What this does NOT verify
    Numerical fidelity. These modules produce correctly-shaped tensors, not
    OpenCOOD's actual values. A passing test here means the adapter drives the
    real model correctly; it does NOT mean the reproduced Table II numbers are
    right. That can only be established by running against real weights on the
    cluster.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class StubPillarVFE(nn.Module):
    """Mirrors ``PillarVFE``: dict in, dict out with ``pillar_features``."""

    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        n = int(batch["voxel_features"].shape[0])
        batch["pillar_features"] = torch.zeros(n, self.out_channels)
        return batch


class StubScatter(nn.Module):
    """Mirrors ``PointPillarScatter``: adds ``spatial_features``."""

    def __init__(self, grid_hw: Tuple[int, int], channels: int = 64) -> None:
        super().__init__()
        self.grid_hw = grid_hw
        self.channels = channels

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        n_agents = int(batch["record_len"].sum())
        h, w = self.grid_hw
        batch["spatial_features"] = torch.zeros(n_agents, self.channels, h, w)
        return batch


class StubBackbone(nn.Module):
    """Mirrors ``BaseBEVBackbone``: adds ``spatial_features_2d``."""

    def __init__(self, feature_hw: Tuple[int, int], channels: int = 256) -> None:
        super().__init__()
        self.feature_hw = feature_hw
        self.channels = channels
        self.proj = nn.Conv2d(64, channels, 1)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        n_agents = int(batch["spatial_features"].shape[0])
        h, w = self.feature_hw
        batch["spatial_features_2d"] = torch.randn(n_agents, self.channels, h, w)
        return batch


class StubAttentionFusion(nn.Module):
    """Mirrors ``AttentionFusion``: (V, C, h, w) -> (C, h, w), keeping index 0."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cav_num, c, h, w = x.shape
        tokens = x.view(cav_num, c, -1).permute(2, 0, 1)
        attended = torch.softmax(tokens @ tokens.transpose(-2, -1), dim=-1) @ tokens
        return attended.permute(1, 2, 0).view(cav_num, c, h, w)[0]


class StubWhere2commFusionNet(nn.Module):
    """Mirrors ``Where2comm``: a ``fuse_modules`` ModuleList when multi-scale."""

    def __init__(self, channels: int = 256, multi_scale: bool = True) -> None:
        super().__init__()
        if multi_scale:
            self.fuse_modules = nn.ModuleList(
                [StubAttentionFusion(c) for c in (64, 128, channels)]
            )
        else:
            self.fuse_modules = StubAttentionFusion(channels)


class StubSwapFusionEncoder(nn.Module):
    """Mirrors CoBEVT's ``SwapFusionEncoder``: (B,L,H,W,C) + mask -> (B,C,H,W)."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # channel-last in, channel-first out; keep the ego element
        return x[:, 0].permute(0, 3, 1, 2)


class StubAttWithWarp(nn.Module):
    """Mirrors CoAlign's ``Att_w_Warp``: (V,C,h,w) + record_len + affine."""

    def forward(
        self, xx: torch.Tensor, record_len: torch.Tensor, affine: torch.Tensor
    ) -> torch.Tensor:
        return xx[0]


class StubOpenCOODModel(nn.Module):
    """An OpenCOOD-shaped model with the submodules the adapter drives.

    Example
    -------
    >>> m = StubOpenCOODModel(core_method="point_pillar_where2comm",
    ...                       grid_hw=(64, 192), feature_hw=(16, 48))
    >>> m.eval() is m
    True
    >>> hasattr(m, "cls_head") and hasattr(m, "fusion_net")
    True
    """

    def __init__(
        self,
        core_method: str = "point_pillar_where2comm",
        grid_hw: Tuple[int, int] = (64, 192),
        feature_hw: Tuple[int, int] = (16, 48),
        channels: int = 256,
        num_anchors: int = 2,
        shrink: bool = False,
    ) -> None:
        super().__init__()
        self.core_method = core_method
        self.pillar_vfe = StubPillarVFE()
        self.scatter = StubScatter(grid_hw)
        self.backbone = StubBackbone(feature_hw, channels)
        self.shrink_flag = shrink
        if shrink:
            self.shrink_conv = nn.Conv2d(channels, channels, 1)

        if core_method == "point_pillar_where2comm":
            self.fusion_net = StubWhere2commFusionNet(channels)
        elif core_method == "point_pillar_cobevt":
            self.fusion_net = StubSwapFusionEncoder()
        elif core_method == "point_pillar_coalign":
            self.fusion_net = nn.ModuleList([StubAttWithWarp() for _ in range(3)])
        else:
            raise KeyError(f"unknown core_method {core_method!r}")

        self.cls_head = nn.Conv2d(channels, num_anchors, 1)
        self.reg_head = nn.Conv2d(channels, 7 * num_anchors, 1)


def stub_agent_inputs(n_agents: int = 3, n_voxels: int = 40):
    """AgentInputs carrying an OpenCOOD-shaped ``processed_lidar`` payload."""
    import numpy as np

    from ..protocol import AgentInputs

    return AgentInputs(
        features=torch.zeros(0, 1, 10),
        coords=torch.zeros(0, 3, dtype=torch.long),
        num_points=torch.zeros(0, dtype=torch.long),
        n_agents=n_agents,
        agent_ids=tuple(f"cav{i}" for i in range(n_agents)),
        positions=np.zeros((n_agents, 2)),
        extra={
            "processed_lidar": {
                "voxel_features": torch.randn(n_voxels, 32, 4),
                "voxel_coords": torch.zeros(n_voxels, 4, dtype=torch.long),
                "voxel_num_points": torch.ones(n_voxels, dtype=torch.long),
            }
        },
    )
