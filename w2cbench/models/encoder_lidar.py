"""
encoder_lidar.py
----------------
The LiDAR track's Stage-1 encoder: PointPillars, wrapped to the
:class:`~w2cbench.models.encoder.ObservationEncoder` contract.

Almost nothing here is new code, and that is the point. The pillar feature
encoder, the scatter and the BEV backbone are ``cpbench``'s, shared verbatim
with corabench and cobevtbench, which means a Where2comm-vs-CoBEVT robustness
comparison differs in the communication and fusion stages and nowhere else:
same voxelisation, same tap names (``encoder/pillar_features``,
``encoder/scatter_bev``, ``encoder/bev_features``), same anchor conventions,
same AP definition. A reimplemented encoder would put an uncontrolled variable
between two papers' numbers, which is the failure mode this repository's
shared core exists to prevent.

What this module adds is the contract and the geometry checks -- the two
things a wrapper is for.

    batch["features"]   (P, T, 10)  -> PillarVFE          -> (P, C_vfe)
    batch["coords"]     (P, 3)     -> PointPillarScatter -> (N, C_vfe, H0, W0)
                                   -> BEVBackbone        -> (N, D, H, W)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import torch

from cpbench.data import GridSpec
from cpbench.models import (PointPillarEncoder,
                            validate_backbone_geometry)
from cpbench.observation import TapProtocol

from .encoder import ObservationEncoder

logger = logging.getLogger(__name__)


class LidarPillarEncoder(ObservationEncoder):
    """PointPillars as a Where2comm observation encoder.

    Purpose
        Produce ``F^(0)``, the per-agent BEV feature map the spatial
        confidence generator reads and the communication module transmits.

    Inputs
    ------
    grid                cpbench :class:`GridSpec`; drives voxelisation, the
                        feature resolution, the spatial warp and the anchors
    in_channels         decorated point features (OpenCOOD PointPillars: 10)
    vfe_channels        pillar feature width
    block_channels / block_strides / block_layers
                        the BEV backbone pyramid
    upsample_channels   width each pyramid level is upsampled to
    out_channels        D, the width leaving the encoder (paper: 256)

    Outputs
    -------
    ``(N, D, H, W)`` where ``N = sum(batch["record_len"])`` and ``(H, W)`` is
    ``grid.feature_hw``.

    Batch keys read
    ---------------
    ``features`` (P, T, 10), ``coords`` (P, 3) as [agent, row, col],
    ``num_points`` (P,), ``record_len`` (B,).

    Taps emitted
    ------------
    ``encoder/pillar_features``, ``encoder/scatter_bev`` and
    ``encoder/bev_features`` -- all three from the cpbench modules that
    produce them, so this wrapper emits nothing of its own and cannot
    double-count.

    Example
    -------
    >>> import torch
    >>> from cpbench.data import GridSpec
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> enc = LidarPillarEncoder(spec, out_channels=32)
    >>> enc.out_channels, enc.feature_hw
    (32, (32, 32))
    >>> batch = {"features": torch.randn(4, 8, 10),
    ...          "coords": torch.tensor([[0, 1, 1], [0, 2, 2],
    ...                                  [1, 3, 3], [1, 4, 4]]),
    ...          "num_points": torch.full((4,), 8),
    ...          "record_len": [2]}
    >>> enc.eval()(batch).shape
    torch.Size([2, 32, 32, 32])
    """

    def __init__(self, grid: GridSpec, in_channels: int = 10,
                 vfe_channels: int = 64,
                 block_channels: Sequence[int] = (64, 128, 256),
                 block_strides: Sequence[int] = (2, 2, 2),
                 block_layers: Sequence[int] = (3, 5, 5),
                 upsample_channels: int = 128,
                 out_channels: int = 256) -> None:
        block_strides = tuple(int(s) for s in block_strides)
        self._validate_geometry(grid, block_strides)
        super().__init__(out_channels=out_channels, feature_hw=grid.feature_hw)

        self.grid = grid
        self.block_strides = block_strides
        self.encoder = PointPillarEncoder(
            grid_hw=grid.grid_hw, in_channels=in_channels,
            vfe_channels=vfe_channels, block_channels=block_channels,
            block_strides=block_strides, block_layers=block_layers,
            upsample_channels=upsample_channels, out_channels=out_channels)
        logger.info("LidarPillarEncoder: pillars %s -> features %s, D=%d",
                    grid.grid_hw, grid.feature_hw, out_channels)

    # -- eager validation ---------------------------------------------------

    @staticmethod
    def _validate_geometry(grid: GridSpec,
                           block_strides: Sequence[int]) -> None:
        """Fail at construction, not twenty minutes into a cluster job.

        Delegates to ``cpbench.models.validate_backbone_geometry``: the same
        two mistakes are reachable from every package that drives a
        PointPillars backbone from a GridSpec, and the check was duplicated
        here first.
        """
        validate_backbone_geometry(grid, block_strides)

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, Any],
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        """Encode every agent's point cloud into its BEV feature map.

        Shapes
        ------
        in   features (P, T, 10), coords (P, 3), num_points (P,)
        out  (N, D, H, W), N = sum(record_len)
        """
        n_agents = self.total_agents(batch)
        features = self.encoder(batch["features"], batch["coords"],
                                batch["num_points"], n_agents, taps=taps)
        return self.validate_output(features)
