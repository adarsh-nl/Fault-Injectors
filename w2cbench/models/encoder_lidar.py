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

    batch["features"]   (P, T, 9)  -> PillarVFE          -> (P, C_vfe)
    batch["coords"]     (P, 3)     -> PointPillarScatter -> (N, C_vfe, H0, W0)
                                   -> BEVBackbone        -> (N, D, H, W)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import torch

from cpbench.data import GridSpec
from cpbench.models import PointPillarEncoder
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
    in_channels         decorated point features (PointPillars: 9)
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
    ``features`` (P, T, 9), ``coords`` (P, 3) as [agent, row, col],
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
    >>> batch = {"features": torch.randn(4, 8, 9),
    ...          "coords": torch.tensor([[0, 1, 1], [0, 2, 2],
    ...                                  [1, 3, 3], [1, 4, 4]]),
    ...          "num_points": torch.full((4,), 8),
    ...          "record_len": [2]}
    >>> enc.eval()(batch).shape
    torch.Size([2, 32, 32, 32])
    """

    def __init__(self, grid: GridSpec, in_channels: int = 9,
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

        Two checks, both guarding failure modes that surface far from their
        cause.

        *Divisibility.* The backbone downsamples by the product of
        ``block_strides`` and upsamples each level back to the first level's
        resolution. An indivisible pillar grid produces levels that are off by
        one pixel and surfaces as ``Sizes of tensors must match ... Expected
        size 25 but got size 26`` from inside a ``torch.cat``, with nothing to
        suggest the real cause is a point range in a YAML file.

        *Declared vs. actual stride.* ``GridSpec.downsample`` is what the
        anchor generator, the spatial warp and the selection mask are all
        sized from, but the resolution the backbone ACTUALLY produces is
        ``grid_hw // block_strides[0]`` -- the first level is the one every
        other level is upsampled back to. When the two disagree, nothing
        raises: the encoder returns a feature map of one size while every
        consumer was built for another, and the first symptom is a shape
        error several modules downstream, or worse, a silently mismatched
        anchor grid that lowers AP without failing.
        """
        stride_product = 1
        for stride in block_strides:
            stride_product *= int(stride)
        pillar_h, pillar_w = grid.grid_hw
        if pillar_h % stride_product or pillar_w % stride_product:
            raise ValueError(
                f"pillar grid {pillar_h}x{pillar_w} does not divide by the "
                f"backbone stride product {stride_product} "
                f"(block_strides={tuple(block_strides)}). Choose a "
                "point_range and voxel_size whose ratio is a multiple of it.")

        if int(grid.downsample) != int(block_strides[0]):
            raise ValueError(
                f"grid.downsample={grid.downsample} disagrees with "
                f"block_strides[0]={block_strides[0]}. The BEV backbone "
                "upsamples every pyramid level back to the FIRST level's "
                "resolution, so its output is grid_hw // block_strides[0] "
                f"= {grid.grid_hw[0] // int(block_strides[0])} cells, while "
                f"GridSpec.feature_hw reports {grid.feature_hw} and the "
                "anchors, the spatial warp and the selection mask are all "
                "sized from that. Set them equal.")

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, Any],
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        """Encode every agent's point cloud into its BEV feature map.

        Shapes
        ------
        in   features (P, T, 9), coords (P, 3), num_points (P,)
        out  (N, D, H, W), N = sum(record_len)
        """
        n_agents = self.total_agents(batch)
        features = self.encoder(batch["features"], batch["coords"],
                                batch["num_points"], n_agents, taps=taps)
        return self.validate_output(features)
