"""
cobevt_lidar.py
---------------
The LiDAR track: PointPillars -> regroup -> spatial warp -> FuseBEVT ->
detection head.

Paper Table 2 (AP@0.7 85.2, best of the compared fusion methods). The paper
describes this track in Appendix C.3 but the official repository contains no
LiDAR model, so this is a **reconstruction** rather than a port -- see
assumption A10. What is faithful is the part that matters: FuseBEVT is the
same module the camera track uses, unchanged, which is precisely the claim
Table 2 makes.

Everything except FuseBEVT is borrowed from ``cpbench``: the pillar encoder,
the detection head, the anchor and box conventions. That is deliberate. It
means a CoBEVT-vs-CoRA robustness comparison differs in the fusion block and
nothing else -- same voxelisation, same head, same AP definition, same tap
names. A reimplemented encoder would put an uncontrolled variable between
the two papers' numbers.

    batch --> PointPillarEncoder --> (N_total, C, H, W)
          --> regroup(record_len)  --> (B, L, C, H, W) + agent mask
          --> SpatialTransform     --> (B, L, C, H, W) + validity mask
          --> FuseBEVT             --> (B, C, H, W)
          --> DetectionHead        --> cls, reg
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from cpbench.data import GridSpec
from cpbench.models import (DetectionHead, PointPillarEncoder,
                            validate_backbone_geometry)
from cpbench.observation import TapProtocol, emit

from ..fusion.fusebevt import FuseBEVT
from ..fusion.geometry import SpatialTransform, regroup


class CoBEVTLidar(nn.Module):
    """CoBEVT's LiDAR detection track.

    Purpose
        Give the benchmark a CoBEVT model whose metric (AP) is directly
        comparable with ``corabench`` and ``lgcpbench``, and give FuseBEVT a
        second modality to prove it is modality-agnostic.

    Inputs
    ------
    grid          cpbench GridSpec; drives both voxelisation and the warp
    max_cav       fixed agent-axis extent (CoBEVT: 5)
    encoder_out_channels  BEV channels leaving PointPillars (paper: 256)
    fuse_dim      FuseBEVT working width; defaults to encoder_out_channels.
                  The paper gives the LiDAR feature map as 176x48x256 but
                  states the FuseBEVT config is identical to the camera
                  track's (dim 128), which cannot both be true. Defaulting to
                  the encoder width avoids an unstated projection; set it
                  explicitly to reproduce the camera-track width.
    fuse_depth, fuse_window, fuse_dim_head, fuse_mlp_dim, fuse_dropout
                  FuseBEVT hyperparameters
    use_local, use_global, pool   passed through to FuseBEVT (ablations, A11)
    num_anchors, num_classes      detection head geometry

    Outputs
    -------
    ``{"cls": (B, A*n_cls, H, W), "reg": (B, A*7, H, W),
       "fused": (B, C, H, W), "agent_mask": (B, L, H, W)}``

    The mask is returned, not just used, because "how many collaborators
    actually contributed at this pixel" is needed to interpret any robustness
    number computed from the boxes.

    Shapes
    ------
    batch["features"]      (P, max_points, 10)
    batch["coords"]        (P, 3) -- [flat agent index, row, col]
    batch["num_points"]    (P,)
    batch["record_len"]    (B,) agents per sample, summing to N_total
    batch["T_agent_to_ego"] (B, max_cav, 4, 4)

    Example
    -------
    >>> import torch
    >>> from cpbench.data import GridSpec
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),          # 64x64 pillars, 32x32 features
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> model = CoBEVTLidar(spec, max_cav=2, encoder_out_channels=32,
    ...                     fuse_depth=1, fuse_window=8, fuse_dim_head=8)
    >>> batch = {
    ...     "features": torch.randn(6, 4, 10),
    ...     "coords": torch.tensor([[0, 1, 1], [0, 2, 2], [1, 3, 3],
    ...                             [1, 4, 4], [1, 5, 5], [0, 6, 6]]),
    ...     "num_points": torch.full((6,), 4),
    ...     "record_len": [2],
    ...     "T_agent_to_ego": torch.eye(4).expand(1, 2, 4, 4).contiguous(),
    ... }
    >>> out = model(batch)
    >>> out["cls"].shape, out["reg"].shape
    (torch.Size([1, 2, 32, 32]), torch.Size([1, 14, 32, 32]))
    >>> out["fused"].shape, out["agent_mask"].shape
    (torch.Size([1, 32, 32, 32]), torch.Size([1, 2, 32, 32]))
    """

    def __init__(self, grid: GridSpec, max_cav: int = 5,
                 encoder_out_channels: int = 256,
                 fuse_dim: Optional[int] = None, fuse_depth: int = 3,
                 fuse_window: int = 8, fuse_dim_head: int = 32,
                 fuse_mlp_dim: Optional[int] = None,
                 fuse_dropout: float = 0.0, use_local: bool = True,
                 use_global: bool = True, pool: str = "mean",
                 num_anchors: int = 2, num_classes: int = 1,
                 block_strides: Sequence[int] = (2, 2, 2)) -> None:
        super().__init__()
        self.grid = grid
        self.max_cav = int(max_cav)
        self.block_strides = tuple(int(s) for s in block_strides)
        fuse_dim = int(fuse_dim if fuse_dim is not None else encoder_out_channels)
        fuse_mlp_dim = int(fuse_mlp_dim if fuse_mlp_dim is not None
                           else 2 * fuse_dim)

        # Before any submodule is built. Otherwise an inner module raises
        # first, naming its own parameter (`dim_head`) rather than the config
        # key the user actually set (`fuse_dim_head`) -- which is the whole
        # reason this validation exists.
        self._validate_geometry(fuse_window, fuse_dim, fuse_dim_head)

        self.encoder = PointPillarEncoder(grid_hw=grid.grid_hw,
                                          block_strides=self.block_strides,
                                          out_channels=encoder_out_channels)
        # A projection only when the widths genuinely differ, so the default
        # path has no extra parameters the paper does not describe.
        self.project = (nn.Conv2d(encoder_out_channels, fuse_dim, 1, bias=False)
                        if fuse_dim != encoder_out_channels else nn.Identity())
        self.sttf = SpatialTransform.from_grid_spec(grid)
        self.fuse = FuseBEVT(dim=fuse_dim, mlp_dim=fuse_mlp_dim,
                             agent_size=self.max_cav, window_size=fuse_window,
                             dim_head=fuse_dim_head, dropout=fuse_dropout,
                             depth=fuse_depth, use_local=use_local,
                             use_global=use_global, pool=pool)
        self.head = DetectionHead(in_channels=fuse_dim, num_anchors=num_anchors,
                                  num_classes=num_classes)

    # -- eager validation ---------------------------------------------------

    def _validate_geometry(self, window: int, dim: int, dim_head: int) -> None:
        """Fail at construction, not twenty minutes into a cluster job.

        The pillar-grid and backbone-stride checks are delegated to
        ``cpbench.models.validate_backbone_geometry``; see it for why each
        matters. The FuseBEVT window check below is CoBEVT's own, because no
        other package has a window.
        """
        validate_backbone_geometry(self.grid, self.block_strides)

        height, width = self.grid.feature_hw
        if height % window or width % window:
            raise ValueError(
                f"BEV feature grid {height}x{width} does not divide by the "
                f"FuseBEVT window {window}. Adjust grid.point_range, "
                "grid.voxel_size, grid.downsample or fuse_window so that "
                "both dimensions are multiples of the window.")
        if dim % dim_head:
            raise ValueError(
                f"fuse_dim {dim} is not divisible by fuse_dim_head {dim_head}; "
                "CoBEVT derives the head count as dim // dim_head")

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, Any],
                taps: Optional[TapProtocol] = None) -> Dict[str, torch.Tensor]:
        record_len = [int(n) for n in batch["record_len"]]
        n_agents_total = sum(record_len)

        emit(taps, batch["features"], module="CoBEVTLidar",
             location="input/points")

        features = self.encoder(batch["features"], batch["coords"],
                                batch["num_points"], n_agents_total, taps=taps)
        features = self.project(features)

        padded, agent_mask = regroup(features, record_len, self.max_cav,
                                     taps=taps)
        emit(taps, agent_mask, module="CoBEVTLidar", location="input/agent_mask")

        transforms = batch["T_agent_to_ego"].to(padded.dtype)
        emit(taps, transforms, module="CoBEVTLidar", location="input/poses")
        warped, valid = self.sttf(padded, transforms, taps=taps)

        # An agent contributes at a pixel only if it exists AND its own map
        # covered that location after warping. Conflating the two would let
        # zero-padding outside a collaborator's range read as a measurement.
        fuse_mask = agent_mask[:, :, None, None] & valid

        fused = self.fuse(warped, mask=fuse_mask, taps=taps)
        out = self.head(fused, taps=taps, branch="cobevt")
        out["fused"] = fused
        out["agent_mask"] = fuse_mask
        return out

    def extra_repr(self) -> str:
        return f"max_cav={self.max_cav}, feature_hw={self.grid.feature_hw}"
