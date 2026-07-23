"""
v2xvit.py
---------
The V2X-ViT model: PointPillars -> shrink -> compress -> regroup ->
V2X-ViT encoder (RTE, STTF, HMSA/MSwin/FFN) -> ego slice -> detection head.

Everything except the fusion stack is borrowed from ``cpbench``: the pillar
encoder, the detection head, the anchor and box conventions. That is
deliberate. It means a V2X-ViT-vs-CoBEVT robustness comparison differs in
the fusion block and nothing else -- same voxelisation, same head, same AP
definition, same tap names. A reimplemented encoder would put an
uncontrolled variable between the two papers' numbers.

    batch --> PointPillarEncoder --> (N_total, 384, H0/2, W0/2)
          --> ShrinkConv          --> (N_total, 256, H, W)     stride 4 total
          --> NaiveCompressor     --> (N_total, 256, H, W)
          --> regroup(record_len) --> (B, L, 256, H, W) + agent mask
          --> V2XTEncoder         --> (B, L, H, W, 256) + roi-joined mask
          --> ego slice           --> (B, 256, H, W)
          --> DetectionHead       --> cls, reg

The two-GridSpec bookkeeping (assumption A2)
--------------------------------------------
V2X-ViT detects at stride 4: backbone stride 2 x shrink stride 2. cpbench's
``validate_backbone_geometry`` ties a GridSpec's ``downsample`` to what the
BACKBONE alone produces, so the model works with two specs sharing voxel
size and range:

    encoder_grid   downsample = block_strides[0]   -- validated against the
                   backbone, sizes the pillar canvas
    self.grid      downsample = that * shrink_stride -- the FUSION spec the
                   caller supplies; anchors, the box decoder and the STTF
                   warp are all sized from it

``_validate_geometry`` asserts the identity between them at construction,
because getting it wrong is silent: everything runs at the wrong geometry
and only AP notices.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from cpbench.data import GridSpec
from cpbench.models import (DetectionHead, PointPillarEncoder,
                            validate_backbone_geometry)
from cpbench.observation import TapProtocol, emit

from v2xvitbench.fusion.encoder import V2XFusionBlock, V2XTEncoder
from v2xvitbench.fusion.geometry import SpatialTransform, regroup
from v2xvitbench.fusion.hmsa import HGTCavAttention
from v2xvitbench.fusion.mlp import FeedForward
from v2xvitbench.fusion.mswin import PyramidWindowAttention
from v2xvitbench.fusion.prior import DelayPositionalEncoding, PriorEncoder


class V2XViT(nn.Module):
    """V2X-ViT: heterogeneous multi-agent detection with a fusion transformer.

    Purpose
        The paper's full pipeline as a composition of independently tested
        modules, every seam a named observation point, so faults inject at
        the data plane (points, poses) or the metadata plane (delay, type)
        without touching model code.

    Inputs
    ------
    grid            the FUSION GridSpec: ``downsample`` must equal
                    ``block_strides[0] * shrink_stride`` (reference: 2*2=4)
    max_cav         fixed agent-axis extent (reference: 5)
    vfe_channels, block_channels, block_strides, block_layers,
    upsample_channels, encoder_out_channels
                    PointPillars geometry (reference: 64, [64,128,256],
                    [2,2,2], [3,5,8], 128, 384)
    shrink_channels, shrink_stride, shrink_kernel
                    the shrink header (reference: 256, 2, 3); the fusion
                    width IS ``shrink_channels``
    compression_factor  NaiveCompressor factor (reference: 0 = off)
    depth, num_blocks   fusion layers and HMSA/MSwin pairs per layer (3, 1)
    hmsa_heads, hmsa_dim_head, num_types, num_relations, dropout
                    HMSA geometry (reference: 8, 32, 2, 4, 0.3)
    window_sizes, mswin_heads, mswin_dim_heads, relative_pos_embedding,
    fusion_method   MSwin geometry (reference: [4,8,16], [16,8,4],
                    [16,32,64], true, split_attn)
    mlp_dim         feed-forward hidden width (reference: 256)
    use_rte, rte_ratio, max_delay   delay encoding (true, 2, 100)
    use_roi_mask    join warp validity into attention (reference: true)
    num_anchors, num_classes        detection head geometry (2, 1)

    Outputs
    -------
    ``{"cls": (B, A*n_cls, H, W), "reg": (B, A*7, H, W),
       "fused": (B, C, H, W), "agent_mask": (B, L, H, W)}``

    Shapes
    ------
    batch["features"]       (P, max_points, 9)
    batch["coords"]         (P, 3) -- [flat agent index, row, col]
    batch["num_points"]     (P,)
    batch["record_len"]     (B,) agents per sample, summing to N_total
    batch["T_agent_to_ego"] (B, max_cav, 4, 4)
    batch["time_delay"]     (B, max_cav) reported delay, frames
    batch["infra"]          (B, max_cav) 0 = vehicle, 1 = infrastructure
    batch["velocity"]       (B, max_cav) m/s (optional; defaults to 0)

    Example
    -------
    >>> import torch
    >>> from cpbench.data import GridSpec
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),     # 64x64 pillars, 16x16 fused
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
    ...                 downsample=4)
    >>> model = V2XViT(spec, max_cav=2, encoder_out_channels=48,
    ...                shrink_channels=32, depth=1, hmsa_heads=2,
    ...                hmsa_dim_head=16, window_sizes=(2, 4),
    ...                mswin_heads=(2, 2), mswin_dim_heads=(16, 16),
    ...                mlp_dim=32, dropout=0.0)
    >>> _ = model.eval()
    >>> batch = {
    ...     "features": torch.randn(6, 4, 9),
    ...     "coords": torch.tensor([[0, 1, 1], [0, 2, 2], [1, 3, 3],
    ...                             [1, 4, 4], [1, 5, 5], [0, 6, 6]]),
    ...     "num_points": torch.full((6,), 4),
    ...     "record_len": [2],
    ...     "T_agent_to_ego": torch.eye(4).expand(1, 2, 4, 4).contiguous(),
    ...     "time_delay": torch.tensor([[0, 2]]),
    ...     "infra": torch.tensor([[0, 1]]),
    ...     "velocity": torch.tensor([[10.0, 0.0]]),
    ... }
    >>> out = model(batch)
    >>> out["cls"].shape, out["reg"].shape
    (torch.Size([1, 2, 16, 16]), torch.Size([1, 14, 16, 16]))
    >>> out["fused"].shape, out["agent_mask"].shape
    (torch.Size([1, 32, 16, 16]), torch.Size([1, 2, 16, 16]))
    """

    def __init__(self, grid: GridSpec, max_cav: int = 5,
                 vfe_channels: int = 64,
                 block_channels: Sequence[int] = (64, 128, 256),
                 block_strides: Sequence[int] = (2, 2, 2),
                 block_layers: Sequence[int] = (3, 5, 8),
                 upsample_channels: int = 128,
                 encoder_out_channels: int = 384,
                 shrink_channels: int = 256, shrink_stride: int = 2,
                 shrink_kernel: int = 3,
                 compression_factor: int = 0,
                 depth: int = 3, num_blocks: int = 1,
                 hmsa_heads: int = 8, hmsa_dim_head: int = 32,
                 num_types: int = 2, num_relations: int = 4,
                 dropout: float = 0.3,
                 window_sizes: Sequence[int] = (4, 8, 16),
                 mswin_heads: Sequence[int] = (16, 8, 4),
                 mswin_dim_heads: Sequence[int] = (16, 32, 64),
                 relative_pos_embedding: bool = True,
                 fusion_method: str = "split_attn",
                 mlp_dim: int = 256,
                 use_rte: bool = True, rte_ratio: int = 2,
                 max_delay: int = 100,
                 use_roi_mask: bool = True,
                 num_anchors: int = 2, num_classes: int = 1) -> None:
        super().__init__()
        # local imports keep the module namespace clean of factory helpers
        from v2xvitbench.models.compression import NaiveCompressor
        from v2xvitbench.models.shrink import ShrinkConv

        self.grid = grid
        self.max_cav = int(max_cav)
        self.block_strides = tuple(int(s) for s in block_strides)
        self.shrink_stride = int(shrink_stride)
        dim = int(shrink_channels)

        # Before any submodule is built. Otherwise an inner module raises
        # first, naming its own parameter (`window_size`) rather than the
        # config key the user actually set -- which is the whole reason this
        # validation exists.
        encoder_grid = self._validate_geometry(window_sizes)

        self.encoder = PointPillarEncoder(
            grid_hw=encoder_grid.grid_hw, vfe_channels=int(vfe_channels),
            block_channels=tuple(int(c) for c in block_channels),
            block_strides=self.block_strides,
            block_layers=tuple(int(n) for n in block_layers),
            upsample_channels=int(upsample_channels),
            out_channels=int(encoder_out_channels))
        self.shrink = ShrinkConv(int(encoder_out_channels), dim,
                                 kernel=int(shrink_kernel),
                                 stride=self.shrink_stride)
        self.compressor = NaiveCompressor(dim, int(compression_factor))
        self.prior = PriorEncoder()
        self.fusion = V2XTEncoder(
            depth=int(depth),
            block_factory=lambda: V2XFusionBlock(
                hmsa_factory=lambda: HGTCavAttention(
                    dim=dim, heads=int(hmsa_heads),
                    dim_head=int(hmsa_dim_head), num_types=int(num_types),
                    num_relations=int(num_relations), dropout=dropout),
                mswin_factory=lambda: PyramidWindowAttention(
                    dim=dim, heads=tuple(mswin_heads),
                    dim_heads=tuple(mswin_dim_heads),
                    window_sizes=tuple(window_sizes),
                    relative_pos_embedding=relative_pos_embedding,
                    fusion_method=fusion_method, dropout=dropout),
                num_blocks=int(num_blocks)),
            ffn_factory=lambda: FeedForward(dim=dim, mlp_dim=int(mlp_dim),
                                            dropout=dropout),
            rte=(DelayPositionalEncoding(dim, max_delay=int(max_delay),
                                         ratio=int(rte_ratio))
                 if use_rte else None),
            sttf=SpatialTransform.from_grid_spec(grid),
            use_roi_mask=use_roi_mask)
        self.head = DetectionHead(in_channels=dim, num_anchors=int(num_anchors),
                                  num_classes=int(num_classes))

    # -- eager validation ---------------------------------------------------

    def _validate_geometry(self, window_sizes: Sequence[int]) -> GridSpec:
        """Fail at construction, not twenty minutes into a cluster job.

        Builds the ENCODER GridSpec (same voxel size and range, downsample =
        first backbone stride), delegates the pillar-grid checks to
        ``cpbench.models.validate_backbone_geometry``, then asserts the
        dual-GridSpec identity and the MSwin window divisibility that no
        other package has.
        """
        encoder_grid = GridSpec(voxel_size=self.grid.voxel_size,
                                point_range=self.grid.point_range,
                                downsample=self.block_strides[0])
        validate_backbone_geometry(encoder_grid, self.block_strides)

        expected = self.block_strides[0] * self.shrink_stride
        if int(self.grid.downsample) != expected:
            raise ValueError(
                f"grid.downsample={self.grid.downsample} disagrees with "
                f"block_strides[0]={self.block_strides[0]} x "
                f"shrink_stride={self.shrink_stride} = {expected}. The "
                "fusion GridSpec sizes the anchors, the box decoder and the "
                "STTF warp, so it must describe the map the shrink header "
                "actually produces. Set dataset grid.downsample (or the "
                "shrink stride) so the identity holds.")

        height, width = self.grid.feature_hw
        for window in window_sizes:
            if height % int(window) or width % int(window):
                raise ValueError(
                    f"fused BEV grid {height}x{width} does not divide by the "
                    f"MSwin window {window} (window_sizes="
                    f"{tuple(window_sizes)}). Adjust grid.point_range, "
                    "grid.voxel_size or fusion.mswin.window_sizes so every "
                    "window tiles the grid exactly.")
        return encoder_grid

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, Any],
                taps: Optional[TapProtocol] = None) -> Dict[str, torch.Tensor]:
        record_len = [int(n) for n in batch["record_len"]]
        n_agents_total = sum(record_len)

        emit(taps, batch["features"], module="V2XViT", location="input/points")
        emit(taps, batch["coords"], module="V2XViT", location="input/coords")

        features = self.encoder(batch["features"], batch["coords"],
                                batch["num_points"], n_agents_total, taps=taps)
        features = self.shrink(features, taps=taps)
        features = self.compressor(features, taps=taps)

        padded, agent_mask = regroup(features, record_len, self.max_cav,
                                     taps=taps)
        emit(taps, agent_mask, module="V2XViT", location="input/agent_mask")

        transforms = batch["T_agent_to_ego"].to(padded.dtype)
        emit(taps, transforms, module="V2XViT", location="input/poses")

        dts = batch["time_delay"].long()
        types = batch["infra"].long()
        velocity = batch.get("velocity")
        if velocity is None:
            velocity = torch.zeros_like(dts, dtype=padded.dtype)
        emit(taps, dts, module="V2XViT", location="input/time_delay")
        emit(taps, types, module="V2XViT", location="input/agent_types")
        self.prior(velocity, dts, types, taps=taps)

        fused_all, roi_mask = self.fusion(padded, transforms, agent_mask,
                                          dts, types, taps=taps)

        ego = fused_all[:, 0].permute(0, 3, 1, 2).contiguous()
        emit(taps, ego, module="V2XViT", location="fusion/ego_features")

        out = self.head(ego, taps=taps, branch="v2xvit")
        out["fused"] = ego
        out["agent_mask"] = roi_mask
        return out

    def extra_repr(self) -> str:
        return (f"max_cav={self.max_cav}, fused_hw={self.grid.feature_hw}, "
                f"shrink_stride={self.shrink_stride}")
