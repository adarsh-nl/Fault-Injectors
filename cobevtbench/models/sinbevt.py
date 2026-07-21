"""
sinbevt.py
----------
SinBEVT: single-agent camera-to-BEV lifting.

A learned BEV query grid is refined by three cross-view blocks, each reading
a different scale of image features, halving the BEV resolution between
blocks, and finished with one dense self-attention pass::

    learned prior (128, 128, 128)
      -> block 0 with layer2 features (128ch, 64x64)   -> bottlenecks -> /2
      -> block 1 with layer3 features (256ch, 32x32)   -> bottlenecks -> /2
      -> block 2 with layer4 features (512ch, 16x16)   -> bottlenecks
      -> dense self-attention over the 32x32 map
      -> (B*L, 128, 32, 32)          <- this is what goes on the wire

Coarse-to-fine in *image* scale, fine-to-coarse in *BEV* scale. The first
block has the most BEV cells and the sharpest image features, so it does the
localisation; the last block has a 32x32 BEV grid attending to 16x16 image
features in a single window, which is global context.

The output is 32x32x128 -- about 524 KB uncompressed. That number is the
point of the whole architecture: it is what a vehicle can broadcast at
10 Hz, and it is what the compression ablation (0x to 64x) trades away.

Frame convention
----------------
Everything here is in **one agent's own ego frame**, not a shared world
frame. Each agent lifts independently and transmits; aligning the results is
FuseBEVT's problem, via the warp in ``fusion/geometry.py``. So the extrinsics
this module wants are ``T_cam_to_ego`` for that agent, and the BEV grids are
agent-centred.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from einops import rearrange
from torch import nn
from torchvision.models.resnet import Bottleneck

from cpbench.data import BEVGrid
from cpbench.observation import TapProtocol, emit

from ..attention.attention import ScaledDotProductAttention
from ..attention.fax_cross import FAXCrossAttentionBlock
from ..attention.qkv import FusedQKVProjection, merge_heads
from ..attention.rel_pos_bias import RelativePositionBias
from ..fusion.camera_embedding import CameraGeometryEmbedding


class BEVSelfAttention(nn.Module):
    """Dense self-attention over the whole BEV map, with a 2-D position bias.

    Purpose
        SinBEVT's terminal block. At 32x32 the map is small enough for full
        attention, so unlike FuseBEVT there is no windowing here -- every
        cell sees every other cell.

    Inputs
    ------
    dim, dim_head, dropout
    grid_hw   the BEV size this runs at; sets the relative position bias
              table, so it is fixed at construction

    Outputs
    -------
    Same shape as the input.

    Shapes
    ------
    x  (B, C, H, W)  ->  (B, C, H, W); internally (B, H*W, C) tokens

    Example
    -------
    >>> import torch
    >>> attn = BEVSelfAttention(dim=16, dim_head=8, grid_hw=(4, 4))
    >>> attn(torch.randn(2, 16, 4, 4)).shape
    torch.Size([2, 16, 4, 4])
    """

    def __init__(self, dim: int, dim_head: int, grid_hw: Tuple[int, int],
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.norm = nn.LayerNorm(dim)
        self.qkv = FusedQKVProjection(dim, dim_head, bias=False)
        self.rel_pos_bias = RelativePositionBias(self.grid_hw,
                                                 num_heads=self.qkv.num_heads)
        self.attend = ScaledDotProductAttention(dim_head)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, taps: Optional[TapProtocol] = None,
                location_prefix: str = "sinbevt/self_attn") -> torch.Tensor:
        height, width = x.shape[-2], x.shape[-1]
        if (height, width) != self.grid_hw:
            raise ValueError(
                f"BEVSelfAttention was built for a {self.grid_hw} grid but got "
                f"{(height, width)}; the relative position bias table has a "
                "fixed extent and cannot be resized per batch")

        tokens = rearrange(x, "b c h w -> b (h w) c")
        normed = self.norm(tokens)
        q, k, v = self.qkv(normed, taps=taps, location_prefix=location_prefix)
        bias = self.rel_pos_bias(taps=taps,
                                 location=f"{location_prefix}/rel_pos_bias")
        attended = self.attend(q, k, v, bias=bias, taps=taps,
                               location_prefix=location_prefix)
        delta = self.out_drop(self.to_out(merge_heads(attended)))
        tokens = tokens + delta
        return rearrange(tokens, "b (h w) c -> b c h w", h=height, w=width)


class _Downsample(nn.Module):
    """Halve the BEV resolution between cross-view blocks.

    ``PixelUnshuffle`` rather than a strided conv: it is lossless, folding
    the discarded spatial detail into channels instead of averaging it away.
    At this point in the network the BEV grid still holds the only geometric
    evidence the model has, so throwing a quarter of it away is worse than
    carrying it in channels.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_dim, in_dim // 4, 3, padding=1, bias=False)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.refine = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.unshuffle(x)
        return self.refine(x)


class SinBEVT(nn.Module):
    """Lift one agent's multi-camera images to a BEV feature map.

    Purpose
        CoBEVT's single-agent half (paper section 4.1). Also usable alone --
        that is the paper's nuScenes result (37.1 IoU) and its "No Fusion"
        row (37.7).

    Inputs
    ------
    dims            BEV width per block (CoBEVT: [128, 128, 128])
    feat_channels   image feature channels per block (CoBEVT: [128, 256, 512],
                    i.e. ResNet34 layer2/3/4 -- assumption A2)
    bev_size        BEV grid of the *first* block (CoBEVT: 128); halves after
                    each block
    bev_meters      metric extent of the BEV area (CoBEVT: 100 m)
    image_size      original image size, for scaling K correctly
    q_win_sizes     BEV query windows (CoBEVT: [16, 16, 32])
    feat_win_sizes  image key/value windows (CoBEVT: [8, 8, 16])
    heads, dim_head per block
    middle          ResNet bottlenecks between blocks (CoBEVT: [2, 2, 2])
    bev_embedding_flags  which blocks add the BEV positional embedding
                    (CoBEVT: [True, False, False])
    self_attn_dim_head, self_attn_dropout
    no_image_features, camera_reduce   ablation / assumption A6

    Outputs
    -------
    ``(B, dims[-1], bev_size / 2^(n-1), ...)`` -- for CoBEVT's settings,
    ``(B, 128, 32, 32)``.

    Shapes
    ------
    features      list of (B, M, feat_channels[i], h_i, w_i), coarse-to-fine
    K             (B, M, 3, 3)
    T_cam_to_ego  (B, M, 4, 4)

    Example
    -------
    >>> import torch
    >>> model = SinBEVT(dims=[16, 16], feat_channels=[8, 8], bev_size=16,
    ...                 bev_meters=40.0, image_size=(32, 32),
    ...                 q_win_sizes=[8, 8], feat_win_sizes=[4, 4],
    ...                 heads=[2, 2], dim_head=[8, 8], middle=[1, 1],
    ...                 bev_embedding_flags=[True, False],
    ...                 self_attn_dim_head=8)
    >>> feats = [torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 4, 4)]
    >>> K = torch.eye(3).expand(1, 4, 3, 3).contiguous()
    >>> T = torch.eye(4).expand(1, 4, 4, 4).contiguous()
    >>> model(feats, K, T).shape
    torch.Size([1, 16, 8, 8])
    """

    def __init__(self, dims: Sequence[int], feat_channels: Sequence[int],
                 bev_size: int, bev_meters: float,
                 image_size: Tuple[int, int], q_win_sizes: Sequence,
                 feat_win_sizes: Sequence, heads: Sequence[int],
                 dim_head: Sequence[int], middle: Sequence[int],
                 bev_embedding_flags: Sequence[bool],
                 self_attn_dim_head: int = 32, self_attn_dropout: float = 0.0,
                 dropout: float = 0.0, qkv_bias: bool = True,
                 no_image_features: bool = False,
                 camera_reduce: str = "mean", sigma: float = 1.0) -> None:
        super().__init__()
        n_blocks = len(dims)
        for name, seq in (("feat_channels", feat_channels),
                          ("q_win_sizes", q_win_sizes),
                          ("feat_win_sizes", feat_win_sizes),
                          ("heads", heads), ("dim_head", dim_head),
                          ("middle", middle),
                          ("bev_embedding_flags", bev_embedding_flags)):
            if len(seq) != n_blocks:
                raise ValueError(
                    f"{name} has {len(seq)} entries but dims has {n_blocks}; "
                    "every per-block setting must be given for every block")

        self.n_blocks = n_blocks
        self.bev_sizes = [int(bev_size) // (2 ** i) for i in range(n_blocks)]
        if self.bev_sizes[-1] < 1:
            raise ValueError(
                f"bev_size {bev_size} halves to {self.bev_sizes[-1]} by block "
                f"{n_blocks - 1}; increase bev_size or use fewer blocks")

        # The learned BEV prior: the query grid before any image is seen.
        self.bev_prior = nn.Parameter(
            sigma * torch.randn(dims[0], self.bev_sizes[0], self.bev_sizes[0]))

        self.geometries = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.bottlenecks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(n_blocks):
            grid = BEVGrid(height=self.bev_sizes[i], width=self.bev_sizes[i],
                           h_meters=bev_meters, w_meters=bev_meters)
            self.geometries.append(CameraGeometryEmbedding(
                dims[i], grid, image_size,
                with_bev_embedding=bool(bev_embedding_flags[i])))
            self.blocks.append(FAXCrossAttentionBlock(
                dim=dims[i], feat_channels=feat_channels[i],
                q_win_size=q_win_sizes[i], feat_win_size=feat_win_sizes[i],
                dim_head=dim_head[i], num_heads=heads[i], qkv_bias=qkv_bias,
                dropout=dropout,
                use_bev_embedding=bool(bev_embedding_flags[i]),
                no_image_features=no_image_features,
                camera_reduce=camera_reduce))
            self.bottlenecks.append(nn.Sequential(
                *[Bottleneck(dims[i], dims[i] // 4) for _ in range(middle[i])]))
            self.downsamples.append(
                _Downsample(dims[i], dims[i + 1]) if i < n_blocks - 1
                else nn.Identity())

        final = self.bev_sizes[-1]
        self.self_attn = BEVSelfAttention(dims[-1], self_attn_dim_head,
                                          (final, final), self_attn_dropout)
        self.out_dim = dims[-1]
        self.out_size = final

    def forward(self, features: Sequence[torch.Tensor], K: torch.Tensor,
                T_cam_to_ego: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        if len(features) != self.n_blocks:
            raise ValueError(
                f"got {len(features)} feature scales for {self.n_blocks} "
                "cross-view blocks; the image backbone must emit one map per "
                "block")

        batch = features[0].shape[0]
        emit(taps, self.bev_prior, module="BEVEmbedding",
             location="sinbevt/bev_prior")
        x = self.bev_prior[None].expand(batch, -1, -1, -1)

        for i in range(self.n_blocks):
            prefix = f"sinbevt/b{i}"
            feature_hw = (features[i].shape[-2], features[i].shape[-1])
            img_embed, bev_embed = self.geometries[i](
                K, T_cam_to_ego, feature_hw, taps=taps, location_prefix=prefix)
            x = self.blocks[i](x, features[i], img_embed, bev_embed,
                               taps=taps, location_prefix=prefix)
            x = self.bottlenecks[i](x)
            emit(taps, x, module="SinBEVT", location=f"{prefix}/bottleneck_out")
            x = self.downsamples[i](x)
            if i < self.n_blocks - 1:
                emit(taps, x, module="SinBEVT",
                     location=f"{prefix}/downsampled")

        x = self.self_attn(x, taps=taps)
        emit(taps, x, module="SinBEVT", location="sinbevt/output")
        return x

    def extra_repr(self) -> str:
        return (f"blocks={self.n_blocks}, bev_sizes={self.bev_sizes}, "
                f"out={self.out_dim}x{self.out_size}x{self.out_size}")
