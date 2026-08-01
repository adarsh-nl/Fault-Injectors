"""
encoder.py
----------
PointPillars encoder (Lang et al. 2019), the local encoder E of CoRA.

Three independent nn.Modules composed by `PointPillarEncoder`:

    PillarVFE            (P, T, 10) decorated points -> (P, C_vfe) pillar feats
    PointPillarScatter   pillar feats + coords      -> (N, C_vfe, H0, W0)
    BEVBackbone          dense canvas               -> (N, C, H, W) = F_j

No spconv dependency -- pillars only need dense scatter ops. Every stage
emits its output at a named observation point (read-only taps).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from ..observation.taps import TapProtocol, emit


def validate_backbone_geometry(grid, block_strides: Sequence[int]) -> None:
    """Check a ``GridSpec`` against the BEV backbone that will consume it.

    Purpose
        Catch, at construction, two geometry mistakes whose symptoms appear
        far from their cause -- one loudly and late, one silently and never.

    Inputs
    ------
    grid           anything with ``grid_hw``, ``feature_hw`` and
                   ``downsample`` (i.e. a ``cpbench.data.GridSpec``).
                   Duck-typed rather than imported, so ``cpbench.models``
                   stays independent of ``cpbench.data``.
    block_strides  the strides :class:`BEVBackbone` will be built with.

    Raises
    ------
    ValueError naming both sides of whichever disagreement it found.

    The two checks
    --------------
    *Divisibility.* The backbone downsamples by the product of
    ``block_strides`` and upsamples each level back. An indivisible pillar
    grid produces levels off by one pixel, and it surfaces as
    ``Sizes of tensors must match ... Expected size 25 but got size 26`` from
    inside a ``torch.cat`` -- with nothing to suggest the real cause is a
    point range in a YAML file.

    *Declared versus actual stride.* This is the quiet one.
    ``GridSpec.feature_hw`` is ``grid_hw // downsample``, and it is what the
    anchor generator, the spatial warp and the box decoder are all sized
    from. But the resolution the backbone ACTUALLY produces is
    ``grid_hw // block_strides[0]`` -- every pyramid level is upsampled back
    to the *first* level, not to the input. When the two disagree nothing
    raises on its own: the encoder returns a feature map of one size while
    every consumer was built for another, and the first symptom is either a
    shape error several modules downstream or a silently mismatched anchor
    grid that lowers AP without ever failing.

    Example
    -------
    >>> from cpbench.data import GridSpec
    >>> spec = GridSpec(voxel_size=(0.4, 0.4),
    ...                 point_range=(-51.2, -51.2, -3.0, 51.2, 51.2, 1.0))
    >>> validate_backbone_geometry(spec, (2, 2, 2))          # consistent
    >>> validate_backbone_geometry(spec, (4, 2, 2))
    Traceback (most recent call last):
    ValueError: grid.downsample=2 disagrees with block_strides[0]=4...
    """
    stride_product = 1
    for stride in block_strides:
        stride_product *= int(stride)
    pillar_h, pillar_w = grid.grid_hw
    if pillar_h % stride_product or pillar_w % stride_product:
        raise ValueError(
            f"pillar grid {pillar_h}x{pillar_w} does not divide by the "
            f"backbone stride product {stride_product} "
            f"(block_strides={tuple(block_strides)}). Choose a point_range "
            "and voxel_size whose ratio is a multiple of it.")

    first = int(block_strides[0])
    if int(grid.downsample) != first:
        raise ValueError(
            f"grid.downsample={grid.downsample} disagrees with "
            f"block_strides[0]={first}. The BEV backbone upsamples every "
            "pyramid level back to the FIRST level's resolution, so it "
            f"produces grid_hw // {first} = {pillar_h // first} cells, while "
            f"GridSpec.feature_hw reports {grid.feature_hw} and the anchors, "
            "the spatial warp and the box decoder are all sized from that. "
            "Set them equal.")


class PillarVFE(nn.Module):
    """Pillar feature encoder (single PFN layer, per the paper's backbone).

    Purpose  turn each pillar's decorated points into one feature vector.
    Inputs   features (P, T, 10), num_points (P,) valid counts.
    Output   (P, C_vfe) pillar features (max-pooled over valid points).
    Shapes   T = max points per pillar (padded rows are masked out).

    Example
    -------
    >>> vfe = PillarVFE(in_channels=10, out_channels=64)
    >>> vfe(torch.rand(10, 32, 10), torch.full((10,), 32)).shape
    torch.Size([10, 64])
    """

    def __init__(self, in_channels: int = 10, out_channels: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.out_channels = out_channels

    def forward(self, features: torch.Tensor, num_points: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        p, t, _ = features.shape
        if p == 0:
            return features.new_zeros((0, self.out_channels))
        x = self.linear(features)                            # (P, T, C)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(x)
        mask = (torch.arange(t, device=features.device)[None, :]
                < num_points[:, None])                       # (P, T)
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        x = x.max(dim=1).values                              # (P, C)
        x = torch.nan_to_num(x, neginf=0.0)                  # all-empty pillars
        emit(taps, x, module="PillarVFE", location="encoder/pillar_features")
        return x


class PointPillarScatter(nn.Module):
    """Scatter pillar features back onto the dense BEV canvas.

    Inputs   pillar_features (P, C), coords (P, 3) [agent, row, col],
             n_agents (int).
    Output   (N, C, H0, W0) dense canvas (zeros where no pillar).
    """

    def __init__(self, grid_hw: Tuple[int, int]) -> None:
        super().__init__()
        self.h0, self.w0 = grid_hw

    def forward(self, pillar_features: torch.Tensor, coords: torch.Tensor,
                n_agents: int,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        c = pillar_features.shape[1] if pillar_features.numel() else 0
        canvas = pillar_features.new_zeros((n_agents, c, self.h0, self.w0)) \
            if c else pillar_features.new_zeros((n_agents, 64, self.h0, self.w0))
        if pillar_features.numel():
            canvas[coords[:, 0], :, coords[:, 1], coords[:, 2]] = pillar_features
        emit(taps, canvas, module="PointPillarScatter",
             location="encoder/scatter_bev")
        return canvas


class _ConvBlock(nn.Module):
    """Strided conv stack: one downsampling conv + `layers` refining convs."""

    def __init__(self, cin: int, cout: int, stride: int, layers: int) -> None:
        super().__init__()
        mods = [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True)]
        for _ in range(layers):
            mods += [nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                     nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
                     nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BEVBackbone(nn.Module):
    """Multi-scale 2-D backbone with deconv fusion (SECOND/PointPillars style).

    Purpose  produce the per-agent BEV feature map F_j the whole framework
             works on.
    Inputs   (N, C_vfe, H0, W0) canvas.
    Output   (N, out_channels, H0/downsample, W0/downsample).

    Config   block_channels/strides/layers define the pyramid; each level is
             upsampled back to the first level's resolution and concatenated,
             then a 1x1 conv maps to `out_channels`.
    """

    def __init__(self, in_channels: int = 64,
                 block_channels: Sequence[int] = (64, 128, 256),
                 block_strides: Sequence[int] = (2, 2, 2),
                 block_layers: Sequence[int] = (3, 5, 5),
                 upsample_channels: int = 128,
                 out_channels: int = 256) -> None:
        super().__init__()
        assert len(block_channels) == len(block_strides) == len(block_layers)
        self.blocks = nn.ModuleList()
        cin = in_channels
        for cout, stride, layers in zip(block_channels, block_strides,
                                        block_layers):
            self.blocks.append(_ConvBlock(cin, cout, stride, layers))
            cin = cout
        # each level upsamples back to the first level's resolution
        self.deconvs = nn.ModuleList()
        cum = 1
        first_cum = block_strides[0]
        for cout, stride in zip(block_channels, block_strides):
            cum *= stride
            up = cum // first_cum
            if up > 1:
                self.deconvs.append(nn.Sequential(
                    nn.ConvTranspose2d(cout, upsample_channels, up, stride=up,
                                       bias=False),
                    nn.BatchNorm2d(upsample_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)))
            else:
                self.deconvs.append(nn.Sequential(
                    nn.Conv2d(cout, upsample_channels, 1, bias=False),
                    nn.BatchNorm2d(upsample_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)))
        self.out = nn.Conv2d(upsample_channels * len(self.blocks),
                             out_channels, 1, bias=False)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        ups = []
        for block, deconv in zip(self.blocks, self.deconvs):
            x = block(x)
            ups.append(deconv(x))
        feat = self.out(torch.cat(ups, dim=1))
        emit(taps, feat, module="BEVBackbone", location="encoder/bev_features")
        return feat


class PointPillarEncoder(nn.Module):
    """The full local encoder E: pillars -> F_j.

    Inputs   the collated batch dict (features, coords, num_points) and the
             number of agent rows to scatter into.
    Output   (N_agents, C, H, W) per-agent BEV features.

    Example
    -------
    >>> enc = PointPillarEncoder(grid_hw=(100, 100))       # doctest: +SKIP
    >>> f = enc(batch["features"], batch["coords"],
    ...         batch["num_points"], n_agents=4)           # doctest: +SKIP
    """

    def __init__(self, grid_hw: Tuple[int, int], in_channels: int = 10,
                 vfe_channels: int = 64,
                 block_channels: Sequence[int] = (64, 128, 256),
                 block_strides: Sequence[int] = (2, 2, 2),
                 block_layers: Sequence[int] = (3, 5, 5),
                 upsample_channels: int = 128,
                 out_channels: int = 256) -> None:
        super().__init__()
        self.vfe = PillarVFE(in_channels, vfe_channels)
        self.scatter = PointPillarScatter(grid_hw)
        self.backbone = BEVBackbone(vfe_channels, block_channels,
                                    block_strides, block_layers,
                                    upsample_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, features: torch.Tensor, coords: torch.Tensor,
                num_points: torch.Tensor, n_agents: int,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        pillar_feats = self.vfe(features, num_points, taps=taps)
        canvas = self.scatter(pillar_feats, coords, n_agents, taps=taps)
        return self.backbone(canvas, taps=taps)
