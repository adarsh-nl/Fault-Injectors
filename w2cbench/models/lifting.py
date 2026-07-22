"""
lifting.py
----------
Image features -> a BEV feature map, by depth-distribution splatting (A13).

The problem, and why the camera track needs a module the LiDAR track does not
---------------------------------------------------------------------------
A pixel does not say how far away it is. LiDAR arrives as 3-D points and drops
straight into grid cells; an image has to have depth *supplied* before any of
it can be placed on a ground plane. That single missing quantity is the whole
difference between the two encoders, and everything downstream of
``encoder/bev_features`` is identical.

The paper's only statement on the matter is that camera input is "warped from
front-view to BEV", and the released repository contains no camera model at
all (A13/A14). So this is our construction, and it is the literal reading:
predict a distribution over discrete depths, smear each pixel's feature along
its ray weighted by that distribution, and sum whatever lands in each BEV cell
(Lift-Splat-Shoot, Philion & Fidler 2020).

Why *this* lift rather than a cross-attention one
-------------------------------------------------
A cross-attention lift -- CVT, BEVFormer, CoBEVT's SinBEVT -- would likely
score better on clean data. It has no explicit depth tensor, which is exactly
the problem for a fault benchmark: ``lift/depth_distribution`` is where an
image-domain fault becomes a *geometric* error, and it is observable only
because depth is predicted rather than learned implicitly.

Fog is the case that matters. It does not merely blur features; it produces a
**confident** distribution centred on the wrong bin, so the feature is splatted
into the wrong BEV cell, and the spatial confidence map computed downstream
then reports high confidence at a location holding nothing. Where2comm
faithfully transmits it. With an attention lift that failure is diffused across
attention weights and only its consequence is visible.

Memory
------
The frustum tensor is ``(N_cam, D, Z, h, w)`` and is the camera track's cost
centre: at 4 cameras, 5 agents, D=256, 41 depth bins and a 10x26 feature map it
is roughly 218 MB in fp32. ``depth_bins`` is the first knob to turn when memory
is tight, which is why it is in config rather than derived.

The splat uses ``index_add_`` rather than the cumulative-sum trick of the
original implementation. That trick existed to avoid materialising per-point
gradients on 2020 hardware; ``index_add_`` is differentiable, an order of
magnitude simpler to read, and the frustum tensor is materialised either way.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cpbench.data import GridSpec
from cpbench.observation import TapProtocol, emit

logger = logging.getLogger(__name__)


class BEVLifting(nn.Module, ABC):
    """Contract for turning image features into a BEV map.

    Purpose
        Keep the lift swappable. A5-style ablations aside, the choice of lift
        is the single largest assumption in the camera track (A13), so it sits
        behind an interface rather than being inlined into the encoder.

    Inputs (to :meth:`forward`)
    ---------------------------
    features    ``(B*L, M, C, h, w)`` image features, one set per camera.
    intrinsics  ``(B*L, M, 3, 3)``
    extrinsics  ``(B*L, M, 4, 4)`` -- camera-to-agent.

    Outputs
    -------
    ``(B*L, D, H, W)`` in the agent's own frame. **Not** the ego frame: the
    per-agent warp happens later, in fusion, which is what makes this
    intermediate rather than early fusion and what leaves a pose error unable
    to touch selection.
    """

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """D, the BEV width this lift produces."""

    @abstractmethod
    def forward(self, features: torch.Tensor, intrinsics: torch.Tensor,
                extrinsics: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        ...


class DepthDistributionHead(nn.Module):
    """Per-pixel context features and a distribution over discrete depths.

    One convolution produces both, split on the channel axis, exactly as the
    reference implementation does: the two are predicted from the same
    features and separating them into two convolutions would add parameters
    without adding information.

    Shapes
    ------
    in   ``(N, C, h, w)``
    out  ``(context (N, D, h, w), depth (N, Z, h, w))`` with `depth` a softmax
         over the Z axis.

    Example
    -------
    >>> import torch
    >>> head = DepthDistributionHead(in_channels=8, out_channels=4, depth_bins=6)
    >>> context, depth = head(torch.randn(2, 8, 5, 5))
    >>> context.shape, depth.shape
    (torch.Size([2, 4, 5, 5]), torch.Size([2, 6, 5, 5]))
    >>> bool(torch.allclose(depth.sum(1), torch.ones(2, 5, 5), atol=1e-6))
    True
    """

    def __init__(self, in_channels: int, out_channels: int,
                 depth_bins: int) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.depth_bins = int(depth_bins)
        self.project = nn.Conv2d(in_channels, out_channels + depth_bins, 1)

    def forward(self, features: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = self.project(features)
        depth_logits = combined[:, :self.depth_bins]
        context = combined[:, self.depth_bins:]
        emit(taps, context, module="DepthSplatLifting",
             location="lift/image_features")
        emit(taps, depth_logits, module="DepthDistributionHead",
             location="lift/depth_logits")
        depth = depth_logits.softmax(dim=1)
        emit(taps, depth, module="DepthDistributionHead",
             location="lift/depth_distribution")
        return context, depth


class FrustumSplat(nn.Module):
    """Place frustum features onto the BEV grid by summation.

    Purpose
        The geometric half of the lift: build the camera frustum in agent
        coordinates, find each frustum cell's BEV index, and accumulate.

    Inputs
    ------
    grid        ``cpbench.data.GridSpec`` -- the SAME object the LiDAR track
                uses, so both encoders land on an identical BEV grid. That is
                not tidiness: everything downstream is shared, and a camera
                lift onto a differently-sized grid would fail deep inside
                attention with a broadcast error.
    depth_bins  ``(min, max, step)`` metres.
    image_size  ``(H_img, W_img)`` the intrinsics were calibrated at.

    Shapes
    ------
    frustum    ``(N_cam, D, Z, h, w)``
    returns    ``(N_agent, D, H, W)``
    """

    def __init__(self, grid: GridSpec, depth_bins: Sequence[float],
                 image_size: Tuple[int, int]) -> None:
        super().__init__()
        low, high, step = (float(v) for v in depth_bins)
        if step <= 0 or high <= low:
            raise ValueError(
                f"depth_bins must be (min, max, step) with min < max and "
                f"step > 0, got {tuple(depth_bins)}")
        self.grid = grid
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.register_buffer(
            "depths", torch.arange(low, high, step, dtype=torch.float32))
        self.n_depths = int(self.depths.numel())

    def frustum_points(self, intrinsics: torch.Tensor,
                       extrinsics: torch.Tensor, feature_hw: Tuple[int, int],
                       taps: Optional[TapProtocol] = None) -> torch.Tensor:
        """Frustum cell centres in the agent frame.

        ``(N_cam, Z, h, w, 3)``. This is the tensor a calibration fault acts
        through: ``K`` and ``E`` enter here and nowhere else, so a perturbed
        focal length or a rotated mount displaces every splatted feature
        without touching the image content at all.
        """
        n_cam = intrinsics.shape[0]
        height, width = feature_hw
        device, dtype = intrinsics.device, intrinsics.dtype
        stride_v = self.image_size[0] / height
        stride_u = self.image_size[1] / width

        # Feature-cell centres, expressed in image pixels.
        us = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride_u
        vs = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride_v
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
        depths = self.depths.to(device=device, dtype=dtype)

        # Homogeneous image points scaled by depth: K^-1 @ [u*d, v*d, d].
        ones = torch.ones_like(grid_u)
        pixels = torch.stack((grid_u, grid_v, ones), dim=-1)      # (h, w, 3)
        rays = pixels[None] * depths[:, None, None, None]         # (Z, h, w, 3)

        k_inv = torch.linalg.inv(intrinsics)                      # (N, 3, 3)
        camera = torch.einsum("nij,zhwj->nzhwi", k_inv, rays)
        rotation = extrinsics[:, :3, :3]
        translation = extrinsics[:, :3, 3]
        agent = (torch.einsum("nij,nzhwj->nzhwi", rotation, camera)
                 + translation[:, None, None, None, :])
        emit(taps, agent.reshape(n_cam, -1, 3), module="FrustumSplat",
             location="lift/frustum_points")
        return agent

    def forward(self, frustum: torch.Tensor, points: torch.Tensor,
                n_agents: int, cameras: int,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        """Sum-pool frustum features into their BEV cells."""
        emit(taps, frustum, module="FrustumSplat", location="lift/frustum")
        n_cam, channels = frustum.shape[0], frustum.shape[1]
        height, width = self.grid.feature_hw
        stride_x, stride_y = self.grid.feature_stride_m
        x_min, y_min = self.grid.point_range[0], self.grid.point_range[1]

        col = ((points[..., 0] - x_min) / stride_x).long()
        row = ((points[..., 1] - y_min) / stride_y).long()
        inside = ((col >= 0) & (col < width) & (row >= 0) & (row < height))

        # Agent index per camera, so every camera of one agent accumulates
        # onto that agent's canvas and no other.
        agent_of_camera = (torch.arange(n_cam, device=frustum.device)
                           // max(cameras, 1))
        agent = agent_of_camera[:, None, None, None].expand_as(col)

        flat_index = ((agent * height + row) * width + col)
        values = frustum.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        flat_index = flat_index.reshape(-1)
        keep = inside.reshape(-1)

        canvas = frustum.new_zeros((n_agents * height * width, channels))
        canvas.index_add_(0, flat_index[keep], values[keep])
        out = canvas.reshape(n_agents, height, width, channels).permute(
            0, 3, 1, 2).contiguous()
        emit(taps, out, module="DepthSplatLifting", location="lift/splatted")
        return out


class DepthSplatLifting(BEVLifting):
    """Lift-Splat-Shoot style lifting (A13).

    Purpose
        Turn a camera feature pyramid into a per-agent BEV map, with the depth
        estimate left explicit and observable.

    Inputs
    ------
    in_channels  image feature width entering the lift.
    out_channels D, matching the LiDAR encoder so everything downstream is
                 shared.
    grid         the shared ``GridSpec``.
    depth_bins   ``(min, max, step)`` metres (A15 default: 4 to 45 by 1).
    image_size   ``(H, W)`` the intrinsics were calibrated at (A15: 160x416).

    Outputs
    -------
    ``(B*L, D, H, W)`` in each agent's own frame.

    Example
    -------
    >>> import torch
    >>> from cpbench.data import GridSpec
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> lift = DepthSplatLifting(in_channels=16, out_channels=8, grid=spec,
    ...                          depth_bins=(4.0, 20.0, 4.0),
    ...                          image_size=(64, 64))
    >>> features = torch.randn(2, 3, 16, 4, 4)          # 2 agents, 3 cameras
    >>> K = torch.eye(3).expand(2, 3, 3, 3).contiguous()
    >>> E = torch.eye(4).expand(2, 3, 4, 4).contiguous()
    >>> lift(features, K, E).shape
    torch.Size([2, 8, 32, 32])
    """

    def __init__(self, in_channels: int, out_channels: int, grid: GridSpec,
                 depth_bins: Sequence[float] = (4.0, 45.0, 1.0),
                 image_size: Tuple[int, int] = (160, 416)) -> None:
        super().__init__()
        self._out_channels = int(out_channels)
        self.splat = FrustumSplat(grid, depth_bins, image_size)
        self.head = DepthDistributionHead(in_channels, out_channels,
                                          self.splat.n_depths)
        logger.info("DepthSplatLifting(D=%d, Z=%d, bev=%s, image=%s)",
                    out_channels, self.splat.n_depths, grid.feature_hw,
                    image_size)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, features: torch.Tensor, intrinsics: torch.Tensor,
                extrinsics: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        if features.dim() != 5:
            raise ValueError(
                f"expected features (N_agent, M, C, h, w), got "
                f"{tuple(features.shape)}")
        n_agents, cameras = features.shape[0], features.shape[1]
        flat = features.flatten(0, 1)                       # (N_cam, C, h, w)
        flat_k = intrinsics.flatten(0, 1).to(flat.dtype)
        flat_e = extrinsics.flatten(0, 1).to(flat.dtype)

        context, depth = self.head(flat, taps=taps)
        # The outer product IS the lift: every pixel's feature is smeared
        # along its ray in proportion to how likely each depth is.
        frustum = context.unsqueeze(2) * depth.unsqueeze(1)
        points = self.splat.frustum_points(
            flat_k, flat_e, feature_hw=flat.shape[-2:], taps=taps)
        return self.splat(frustum, points, n_agents, cameras, taps=taps)
