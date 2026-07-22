"""
encoder_camera.py
-----------------
The camera track's Stage-1 encoder: ResNet pyramid, then a BEV lift.

    images -> ResnetEncoder -> feature pyramid -> BEVLifting -> F^(0)

That is the whole camera track. Everything after ``encoder/bev_features`` --
the spatial confidence generator, selection, packing, the warp, fusion, the
decoder -- is byte-for-byte the code the LiDAR track runs, which is the claim
``test_track_parity`` exists to check rather than assert.

What this track is not
----------------------
A reproduction. The released Where2comm repository has no camera model (its
README lists DAIR-V2X as the only supported dataset, with OPV2V and V2X-Sim
unchecked), and the paper says only that camera input is warped from front-view
to BEV. So the lift is our construction (A13) and camera numbers are internal
comparisons -- clean versus faulted under an identical model -- rather than
anything checkable against a published table (A14).

That is enough for what the benchmark needs. The fault surface a camera track
opens is the half of ``src/fault_injectors`` the LiDAR track cannot reach at
all: fog, snow, darkness, brightness, lens occlusion and calibration error.

Only the finest pyramid level is lifted
---------------------------------------
``ResnetEncoder`` returns several scales because CoBEVT's SinBEVT consumes them
coarse-to-fine, lifting each onto a progressively smaller BEV grid. A splatting
lift has no such structure: it places points, and points from a coarse map are
simply fewer and blurrier. Lifting one level and letting the backbone's own
receptive field carry the multi-scale information is the reference LSS design,
and pretending otherwise would add parameters the paper does not describe. The
level is configurable, so the ablation stays reachable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from cpbench.data import GridSpec
from cpbench.models.image import ResnetEncoder
from cpbench.observation import TapProtocol, emit

from .encoder import ObservationEncoder
from .lifting import BEVLifting, DepthSplatLifting

logger = logging.getLogger(__name__)


class CameraEncoder(ObservationEncoder):
    """Multi-camera images -> one BEV feature map per agent.

    Purpose
        Produce ``F^(0)`` from cameras, in the same shape and on the same grid
        as the LiDAR encoder, so nothing downstream needs to know which ran.

    Inputs
    ------
    grid         the shared ``cpbench.data.GridSpec``.
    out_channels D. Must match whatever the rest of the model was sized for.
    backbone_arch, pretrained, id_pick   passed to ``ResnetEncoder``.
    lift_level   which pyramid level to lift, as an index into ``id_pick``.
    depth_bins   ``(min, max, step)`` metres (A13).
    image_size   ``(H, W)`` the intrinsics were calibrated at (A15).
    lifting      an explicit :class:`BEVLifting` to use instead of building
                 the default -- the seam a cross-attention lift would enter
                 through.

    Batch keys read
    ---------------
    ``images`` (B, L, M, 3, H, W) or (B, L, M, H, W, 3), ``intrinsics``
    (B, L, M, 3, 3), ``extrinsics`` (B, L, M, 4, 4), ``record_len`` (B,).

    Outputs
    -------
    ``(N, D, H, W)`` with ``N = sum(record_len)``.

    Example
    -------
    >>> import torch
    >>> from cpbench.data import GridSpec
    >>> spec = GridSpec(voxel_size=(1.6, 1.6),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> enc = CameraEncoder(spec, out_channels=16, backbone_arch="resnet18",
    ...                     pretrained=False, id_pick=[1],
    ...                     depth_bins=(4.0, 20.0, 4.0), image_size=(64, 64))
    >>> batch = {"images": torch.rand(1, 2, 3, 3, 64, 64),
    ...          "intrinsics": torch.eye(3).expand(1, 2, 3, 3, 3).contiguous(),
    ...          "extrinsics": torch.eye(4).expand(1, 2, 3, 4, 4).contiguous(),
    ...          "record_len": [2]}
    >>> enc.eval()(batch).shape
    torch.Size([2, 16, 16, 16])
    """

    def __init__(self, grid: GridSpec, out_channels: int = 256,
                 backbone_arch: str = "resnet34", pretrained: bool = True,
                 id_pick: Sequence[int] = (1, 2, 3), lift_level: int = 0,
                 depth_bins: Sequence[float] = (4.0, 45.0, 1.0),
                 image_size: Tuple[int, int] = (160, 416),
                 lifting: Optional[BEVLifting] = None) -> None:
        super().__init__(out_channels=out_channels, feature_hw=grid.feature_hw)
        self.backbone = ResnetEncoder(arch=backbone_arch, pretrained=pretrained,
                                      id_pick=id_pick)
        if not 0 <= int(lift_level) < len(self.backbone.out_channels):
            raise ValueError(
                f"lift_level {lift_level} is out of range for id_pick="
                f"{list(id_pick)}, which yields {len(self.backbone.out_channels)}"
                " pyramid levels")
        self.lift_level = int(lift_level)
        self.lifting = lifting or DepthSplatLifting(
            in_channels=self.backbone.out_channels[self.lift_level],
            out_channels=out_channels, grid=grid, depth_bins=depth_bins,
            image_size=image_size)
        if self.lifting.out_channels != out_channels:
            raise ValueError(
                f"lifting produces {self.lifting.out_channels} channels but "
                f"the encoder declares out_channels={out_channels}")
        logger.info("CameraEncoder(%s, lift_level=%d, D=%d, bev=%s)",
                    backbone_arch, self.lift_level, out_channels,
                    grid.feature_hw)

    def forward(self, batch: Dict[str, Any],
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        """Encode every agent's cameras into its BEV feature map.

        Agents are flattened out of the batch axis before the backbone runs,
        so a ragged ``record_len`` costs nothing and no padded agent is ever
        given to the confidence generator (see
        :meth:`ObservationEncoder.total_agents`).
        """
        n_agents = self.total_agents(batch)
        images, intrinsics, extrinsics = self._agent_views(batch, n_agents)

        pyramid = self.backbone(images, taps=taps)
        features = pyramid[self.lift_level]              # (N, M, C, h, w)
        bev = self.lifting(features, intrinsics, extrinsics, taps=taps)
        emit(taps, bev, module="CameraEncoder",
             location="encoder/bev_features")
        return self.validate_output(bev)

    @staticmethod
    def _agent_views(batch: Dict[str, Any], n_agents: int) -> tuple:
        """Flatten ``(B, L, M, ...)`` to ``(N, M, ...)``, keeping real agents.

        The batch pads the agent axis to ``max_cav``; ``record_len`` says how
        many of each sample's slots are real. Slicing here rather than masking
        later keeps a padded slot from ever reaching the lift, where it would
        splat an all-zero image into a genuine BEV map.
        """
        record_len = [int(n) for n in batch["record_len"]]
        images = batch["images"]
        intrinsics, extrinsics = batch["intrinsics"], batch["extrinsics"]
        if images.dim() < 5:
            raise ValueError(
                f"expected images (B, L, M, ...), got {tuple(images.shape)}")

        picked_images, picked_k, picked_e = [], [], []
        for index, count in enumerate(record_len):
            picked_images.append(images[index, :count])
            picked_k.append(intrinsics[index, :count])
            picked_e.append(extrinsics[index, :count])
        merged = torch.cat(picked_images)
        if merged.shape[0] != n_agents:      # pragma: no cover - guarded above
            raise ValueError(
                f"record_len sums to {n_agents} but only {merged.shape[0]} "
                "agent rows are present in 'images'")
        return merged, torch.cat(picked_k), torch.cat(picked_e)
