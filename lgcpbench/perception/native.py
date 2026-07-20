"""
native.py
---------
A CPU-only, torch-only reference backbone implementing
``CollabPerceptionModel``.

Purpose
    Two review decisions pull against each other: "OpenCOOD dependency,
    faithful repro" and "synthetic first, CPU unit tests". OpenCOOD is
    hard-locked to Python 3.7 (numba==0.49.0), needs spconv and CUDA, and
    cannot run in a unit test. This module is how both decisions hold:

        * NativeReferenceBackbone  -> every unit test, all synthetic
          development, Python 3.9+, CPU, seconds not hours.
        * OpenCOODBackbone         -> Table II reproduction, its own py3.7
          environment on the HPC.

    Both satisfy the same protocol, so LGCP's control plane -- the part this
    project actually contributes -- is identical under either.

    This backbone is NOT claimed to reproduce the paper's accuracy numbers.
    It exists to make the orchestration testable and to give the fault
    injector a full tensor surface without a 20-minute GPU round trip.

Fidelity where it matters
    ``confidence`` follows design doc derivation D1 exactly: the confidence
    map is the SHARED detection classification head applied to the pre-fusion
    per-agent feature map, reduced by sigmoid then max over anchors. That is
    what Where2comm does (verified in OpenCOOD ``point_pillar_where2comm.py``
    and ``fuse_modules/where2comm_fuse.py``), and it is what LGCP Eq. 1 cites
    as ``f_gen``. Sharing the head -- rather than adding a separate confidence
    network -- is the load-bearing detail.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cpbench.models.encoder import PointPillarEncoder
from cpbench.models.heads import DetectionHead
from cpbench.observation.taps import TapProtocol, emit

from .protocol import AgentInputs


class PerPixelAttentionFusion(nn.Module):
    """Scaled dot-product attention across agents, independently per BEV cell.

    Purpose
        The fusion shape Where2comm uses (``AttentionFusion`` in OpenCOOD):
        treat each of the h*w cells as a token position, attend across the
        agent dimension, and keep the ego row. Reproducing that shape here
        means area-restricted fusion behaves the same way under the native
        backbone as under the real one.

    Inputs
    ------
    x : (V, C, h, w) -- agent 0 is the ego/leader.

    Outputs
    -------
    (C, h, w) -- the ego row after attending over all agents.

    Shapes inside
    -------------
    tokens          (h*w, V, C)
    q, k, v         (h*w, V, C)
    scores          (h*w, V, V)   pre-softmax
    attn            (h*w, V, V)   post-softmax

    Fault injection
    ---------------
    Every one of q, k, v, scores and attn is emitted at a named location, and
    each is a separate statement -- never ``softmax(q @ k.T)`` inline -- so an
    injector can be inserted between any two without touching this file.

    Note
    ----
    OpenCOOD's ``ScaledDotProductAttention`` applies no learned projections.
    We include optional ones (default on) because the brief explicitly
    requires ``query_projection`` as an injection point. Set
    ``use_projections=False`` to match OpenCOOD's shape exactly.
    """

    def __init__(self, channels: int, use_projections: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.use_projections = use_projections
        self.scale = 1.0 / math.sqrt(channels)
        if use_projections:
            self.q_proj = nn.Linear(channels, channels, bias=False)
            self.k_proj = nn.Linear(channels, channels, bias=False)
            self.v_proj = nn.Linear(channels, channels, bias=False)
        else:
            self.q_proj = self.k_proj = self.v_proj = nn.Identity()

    def forward(
        self, x: torch.Tensor, taps: Optional[TapProtocol] = None
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (V, C, h, w), got {tuple(x.shape)}")
        v_agents, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")

        tokens = x.view(v_agents, c, h * w).permute(2, 0, 1)  # (h*w, V, C)
        emit(taps, tokens, module="PerPixelAttentionFusion",
             location="lgcp/perception/attn_tokens")

        q = self.q_proj(tokens)
        emit(taps, q, module="PerPixelAttentionFusion",
             location="lgcp/perception/attn_query")

        k = self.k_proj(tokens)
        emit(taps, k, module="PerPixelAttentionFusion",
             location="lgcp/perception/attn_key")

        v = self.v_proj(tokens)
        emit(taps, v, module="PerPixelAttentionFusion",
             location="lgcp/perception/attn_value")

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        emit(taps, scores, module="PerPixelAttentionFusion",
             location="lgcp/perception/attn_scores")

        attn = torch.softmax(scores, dim=-1)
        emit(taps, attn, module="PerPixelAttentionFusion",
             location="lgcp/perception/attn_softmax")

        fused_tokens = torch.matmul(attn, v)
        emit(taps, fused_tokens, module="PerPixelAttentionFusion",
             location="lgcp/perception/attn_out")

        # keep the ego/leader row -- OpenCOOD's `[0]`
        out = fused_tokens.permute(1, 2, 0).view(v_agents, c, h, w)[0]
        return out


class NativeReferenceBackbone(nn.Module):
    """CPU reference implementation of ``CollabPerceptionModel``.

    Purpose
        Make LGCP's orchestration testable without OpenCOOD. Reuses
        corabench's PointPillar encoder and detection head rather than
        reimplementing them.

    Inputs
    ------
    grid_hw        (H0, W0) dense pillar canvas, from GridSpec.grid_hw.
    feature_hw     (H, W) BEV feature map, from GridSpec.feature_hw. Must
                   equal grid_hw // downsample; checked at construction.
    channels       C of the BEV feature map.
    num_anchors    anchors per cell (2 in the OPV2V convention).
    downsample     encoder stride. 4 matches the OPV2V/Where2comm setting
                   (voxel 0.4 m x stride 4 = a 1.6 m feature cell), which is
                   what makes a 10 x 6 m area span ~6.25 x 3.75 cells.

    Outputs (per protocol)
    ----------------------
    encode      AgentInputs      -> (V, C, H, W)
    confidence  (V, C, H, W)     -> (V, 1, H, W) in [0, 1]
    fuse        (C,h,w) + [...]  -> (C, h, w)
    detect      (C, h, w)        -> {"cls": (A, h, w), "reg": (A*7, h, w)}

    Example
    -------
    >>> m = NativeReferenceBackbone(grid_hw=(64, 64), feature_hw=(16, 16),
    ...                             channels=32, downsample=4)
    >>> feats = torch.zeros(3, 32, 16, 16)
    >>> tuple(m.confidence(feats).shape)
    (3, 1, 16, 16)
    >>> tuple(m.fuse(feats[0, :, :4, :6], [feats[1, :, :4, :6]]).shape)
    (32, 4, 6)
    """

    def __init__(
        self,
        grid_hw: Tuple[int, int],
        feature_hw: Tuple[int, int],
        channels: int = 256,
        num_anchors: int = 2,
        num_classes: int = 1,
        in_channels: int = 9,
        downsample: int = 4,
        use_projections: bool = True,
    ) -> None:
        super().__init__()
        self.feature_channels = int(channels)
        self.feature_hw = (int(feature_hw[0]), int(feature_hw[1]))
        self.num_anchors = int(num_anchors)
        self.downsample = int(downsample)

        # BEVBackbone upsamples every pyramid level back to the first level's
        # resolution, so the encoder's effective stride is block_strides[0].
        # Deriving it from `downsample` -- rather than leaving the caller to
        # discover the coupling -- is what lets the check below be exact.
        expected_hw = (grid_hw[0] // self.downsample, grid_hw[1] // self.downsample)
        if expected_hw != self.feature_hw:
            raise ValueError(
                f"feature_hw {self.feature_hw} is inconsistent with "
                f"grid_hw {tuple(grid_hw)} at downsample={self.downsample}; "
                f"expected {expected_hw}"
            )

        # The pyramid has three levels with strides (downsample, 2, 2), and
        # each level is deconvolved back to level 1's resolution before being
        # concatenated. That only lines up if the canvas divides evenly by the
        # PRODUCT of the strides -- otherwise floor-division at a deeper level
        # makes the upsampled maps differ by a pixel and the concat fails deep
        # inside the backbone with an opaque size error. Check it here.
        self._total_stride = self.downsample * 4
        bad = [
            f"{name}={value}"
            for name, value in (("H0", grid_hw[0]), ("W0", grid_hw[1]))
            if value % self._total_stride != 0
        ]
        if bad:
            raise ValueError(
                f"grid_hw {tuple(grid_hw)} must be divisible by the backbone's total "
                f"stride {self._total_stride} (= downsample {self.downsample} x 4 for "
                f"the 3-level pyramid); offending: {', '.join(bad)}. "
                f"Choose a point_range whose extent divides evenly by "
                f"voxel_size * {self._total_stride}."
            )

        self.encoder = PointPillarEncoder(
            grid_hw=grid_hw,
            in_channels=in_channels,
            out_channels=channels,
            block_strides=(self.downsample, 2, 2),
        )
        # D1: ONE head, shared between confidence (pre-fusion, per agent) and
        # detection (post-fusion, per area). Where2comm does exactly this.
        self.head = DetectionHead(
            channels, num_anchors=num_anchors, num_classes=num_classes
        )
        self.fusion = PerPixelAttentionFusion(channels, use_projections=use_projections)

    # ------------------------------------------------------------------ #
    # protocol
    # ------------------------------------------------------------------ #

    def encode(
        self, inputs: AgentInputs, *, taps: Optional[TapProtocol] = None
    ) -> torch.Tensor:
        """Per-CAV BEV features, encoded once per frame for all CAVs.

        Encoding is frame-level, never area-level: a CAV's features are
        computed once and then sliced per area. Re-encoding per area would
        multiply cost by the number of areas the CAV participates in.
        """
        features = self.encoder(
            inputs.features,
            inputs.coords,
            inputs.num_points,
            n_agents=inputs.n_agents,
            taps=taps,
        )
        emit(taps, features, module="NativeReferenceBackbone",
             location="lgcp/perception/bev_features")
        if tuple(features.shape[-2:]) != self.feature_hw:
            raise RuntimeError(
                f"encoder produced {tuple(features.shape[-2:])} but backbone is "
                f"configured for feature_hw={self.feature_hw}"
            )
        return features

    def confidence(
        self, features: torch.Tensor, *, taps: Optional[TapProtocol] = None
    ) -> torch.Tensor:
        """Paper Eq. 1's ``f_gen`` -- see design doc derivation D1.

        Shared cls_head -> sigmoid -> max over anchors. Deliberately written
        as four separate statements so an injector can sit between any two.

        Inputs  (V, C, H, W).
        Outputs (V, 1, H, W) in [0, 1].
        """
        if features.dim() != 4:
            raise ValueError(f"expected (V, C, H, W), got {tuple(features.shape)}")

        logits = self.head.cls_head(features)
        emit(taps, logits, module="NativeReferenceBackbone",
             location="lgcp/perception/psm_single")

        probs = torch.sigmoid(logits)
        emit(taps, probs, module="NativeReferenceBackbone",
             location="lgcp/perception/psm_sigmoid")

        conf, _ = probs.max(dim=1, keepdim=True)
        emit(taps, conf, module="NativeReferenceBackbone",
             location="lgcp/perception/confidence_map")
        return conf

    def fuse(
        self,
        ego: torch.Tensor,
        collab: Sequence[torch.Tensor],
        *,
        taps: Optional[TapProtocol] = None,
    ) -> torch.Tensor:
        """Leader-side fusion of one group over one area.

        Inputs  ego (C, h, w); collab sequence of (C, h, w).
        Outputs (C, h, w).

        A group of size 1 (the leader alone, no members admitted by Eq. 8)
        returns the ego feature unchanged -- no attention over a single
        agent, which would be an identity with extra cost.
        """
        if ego.dim() != 3:
            raise ValueError(f"ego must be (C, h, w), got {tuple(ego.shape)}")
        for i, f in enumerate(collab):
            if f.shape != ego.shape:
                raise ValueError(
                    f"collab[{i}] has shape {tuple(f.shape)}, expected {tuple(ego.shape)}"
                )

        emit(taps, ego, module="NativeReferenceBackbone",
             location="lgcp/perception/fuse_ego_in", n_collab=len(collab))

        if not collab:
            emit(taps, ego, module="NativeReferenceBackbone",
                 location="lgcp/perception/fused_feature", n_collab=0)
            return ego

        stacked = torch.stack([ego, *collab], dim=0)
        emit(taps, stacked, module="NativeReferenceBackbone",
             location="lgcp/perception/fuse_stack")

        fused = self.fusion(stacked, taps=taps)
        emit(taps, fused, module="NativeReferenceBackbone",
             location="lgcp/perception/fused_feature", n_collab=len(collab))
        return fused

    def detect(
        self, fused: torch.Tensor, *, taps: Optional[TapProtocol] = None
    ) -> Dict[str, torch.Tensor]:
        """Detection maps for one area.

        Inputs  (C, h, w).
        Outputs {"cls": (A, h, w) logits, "reg": (A*7, h, w) deltas}.

        The head is applied with a batch dimension of 1 and squeezed, so the
        shared corabench ``DetectionHead`` is reused unmodified.
        """
        if fused.dim() != 3:
            raise ValueError(f"expected (C, h, w), got {tuple(fused.shape)}")
        out = self.head(fused.unsqueeze(0), taps=taps, branch="lgcp_area")
        return {"cls": out["cls"][0], "reg": out["reg"][0]}
