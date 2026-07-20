"""
fusion.py
---------
Per-model area-restricted fusion for OpenCOOD backbones.

The problem this solves
    LGCP fuses at a LEADER, over a GROUP, restricted to ONE AREA. OpenCOOD's
    three models each fuse over ALL agents on the FULL BEV map, and each does
    it with a different signature:

        Where2comm  fuse_modules[i](x)                  x: (V, C, h, w)
        CoBEVT      fusion_net(regrouped, com_mask)     (B, L, h, w, C)
        CoAlign     fusion_net[i](x, record_len, affine)

    So there is no single call that works for all three. Each gets a strategy
    that adapts the group's area-restricted feature stack to that model's
    fusion module, reusing the PRETRAINED weights unchanged.

Assumption B12 -- multi-scale fusion is applied at a single scale
    Where2comm and CoAlign fuse at several backbone resolutions and then
    decode the pyramid. LGCP restricts features to an area on the FINAL
    feature map, so reproducing multi-scale fusion faithfully would mean
    re-running the backbone once per area -- which destroys the "encode once
    per frame" discipline that keeps the pipeline affordable, and is not what
    the LGCP paper describes either (its Fig. 2 shows one encoder, one
    exchange, one decoder).

    We therefore use the fusion module whose channel count matches the
    encoder output -- the last one -- applied to the area-restricted final
    feature map. This is a real deviation from the original models' internals
    and is recorded as an assumption rather than buried: it may account for
    part of any gap between reproduced and published Table II numbers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FusionStrategy:
    """Adapt a group's area feature stack to one model's fusion module.

    Inputs   stack (V, C, h, w) -- ego/leader first, then members.
    Outputs  (C, h, w) -- the leader's fused feature.
    """

    name = "base"

    def __call__(self, stack: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class Where2commFusion(FusionStrategy):
    """Where2comm's per-pixel cross-agent attention.

    ``AttentionFusion.forward`` reshapes to (h*w, V, C), self-attends over the
    agent axis and keeps index 0. It is spatially agnostic, so it accepts an
    area-restricted stack unchanged -- no reshaping tricks required.

    The communication mask is deliberately NOT applied here. In Where2comm it
    decides WHICH cells to transmit; under LGCP that decision has already been
    made upstream by area assignment (Algorithm 1) and the paper's own design
    replaces per-cell selection with per-area selection. Applying both would
    double-gate the features.
    """

    name = "where2comm"

    def __init__(self, model: nn.Module) -> None:
        fusion_net = getattr(model, "fusion_net", None)
        if fusion_net is None:
            raise AttributeError(
                "expected a Where2comm model with a `fusion_net` attribute; "
                f"got {type(model).__name__}"
            )
        modules = getattr(fusion_net, "fuse_modules", None)
        if modules is None:
            raise AttributeError(
                "expected `fusion_net.fuse_modules`; the OpenCOOD Where2comm "
                "fusion module layout has changed"
            )
        # multi_scale=True (the shipped default) gives an nn.ModuleList over
        # backbone levels; the last matches the encoder output width (B12).
        if isinstance(modules, nn.ModuleList):
            if len(modules) == 0:
                raise ValueError("fusion_net.fuse_modules is an empty ModuleList")
            self.module = modules[-1]
            logger.info(
                "Where2comm multi-scale fusion: using level %d of %d "
                "(assumption B12 -- single-scale on the area-restricted map)",
                len(modules) - 1, len(modules),
            )
        else:
            self.module = modules

    def __call__(self, stack: torch.Tensor) -> torch.Tensor:
        return self.module(stack)


class CoBEVTFusion(FusionStrategy):
    """CoBEVT's fused axial attention (SwapFusionEncoder).

    ``SwapFusionEncoder.forward(x, mask)`` expects channel-LAST
    ``(B, L, H, W, C)`` and a broadcastable mask, and returns ``(B, C, H, W)``.
    A group becomes one batch element with L = group size, and the mask is all
    ones because every member of a group is, by construction, present.
    """

    name = "cobevt"

    def __init__(self, model: nn.Module) -> None:
        fusion_net = getattr(model, "fusion_net", None)
        if fusion_net is None:
            raise AttributeError(
                "expected a CoBEVT model with a `fusion_net` (SwapFusionEncoder); "
                f"got {type(model).__name__}"
            )
        self.module = fusion_net

    def __call__(self, stack: torch.Tensor) -> torch.Tensor:
        v, c, h, w = stack.shape
        # (V, C, h, w) -> (1, V, h, w, C)
        x = stack.permute(0, 2, 3, 1).unsqueeze(0)
        mask = torch.ones(1, v, h, w, 1, dtype=x.dtype, device=x.device)
        fused = self.module(x, mask)
        if fused.dim() == 4:            # (1, C, h, w)
            return fused[0]
        return fused                     # already (C, h, w)


class CoAlignFusion(FusionStrategy):
    """CoAlign's attention-with-warp fusion.

    ``Att_w_Warp.forward(xx, record_len, normalized_affine_matrix)`` warps
    collaborators into the ego frame before attending. Under LGCP the features
    are ALREADY in the ego frame -- the dataset transforms point clouds with
    the (possibly corrupted) shared poses before voxelisation -- so the affine
    matrices passed here are identity.

    That is not a shortcut that hides pose error: the corruption has already
    been applied upstream, in metres, on the point cloud. Warping again with
    a corrupted matrix would apply it twice.

    Note this also means CoAlign's pose-graph correction is NOT reproduced --
    OpenCOOD's port omits it too, stating so in its own header comment.
    """

    name = "coalign"

    def __init__(self, model: nn.Module) -> None:
        fusion_net = getattr(model, "fusion_net", None)
        if fusion_net is None:
            raise AttributeError(
                "expected a CoAlign model with a `fusion_net`; "
                f"got {type(model).__name__}"
            )
        if isinstance(fusion_net, nn.ModuleList):
            if len(fusion_net) == 0:
                raise ValueError("CoAlign fusion_net is an empty ModuleList")
            self.module = fusion_net[-1]      # B12
        else:
            self.module = fusion_net

    def __call__(self, stack: torch.Tensor) -> torch.Tensor:
        v = stack.shape[0]
        record_len = torch.tensor([v], dtype=torch.long, device=stack.device)
        affine = torch.eye(2, 3, dtype=stack.dtype, device=stack.device)
        affine = affine.view(1, 1, 2, 3).repeat(1, v, 1, 1)
        fused = self.module(stack, record_len, affine)
        if fused.dim() == 4:
            return fused[0]
        return fused


_STRATEGIES = {
    "point_pillar_where2comm": Where2commFusion,
    "point_pillar_cobevt": CoBEVTFusion,
    "point_pillar_coalign": CoAlignFusion,
}


def build_fusion_strategy(core_method: str, model: nn.Module) -> FusionStrategy:
    """Pick the fusion strategy for an OpenCOOD ``core_method``.

    Example
    -------
    >>> sorted(available_core_methods())
    ['point_pillar_coalign', 'point_pillar_cobevt', 'point_pillar_where2comm']
    """
    try:
        cls = _STRATEGIES[core_method]
    except KeyError:
        raise KeyError(
            f"no LGCP fusion strategy for OpenCOOD core_method {core_method!r}. "
            f"Known: {sorted(_STRATEGIES)}. Each model fuses with a different "
            f"signature, so a new one needs its own FusionStrategy."
        ) from None
    return cls(model)


def available_core_methods() -> Sequence[str]:
    """OpenCOOD models LGCP can orchestrate."""
    return sorted(_STRATEGIES)
