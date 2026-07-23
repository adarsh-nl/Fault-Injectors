"""
encoder.py
----------
The V2X-ViT encoder: RTE -> STTF -> depth x (HMSA -> MSwin -> FFN).

This is the paper's figure 2 assembled from the modules around it. Nothing
here computes; it *sequences*, and every seam between two stages is an
emitted, named tensor -- which is the property the whole package is built
around: a fault injector or an analysis tap can be inserted between any two
stages without touching this file.

Order of operations (assumption A3): the reference applies the delay
encoding BEFORE the spatial warp -- the embedding is added in each agent's
own frame and warped along with the features. Uniform per agent, the warp
only redistributes it spatially, but the roi-invalid border differs between
the two orders, so the reference order is kept rather than reasoned away.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit

from v2xvitbench.fusion.geometry import SpatialTransform
from v2xvitbench.fusion.hmsa import HGTCavAttention
from v2xvitbench.fusion.mlp import FeedForward
from v2xvitbench.fusion.mswin import PyramidWindowAttention
from v2xvitbench.fusion.prior import DelayPositionalEncoding


class V2XFusionBlock(nn.Module):
    """One fusion layer's attention pair: HMSA then MSwin, each residual.

    Purpose
        Alternate inter-agent attention (who to trust) with per-agent
        spatial attention (what the neighbourhood says), ``num_blocks``
        times. The released config runs one pair per layer; with more, the
        pairs share the layer's tap prefix and emissions repeat per pair.

    Inputs are built by the caller (:class:`V2XTEncoder`); this module owns
    only the pair sequencing and the residual adds.

    Shapes
    ------
    x (B, L, H, W, C), mask (B, L, H, W), types (B, L) -> (B, L, H, W, C)

    Example
    -------
    >>> import torch
    >>> block = V2XFusionBlock(
    ...     hmsa_factory=lambda: HGTCavAttention(dim=16, heads=2, dim_head=8),
    ...     mswin_factory=lambda: PyramidWindowAttention(
    ...         dim=16, heads=(2,), dim_heads=(8,), window_sizes=(2,)))
    >>> x = torch.randn(1, 2, 4, 4, 16)
    >>> mask = torch.ones(1, 2, 4, 4, dtype=torch.bool)
    >>> block(x, mask, torch.tensor([[0, 1]])).shape
    torch.Size([1, 2, 4, 4, 16])
    """

    def __init__(self, hmsa_factory: Callable[[], HGTCavAttention],
                 mswin_factory: Callable[[], PyramidWindowAttention],
                 num_blocks: int = 1) -> None:
        super().__init__()
        self.hmsa = nn.ModuleList([hmsa_factory() for _ in range(num_blocks)])
        self.mswin = nn.ModuleList([mswin_factory() for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                types: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fusion/l0") -> torch.Tensor:
        emit(taps, x, module="V2XFusionBlock",
             location=f"{location_prefix}/input")
        for hmsa, mswin in zip(self.hmsa, self.mswin):
            x = hmsa(x, mask, types, taps, f"{location_prefix}/hmsa") + x
            x = mswin(x, taps, f"{location_prefix}/mswin") + x
        return x


class V2XTEncoder(nn.Module):
    """The full fusion stack between the per-agent encoder and the ego slice.

    Purpose
        Sequence RTE, STTF and the fusion layers; own the mask logic that
        joins agent existence with warp validity. Submodules arrive
        constructed (dependency injection, the package convention): this
        class decides *when*, its parts decide *how*.

    Inputs
    ------
    depth          number of fusion layers
    block_factory  builds one :class:`V2XFusionBlock` per layer
    ffn_factory    builds one :class:`FeedForward` per layer
    rte            the delay encoding, or None to disable (config
                   ``fusion.rte.enabled``)
    sttf           the spatial warp (built from the FUSION GridSpec)
    use_roi_mask   AND warp validity into the attention mask (reference:
                   true). Off, out-of-coverage zeros are attended like data.

    Shapes
    ------
    x               (B, L, C, H, W) channels-first, from regroup
    T_agent_to_ego  (B, L, 4, 4)
    agent_mask      (B, L) bool
    dts             (B, L) delays in frames
    types           (B, L) long in {0, 1}
    ->              ``(fused, mask)``: (B, L, H, W, C) channels-LAST fused
                    maps for all agents (the model slices the ego and
                    re-permutes), and the (B, L, H, W) attention mask that
                    was used -- agent existence joined with warp validity.
                    Returned, not just used, because "how many collaborators
                    actually contributed at this pixel" is needed to
                    interpret any robustness number computed downstream.
    """

    def __init__(self, depth: int,
                 block_factory: Callable[[], V2XFusionBlock],
                 ffn_factory: Callable[[], FeedForward],
                 rte: Optional[DelayPositionalEncoding],
                 sttf: SpatialTransform,
                 use_roi_mask: bool = True) -> None:
        super().__init__()
        self.depth = int(depth)
        self.blocks = nn.ModuleList([block_factory()
                                     for _ in range(self.depth)])
        self.ffns = nn.ModuleList([ffn_factory() for _ in range(self.depth)])
        self.rte = rte
        self.sttf = sttf
        self.use_roi_mask = bool(use_roi_mask)

    def forward(self, x: torch.Tensor, T_agent_to_ego: torch.Tensor,
                agent_mask: torch.Tensor, dts: torch.Tensor,
                types: torch.Tensor,
                taps: Optional[TapProtocol] = None
                ) -> "tuple[torch.Tensor, torch.Tensor]":
        if self.rte is not None:
            x = self.rte(x, dts, taps)

        warped, valid = self.sttf(x, T_agent_to_ego, taps)

        mask = agent_mask[:, :, None, None].expand_as(valid)
        if self.use_roi_mask:
            mask = mask & valid

        x = warped.permute(0, 1, 3, 4, 2)          # channels-last
        for i, (block, ffn) in enumerate(zip(self.blocks, self.ffns)):
            prefix = f"fusion/l{i}"
            x = block(x, mask, types, taps, prefix)
            x = ffn(x, taps, f"{prefix}/ffn") + x
            emit(taps, x, module="V2XTEncoder", location=f"{prefix}/output")
        return x, mask

    def extra_repr(self) -> str:
        return f"depth={self.depth}, use_roi_mask={self.use_roi_mask}"
