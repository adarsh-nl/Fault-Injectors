"""
fusebevt.py
-----------
FuseBEVT: CoBEVT's multi-agent fusion transformer.

``depth`` FAX self-attention blocks over a stack of per-agent BEV maps, then
the agent axis is collapsed and projected::

    (B, L, C, H, W)  ->  depth x FAXSelfAttentionBlock  ->  (B, L, C, H, W)
                     ->  pool over L                    ->  (B, C, H, W)
                     ->  LayerNorm + Linear             ->  (B, C, H, W)

Modality-agnostic by construction
---------------------------------
Nothing here knows whether ``C`` channels came from cameras or from LiDAR
pillars. That is not incidental tidiness -- it is the contract that lets the
camera track and the LiDAR track share one implementation, and the same
contract that makes FuseBEVT a drop-in OpenCOOD fusion module (consume
``(N_total, C, H, W)`` plus ``record_len``, regroup, warp, fuse, return
``(B, C, H, W)``).

Attention fuses; the mean merges
--------------------------------
Note what the agent axis is collapsed by: a **plain mean**, not attention.
The attention's job is to make the per-agent maps mutually *consistent*
first; averaging consistent maps is then a reasonable merge. This ordering is
easy to miss when reading the architecture as "a fusion transformer".
"""

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange
from torch import nn

from cpbench.observation import TapProtocol, emit

from ..attention.fax_self import FAXSelfAttentionBlock

POOL_MEAN = "mean"
POOL_MASKED_MEAN = "masked_mean"


class FuseBEVT(nn.Module):
    """Fuse per-agent BEV feature maps into one ego-frame map.

    Purpose
        The multi-agent half of CoBEVT (paper section 4.2), shared verbatim
        by the camera and LiDAR tracks.

    Inputs
    ------
    dim          channel dim (CoBEVT: 128)
    mlp_dim      FAX feed-forward inner width (CoBEVT: 256)
    agent_size   fixed agent-axis extent == dataset ``max_cav`` (CoBEVT: 5)
    window_size  FAX window (CoBEVT: 8)
    dim_head     channels per head (CoBEVT: 32, giving 4 heads)
    dropout      CoBEVT: 0.1
    depth        number of FAX blocks (CoBEVT: 3)
    use_local / use_global   paper section 7.3 ablation switches
    pool         ``"mean"`` reproduces the reference. ``"masked_mean"``
                 divides by the number of *present* agents instead -- see
                 assumption A11 below.

    Outputs
    -------
    ``(B, dim, H, W)`` -- the fused ego BEV map.

    Shapes
    ------
    x       (B, L, C, H, W)   L must equal agent_size (pad to max_cav)
    mask    (B, L) or (B, L, H, W) bool; True = agent present here
    return  (B, C, H, W)

    Assumption A11 -- what the mean divides by
    ------------------------------------------
    The reference collapses the agent axis with an unweighted mean over all
    ``max_cav`` slots, including the zero-padded ones. An absent agent
    contributes no *keys* -- the mask sees to that, so it cannot reach a
    present agent's tokens -- but it still has query rows, those rows produce
    an attended output, and the plain mean averages that output in.

    The tempting reading is that this attenuates the fused feature by
    ``present / max_cav`` and that agent-drop results are therefore partly a
    scale artefact. That reading is wrong, and ``test_fusebevt.py`` pins why:
    the attenuation is real at ``fusebevt/pooled`` but the head LayerNorm
    immediately renormalises per position across channels, so it is gone by
    the output. What survives is a **direction** change -- the padding's
    attended output mixed into the fused feature -- not a magnitude one.

    ``pool="masked_mean"`` divides by the present-agent count instead, which
    makes the output provably independent of anything in an absent agent's
    slot. Default is ``"mean"``: reproducing the paper is the priority and
    the released weights were trained under it. Runs that need agent-drop
    degradation attributable purely to lost information should set
    ``"masked_mean"`` and say so; ``configs/faults/agent_drop.yaml`` sweeps
    both.

    Example
    -------
    >>> import torch
    >>> fuse = FuseBEVT(dim=32, mlp_dim=64, agent_size=3, window_size=4,
    ...                 dim_head=8, depth=2)
    >>> x = torch.randn(2, 3, 32, 8, 8)
    >>> fuse(x).shape
    torch.Size([2, 32, 8, 8])

    A per-agent presence mask is accepted and broadcast over space:

    >>> mask = torch.tensor([[True, True, False], [True, False, False]])
    >>> fuse(x, mask=mask).shape
    torch.Size([2, 32, 8, 8])
    """

    def __init__(self, dim: int = 128, mlp_dim: int = 256, agent_size: int = 5,
                 window_size=8, dim_head: int = 32, dropout: float = 0.0,
                 depth: int = 3, use_local: bool = True,
                 use_global: bool = True, pool: str = POOL_MEAN) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if pool not in (POOL_MEAN, POOL_MASKED_MEAN):
            raise ValueError(
                f"unknown pool {pool!r}; expected {POOL_MEAN!r} or "
                f"{POOL_MASKED_MEAN!r}")
        self.dim = int(dim)
        self.agent_size = int(agent_size)
        self.depth = int(depth)
        self.pool = pool

        self.blocks = nn.ModuleList([
            FAXSelfAttentionBlock(dim=dim, dim_head=dim_head,
                                  window_size=window_size,
                                  agent_size=agent_size, mlp_dim=mlp_dim,
                                  dropout=dropout, use_local=use_local,
                                  use_global=use_global)
            for _ in range(self.depth)])

        self.head_norm = nn.LayerNorm(dim)
        self.head_proj = nn.Linear(dim, dim)

    # -- helpers ------------------------------------------------------------

    def _normalise_mask(self, mask: Optional[torch.Tensor],
                        x: torch.Tensor) -> Optional[torch.Tensor]:
        """Accept (B, L) or (B, L, H, W); return (B, L, H, W) bool or None."""
        if mask is None:
            return None
        mask = mask.to(torch.bool)
        if mask.dim() == 2:
            height, width = x.shape[-2], x.shape[-1]
            return mask[:, :, None, None].expand(-1, -1, height, width)
        if mask.dim() == 4:
            return mask
        raise ValueError(
            f"mask must be (B, L) or (B, L, H, W), got shape {tuple(mask.shape)}")

    def _pool_agents(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                     taps: Optional[TapProtocol]) -> torch.Tensor:
        """(B, L, C, H, W) -> (B, C, H, W)."""
        if self.pool == POOL_MEAN or mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = mask[:, :, None, :, :].to(x.dtype)     # (B, L, 1, H, W)
            total = weights.sum(dim=1).clamp(min=1.0)        # (B, 1, H, W)
            pooled = (x * weights).sum(dim=1) / total
        emit(taps, pooled, module="FuseBEVT", location="fusebevt/pooled")
        return pooled

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fusebevt") -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"expected (B, L, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != self.agent_size:
            raise ValueError(
                f"agent axis is {x.shape[1]} but FuseBEVT was built for "
                f"agent_size={self.agent_size}. Pad the agent axis to max_cav "
                "and pass a mask; the relative position bias table has a "
                "fixed agent extent and cannot be resized per batch.")

        emit(taps, x, module="FuseBEVT", location=f"{location_prefix}/input")
        mask = self._normalise_mask(mask, x)
        if mask is not None:
            emit(taps, mask, module="FuseBEVT",
                 location=f"{location_prefix}/mask")

        for depth_index, block in enumerate(self.blocks):
            x = block(x, mask=mask, taps=taps,
                      location_prefix=f"{location_prefix}/d{depth_index}")

        pooled = self._pool_agents(x, mask, taps)

        # LayerNorm and Linear act on channels, so move them last and back.
        projected = rearrange(pooled, "b d h w -> b h w d")
        projected = self.head_norm(projected)
        projected = self.head_proj(projected)
        out = rearrange(projected, "b h w d -> b d h w")
        emit(taps, out, module="FuseBEVT", location=f"{location_prefix}/output")
        return out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, depth={self.depth}, "
                f"agent_size={self.agent_size}, pool={self.pool}")
