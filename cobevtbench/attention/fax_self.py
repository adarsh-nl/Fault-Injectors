"""
fax_self.py
-----------
FAX self-attention: one local (window) half followed by one global (grid)
half, each a pre-norm attention plus a pre-norm MLP under residuals.

Paper Eq. 7-8, written out::

    x <- x + Fused-Unblock( 3D-Rel-Attn( Fused-Block(LN(x)) ) )
    x <- x + MLP(LN(x))
    x <- x + Fused-Ungrid ( 3D-Rel-Attn( Fused-Grid (LN(x)) ) )
    x <- x + MLP(LN(x))

The two halves are the same code with a different partition mode, so they are
the same class (:class:`FAXAttentionHalf`) instantiated twice. The ablation in
paper section 7.3 -- local only 57.8, global only 57.9, both 60.4 -- is
reachable by disabling either half.

The agent axis is inside the tokens, not beside them
-----------------------------------------------------
After partitioning, tokens are flattened as ``(l w1 w2)``: one attention
operation mixes every agent at every position in the window simultaneously,
``5 * 8 * 8 = 320`` tokens at CoBEVT's settings. There is no separate
"fuse across agents" step. This is why the relative position bias is 3-D and
why the agent axis must be padded to a fixed ``agent_size`` -- see
:class:`~cobevtbench.attention.rel_pos_bias.RelativePositionBias`.

Structure note
--------------
The reference implementation builds this as::

    nn.Sequential(Rearrange(...), PreNormResidual(Attention(...)),
                  PreNormResidual(FeedForward(...)), Rearrange(...), ...)

which makes every tensor between the two Rearranges unreachable. Here each
step is written out and tapped. The arithmetic is identical; a checkpoint
trained with either loads into the other.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from cpbench.observation import TapProtocol, emit

from .attention import ScaledDotProductAttention
from .mlp import FeedForward
from .partition import GRID, WINDOW, as_window_size, partition, unpartition
from .qkv import FusedQKVProjection, merge_heads
from .rel_pos_bias import RelativePositionBias

# mode -> the name it is reported under in the observation registry
_HALF_NAME = {WINDOW: "local", GRID: "global"}


class FAXAttentionHalf(nn.Module):
    """One half of a FAX block: partition, attend, MLP, un-partition.

    Purpose
        The local half (``mode=WINDOW``) and the global half (``mode=GRID``)
        of Fused Axial Attention. Identical apart from how tokens are
        grouped, which is the paper's entire point.

    Inputs
    ------
    dim          channel dim (CoBEVT: 128)
    dim_head     channels per head; heads = dim // dim_head (assumption A9)
    window_size  int or (w1, w2) (CoBEVT: 8)
    agent_size   fixed extent of the agent axis (CoBEVT: max_cav = 5). The
                 relative position bias table is allocated for exactly this,
                 so inputs must always be padded to it.
    mlp_dim      feed-forward inner width (CoBEVT: 256)
    dropout      applied in the output projection and the MLP, per reference
    mode         ``WINDOW`` (local) or ``GRID`` (global)

    Outputs
    -------
    Same shape as the input.

    Shapes
    ------
    x       (B, L, D, H, W)     L must equal agent_size
    mask    (B, L, H, W) bool   True = agent present and in range here
    return  (B, L, D, H, W)

    Internally, with X = H // w1 and Y = W // w2 and T = L * w1 * w2:
        partitioned  (B, L, X, Y, w1, w2, D)
        tokens       (B * X * Y, T, D)
        q, k, v      (B * X * Y, heads, T, dim_head)
        bias         (heads, T, T)

    Example
    -------
    >>> import torch
    >>> half = FAXAttentionHalf(dim=32, dim_head=8, window_size=4,
    ...                         agent_size=2, mlp_dim=64)
    >>> half(torch.randn(1, 2, 32, 8, 8)).shape
    torch.Size([1, 2, 32, 8, 8])
    """

    def __init__(self, dim: int, dim_head: int, window_size, agent_size: int,
                 mlp_dim: int, dropout: float = 0.0,
                 mode: str = WINDOW) -> None:
        super().__init__()
        if mode not in _HALF_NAME:
            raise ValueError(
                f"unknown mode {mode!r}; expected {WINDOW!r} or {GRID!r}")
        self.mode = mode
        self.half_name = _HALF_NAME[mode]
        self.dim = int(dim)
        self.agent_size = int(agent_size)
        self.window_size: Tuple[int, int] = as_window_size(window_size)
        w1, w2 = self.window_size

        self.norm_attn = nn.LayerNorm(dim)
        self.qkv = FusedQKVProjection(dim, dim_head, bias=False)
        self.num_heads = self.qkv.num_heads
        self.rel_pos_bias = RelativePositionBias(
            window_size=(self.agent_size, w1, w2), num_heads=self.num_heads)
        self.attend = ScaledDotProductAttention(dim_head)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout)

    # -- mask ---------------------------------------------------------------

    def _attention_mask(self, mask: torch.Tensor, batch: int,
                        n_windows: int) -> torch.Tensor:
        """(B, L, H, W) bool -> (B*X*Y, 1, 1, T), broadcast over query rows.

        Masking keys rather than queries is what the reference does and is
        the right semantics: an absent agent must contribute nothing to
        anyone's output, but its own (zero-padded) query rows are discarded
        downstream anyway.
        """
        grouped = partition(mask.unsqueeze(2), self.window_size, self.mode)
        # (B, L, X, Y, w1, w2, 1) -> (B*X*Y, T)
        flat = rearrange(grouped, "b l x y w1 w2 1 -> (b x y) (l w1 w2)")
        return flat[:, None, None, :]

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fax") -> torch.Tensor:
        batch, agents = x.shape[0], x.shape[1]
        if agents != self.agent_size:
            raise ValueError(
                f"agent axis is {agents} but this block was built for "
                f"agent_size={self.agent_size}. The relative position bias "
                "table is allocated for a fixed agent extent, so inputs must "
                "be padded to max_cav rather than passed at their true count.")
        prefix = f"{location_prefix}/{self.half_name}"
        w1, w2 = self.window_size

        grouped = partition(x, self.window_size, self.mode)
        emit(taps, grouped, module="FAXAttentionHalf",
             location=f"{prefix}/partitioned")
        n_x, n_y = grouped.shape[2], grouped.shape[3]

        tokens = rearrange(grouped, "b l x y w1 w2 d -> (b x y) (l w1 w2) d")

        # -- attention, residual written out ---------------------------------
        normed = self.norm_attn(tokens)
        emit(taps, normed, module="FAXAttentionHalf", location=f"{prefix}/normed")
        q, k, v = self.qkv(normed, taps=taps, location_prefix=prefix)
        bias = self.rel_pos_bias(taps=taps, location=f"{prefix}/rel_pos_bias")
        attn_mask = None
        if mask is not None:
            attn_mask = self._attention_mask(mask, batch, n_x * n_y)
            emit(taps, attn_mask, module="FAXAttentionHalf",
                 location=f"{prefix}/attention_mask")
        attended = self.attend(q, k, v, bias=bias, mask=attn_mask, taps=taps,
                               location_prefix=prefix)
        delta = self.to_out(merge_heads(attended))
        delta = self.out_drop(delta)
        emit(taps, delta, module="FAXAttentionHalf",
             location=f"{prefix}/attn_delta")
        tokens = tokens + delta
        emit(taps, tokens, module="FAXAttentionHalf",
             location=f"{prefix}/attn_residual")

        # -- MLP, residual written out ---------------------------------------
        normed = self.norm_mlp(tokens)
        delta = self.mlp(normed, taps=taps, location_prefix=f"{prefix}/mlp")
        tokens = tokens + delta
        emit(taps, tokens, module="FAXAttentionHalf",
             location=f"{prefix}/mlp_residual")

        grouped = rearrange(
            tokens, "(b x y) (l w1 w2) d -> b l x y w1 w2 d",
            b=batch, x=n_x, y=n_y, l=agents, w1=w1, w2=w2)
        out = unpartition(grouped, self.window_size, self.mode)
        emit(taps, out, module="FAXAttentionHalf", location=f"{prefix}/half_out")
        return out

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, window_size={self.window_size}, "
                f"agent_size={self.agent_size}, num_heads={self.num_heads}")


class FAXSelfAttentionBlock(nn.Module):
    """A full FAX block: local half then global half.

    Purpose
        The repeating unit of FuseBEVT. Stacking ``depth`` of these is the
        entire fusion network.

    Inputs
    ------
    Same as :class:`FAXAttentionHalf`, plus:
    use_local / use_global  disable either half, for the paper's section 7.3
                            ablation. Disabling both leaves an identity
                            block, which is the paper's "neither" row.

    Outputs
    -------
    Same shape as the input.

    Shapes
    ------
    x       (B, L, D, H, W)
    mask    (B, L, H, W) bool, optional
    return  (B, L, D, H, W)

    Example
    -------
    >>> import torch
    >>> block = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
    ...                               agent_size=2, mlp_dim=64)
    >>> x = torch.randn(1, 2, 32, 8, 8)
    >>> block(x).shape
    torch.Size([1, 2, 32, 8, 8])

    Local and global are separate parameter sets, not a shared block applied
    twice:

    >>> block.local is block.global_half
    False
    """

    def __init__(self, dim: int, dim_head: int, window_size, agent_size: int,
                 mlp_dim: int, dropout: float = 0.0, use_local: bool = True,
                 use_global: bool = True) -> None:
        super().__init__()
        self.use_local = bool(use_local)
        self.use_global = bool(use_global)
        # Built unconditionally even when disabled, so an ablation and a full
        # model produce the same parameter names and a checkpoint from one
        # loads into the other with strict=False rather than silently
        # mismatching.
        self.local = FAXAttentionHalf(dim, dim_head, window_size, agent_size,
                                      mlp_dim, dropout, mode=WINDOW)
        self.global_half = FAXAttentionHalf(dim, dim_head, window_size,
                                            agent_size, mlp_dim, dropout,
                                            mode=GRID)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fusebevt/d0") -> torch.Tensor:
        if self.use_local:
            x = self.local(x, mask=mask, taps=taps,
                           location_prefix=location_prefix)
        if self.use_global:
            x = self.global_half(x, mask=mask, taps=taps,
                                 location_prefix=location_prefix)
        emit(taps, x, module="FAXSelfAttentionBlock",
             location=f"{location_prefix}/block_out")
        return x

    def extra_repr(self) -> str:
        return f"use_local={self.use_local}, use_global={self.use_global}"
