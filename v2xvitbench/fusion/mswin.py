"""
mswin.py
--------
Multi-scale window attention (MSwin): per-agent spatial attention at several
window sizes in parallel.

Where HMSA asks "which *agent* should I trust at this cell", MSwin asks
"which *neighbourhood* of my own map explains this cell" -- and asks it at
several ranges at once. Each branch runs plain windowed self-attention at its
own window size (4, 8, 16 cells in the released config, with more heads at
the finer scales), so the finest branch sharpens local structure while the
coarsest one can pull context from ~26 m away at stride 4. The paper's
stated motivation is robustness to *localisation error*: a feature displaced
by a bad pose lands within some branch's window, and that branch can still
associate it.

The branches are fused per channel by ``SplitAttn`` (the paper's choice) or
a naive mean (the ablation), which is a configuration decision, not an
architectural one -- hence ``fusion_method`` in the config (assumption A3).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit

from v2xvitbench.fusion.windows import (RelativePositionBias,
                                        window_partition, window_unpartition)


class BaseWindowAttention(nn.Module):
    """Windowed multi-head self-attention over one agent's BEV map.

    Purpose
        One MSwin branch: partition the map into ``window x window`` tiles
        and run self-attention inside each tile, with a learned relative-
        position bias. Agents are independent here -- the agent axis is
        folded into the batch, which is the formal statement of "per-agent
        spatial relationships" in the paper.

    Inputs
    ------
    dim                    feature channels
    heads, dim_head        attention geometry for THIS branch
    window_size            tile side length in BEV cells
    relative_pos_embedding use the learned bias (reference: true)
    dropout                attention + projection dropout

    Outputs
    -------
    Same shape as the input, still channels-last.

    Shapes
    ------
    x  (B, L, H, W, C) -> (B, L, H, W, C); internally
    (B*L*windows, heads, window^2, dim_head) per tile.

    Example
    -------
    >>> import torch
    >>> attn = BaseWindowAttention(dim=16, heads=2, dim_head=8, window_size=2)
    >>> attn(torch.randn(1, 3, 4, 4, 16)).shape
    torch.Size([1, 3, 4, 4, 16])
    """

    def __init__(self, dim: int, heads: int, dim_head: int, window_size: int,
                 relative_pos_embedding: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.window_size = int(window_size)
        self.scale = self.dim_head ** -0.5
        inner = self.heads * self.dim_head

        self.to_qkv = nn.Linear(self.dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, self.dim),
                                    nn.Dropout(dropout))
        self.attn_drop = nn.Dropout(dropout)
        self.bias = (RelativePositionBias(self.window_size, self.heads)
                     if relative_pos_embedding else None)

    def forward(self, x: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fusion/l0/mswin/w0") -> torch.Tensor:
        batch, agents, height, width, _ = x.shape
        w = self.window_size

        tiles = window_partition(x.reshape(batch * agents, height, width,
                                           self.dim), w)
        n_tiles, tokens, _ = tiles.shape
        qkv = self.to_qkv(tiles).reshape(n_tiles, tokens, 3, self.heads,
                                         self.dim_head)
        q, k, v = (t.squeeze(2).permute(0, 2, 1, 3)
                   for t in qkv.chunk(3, dim=2))     # (N, nH, T, d)
        emit(taps, q, module="BaseWindowAttention",
             location=f"{location_prefix}/q")
        emit(taps, k, module="BaseWindowAttention",
             location=f"{location_prefix}/k")
        emit(taps, v, module="BaseWindowAttention",
             location=f"{location_prefix}/v")

        scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.bias is not None:
            scores = scores + self.bias(taps, location_prefix)
        emit(taps, scores, module="BaseWindowAttention",
             location=f"{location_prefix}/scores")

        attn = self.attn_drop(scores.softmax(dim=-1))
        emit(taps, attn, module="BaseWindowAttention",
             location=f"{location_prefix}/softmax")

        out = attn @ v                                # (N, nH, T, d)
        emit(taps, out, module="BaseWindowAttention",
             location=f"{location_prefix}/attn_out")

        out = out.permute(0, 2, 1, 3).reshape(n_tiles, tokens,
                                              self.heads * self.dim_head)
        out = self.to_out(out)
        out = window_unpartition(out, w, (height, width))
        out = out.reshape(batch, agents, height, width, self.dim)
        emit(taps, out, module="BaseWindowAttention",
             location=f"{location_prefix}/out")
        return out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.heads}, "
                f"dim_head={self.dim_head}, window_size={self.window_size}")


class SplitAttn(nn.Module):
    """Learned per-channel arbitration between the MSwin branches.

    Purpose
        The paper's split-attention fusion (borrowed from ResNeSt): pool each
        branch globally, produce a per-channel softmax across branches, and
        mix. The weights say, per channel, whether short- or long-range
        context wins -- which is why they are tapped: under a localisation
        fault the interesting question is whether the model *shifts* toward
        the coarser branches.

    Shapes
    ------
    branches  R tensors (B, L, H, W, C)  ->  (B, L, H, W, C)
    weights   (B, L, R, C), softmax over R, emitted as ``.../weights``

    Example
    -------
    >>> import torch
    >>> fuse = SplitAttn(dim=8, n_branches=2)
    >>> outs = [torch.randn(1, 2, 4, 4, 8) for _ in range(2)]
    >>> fuse(outs).shape
    torch.Size([1, 2, 4, 4, 8])
    """

    def __init__(self, dim: int, n_branches: int, reduction: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_branches = int(n_branches)
        hidden = max(self.dim // reduction, 4)
        self.fc1 = nn.Linear(self.dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, self.dim * self.n_branches)

    def forward(self, branches: List[torch.Tensor],
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fusion/l0/mswin") -> torch.Tensor:
        if len(branches) != self.n_branches:
            raise ValueError(
                f"got {len(branches)} branch outputs but SplitAttn was built "
                f"for {self.n_branches}")
        stacked = torch.stack(branches, dim=2)        # (B, L, R, H, W, C)
        batch, agents, _, height, width, _ = stacked.shape

        pooled = stacked.mean(dim=(3, 4)).sum(dim=2)  # (B, L, C)
        logits = self.fc2(self.act(self.fc1(pooled)))
        logits = logits.reshape(batch, agents, self.n_branches, self.dim)
        weights = logits.softmax(dim=2)               # (B, L, R, C)
        emit(taps, weights, module="SplitAttn",
             location=f"{location_prefix}/weights")

        weights = weights.unsqueeze(3).unsqueeze(3)   # (B, L, R, 1, 1, C)
        return (stacked * weights).sum(dim=2)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_branches={self.n_branches}"


class PyramidWindowAttention(nn.Module):
    """MSwin: the parallel multi-scale window-attention pyramid.

    Purpose
        Run every :class:`BaseWindowAttention` branch on the same input and
        fuse the results -- ``split_attn`` (the paper) or ``naive`` mean
        (the ablation).

    Inputs
    ------
    dim           feature channels
    heads         per-branch head counts, finest window first
    dim_heads     per-branch head widths
    window_sizes  per-branch window sizes; all must divide the fused grid
    fusion_method ``"split_attn"`` or ``"naive"``

    Outputs
    -------
    (B, L, H, W, C), emitted at ``.../out``; residual added by the caller.

    Example
    -------
    >>> import torch
    >>> mswin = PyramidWindowAttention(dim=16, heads=(2, 2), dim_heads=(8, 8),
    ...                                window_sizes=(2, 4))
    >>> mswin(torch.randn(1, 2, 8, 8, 16)).shape
    torch.Size([1, 2, 8, 8, 16])
    """

    def __init__(self, dim: int, heads: Sequence[int] = (16, 8, 4),
                 dim_heads: Sequence[int] = (16, 32, 64),
                 window_sizes: Sequence[int] = (4, 8, 16),
                 relative_pos_embedding: bool = True,
                 fusion_method: str = "split_attn",
                 dropout: float = 0.0) -> None:
        super().__init__()
        if not (len(heads) == len(dim_heads) == len(window_sizes)):
            raise ValueError(
                f"heads {tuple(heads)}, dim_heads {tuple(dim_heads)} and "
                f"window_sizes {tuple(window_sizes)} must be the same length "
                "-- one entry per MSwin branch")
        if fusion_method not in ("split_attn", "naive"):
            raise ValueError(
                f"unknown fusion_method {fusion_method!r}; expected "
                "'split_attn' or 'naive'")
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.fusion_method = fusion_method
        self.branches = nn.ModuleList([
            BaseWindowAttention(dim, h, d, w, relative_pos_embedding, dropout)
            for h, d, w in zip(heads, dim_heads, window_sizes)])
        self.fuse = (SplitAttn(dim, len(self.branches))
                     if fusion_method == "split_attn" else None)

    def forward(self, x: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fusion/l0/mswin") -> torch.Tensor:
        outs = [branch(x, taps, f"{location_prefix}/w{j}")
                for j, branch in enumerate(self.branches)]
        if self.fuse is not None:
            fused = self.fuse(outs, taps, location_prefix)
        else:
            fused = torch.stack(outs, dim=0).mean(dim=0)
        emit(taps, fused, module="PyramidWindowAttention",
             location=f"{location_prefix}/out")
        return fused

    def extra_repr(self) -> str:
        return (f"window_sizes={self.window_sizes}, "
                f"fusion_method={self.fusion_method}")
