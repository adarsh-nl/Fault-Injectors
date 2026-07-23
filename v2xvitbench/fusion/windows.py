"""
windows.py
----------
Window partitioning and the relative-position bias: the plumbing under MSwin.

Kept apart from ``mswin.py`` because the partition/unpartition pair is pure
tensor bookkeeping with an exact round-trip property, and the bias is a
self-contained parameter table -- each is testable in isolation, and neither
should be re-derived inside an attention forward where a transposed axis
would surface only as a slightly worse AP.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """Cut a spatial map into non-overlapping square windows.

    Shapes
    ------
    x  (N, H, W, C)  ->  (N * H//window * W//window, window*window, C)

    H and W must divide by ``window``; MSwin's geometry validation enforces
    that at model construction, and this function re-checks because it is
    also used standalone.

    Example
    -------
    >>> import torch
    >>> x = torch.arange(16).float().reshape(1, 4, 4, 1)
    >>> windows = window_partition(x, 2)
    >>> windows.shape
    torch.Size([4, 4, 1])
    >>> windows[0, :, 0].tolist()      # top-left window, row-major
    [0.0, 1.0, 4.0, 5.0]
    """
    n, height, width, channels = x.shape
    if height % window or width % window:
        raise ValueError(
            f"grid {height}x{width} does not divide into {window}x{window} "
            "windows; the model validates this at construction, so reaching "
            "here means a config bypassed validation")
    x = x.reshape(n, height // window, window, width // window, window,
                  channels)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, window * window, channels)


def window_unpartition(x: torch.Tensor, window: int,
                       hw: Tuple[int, int]) -> torch.Tensor:
    """Invert :func:`window_partition`.

    Shapes
    ------
    x  (N * H//window * W//window, window*window, C)  ->  (N, H, W, C)

    Example
    -------
    >>> import torch
    >>> x = torch.randn(2, 8, 8, 3)
    >>> back = window_unpartition(window_partition(x, 4), 4, (8, 8))
    >>> bool(torch.equal(back, x))     # exact round-trip, not approximate
    True
    """
    height, width = hw
    per_map = (height // window) * (width // window)
    n = x.shape[0] // per_map
    channels = x.shape[-1]
    x = x.reshape(n, height // window, width // window, window, window,
                  channels)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(n, height, width, channels)


class RelativePositionBias(nn.Module):
    """Learned bias on window-attention scores by relative cell offset.

    Purpose
        Give window attention a notion of geometry: two cells one metre
        apart relate differently than two cells across the window, and the
        dot product alone cannot express that. One learned scalar per
        (relative offset, head), shared by every window (the Swin
        construction the reference follows).

    Inputs
    ------
    window_size  the window's side length w
    num_heads    attention heads sharing the table

    Outputs
    -------
    (num_heads, w*w, w*w) bias, one row per query cell, added to the scores
    of every window.

    Example
    -------
    >>> bias = RelativePositionBias(window_size=2, num_heads=3)
    >>> bias().shape
    torch.Size([3, 4, 4])
    >>> bias.table.shape               # (2w-1)^2 offsets per head
    torch.Size([9, 3])
    """

    def __init__(self, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        span = 2 * self.window_size - 1
        self.table = nn.Parameter(torch.zeros(span * span, self.num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(self.window_size), torch.arange(self.window_size),
            indexing="ij"))                              # (2, w, w)
        flat = coords.reshape(2, -1)                     # (2, w*w)
        relative = flat[:, :, None] - flat[:, None, :]   # (2, w*w, w*w)
        relative = relative.permute(1, 2, 0) + (self.window_size - 1)
        index = relative[..., 0] * span + relative[..., 1]
        self.register_buffer("index", index)             # (w*w, w*w)

    def forward(self, taps: Optional[TapProtocol] = None,
                location_prefix: str = "") -> torch.Tensor:
        tokens = self.window_size * self.window_size
        bias = self.table[self.index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.num_heads).permute(2, 0, 1)
        if location_prefix:
            emit(taps, bias, module="RelativePositionBias",
                 location=f"{location_prefix}/rel_pos_bias")
        return bias

    def extra_repr(self) -> str:
        return f"window_size={self.window_size}, num_heads={self.num_heads}"
