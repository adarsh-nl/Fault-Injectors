"""
partition.py
------------
The two token groupings that make FAX sparse: local windows and global grids.

This is the whole idea of the paper, and it is four characters wide::

    window   b n d (x w1) (y w2) -> b n x y w1 w2 d      inner factor = window
    grid     b n d (w1 x) (w2 y) -> b n x y w1 w2 d      outer factor = window

Window partitioning keeps each token group spatially contiguous -- a local
neighbourhood. Grid partitioning keeps the *outer* factor, so a group samples
positions with stride ``H // w1`` -- a sparse, dilated view of the whole map.
Both produce groups of exactly ``N * w1 * w2`` tokens, so attention over
either has the same cost, and both mix **every agent (or camera) at every
position they touch**. That is the "fused" in Fused Axial Attention.

Why these are functions and not nn.Modules
------------------------------------------
They carry no parameters and no state; a module wrapper would add a layer of
indirection around a reshape and, worse, would invite being dropped into an
``nn.Sequential`` -- which is exactly the construction that makes the
reference implementation's attention internals unreachable. Callers invoke
these directly and ``emit`` around them, so the partitioned tensor is a named
observation point rather than a hidden intermediate.

Why einops
----------
Written as raw ``reshape``/``permute`` chains, ``window`` and ``grid`` become
two nearly identical blocks of index arithmetic differing in one transpose.
The pattern strings above put the distinction where a reader can see it. See
``test_partition.py``, which asserts the two actually disagree -- a shapes-only
test would pass with both set to ``window``.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from einops import rearrange

WINDOW = "window"
GRID = "grid"

WindowSize = Union[int, Tuple[int, int]]

# (mode, channels_last) -> (forward pattern, inverse pattern)
_PATTERNS: Dict[Tuple[str, bool], Tuple[str, str]] = {
    (WINDOW, False): ("b n d (x w1) (y w2) -> b n x y w1 w2 d",
                      "b n x y w1 w2 d -> b n d (x w1) (y w2)"),
    (GRID, False): ("b n d (w1 x) (w2 y) -> b n x y w1 w2 d",
                    "b n x y w1 w2 d -> b n d (w1 x) (w2 y)"),
    (WINDOW, True): ("b n (x w1) (y w2) d -> b n x y w1 w2 d",
                     "b n x y w1 w2 d -> b n (x w1) (y w2) d"),
    (GRID, True): ("b n (w1 x) (w2 y) d -> b n x y w1 w2 d",
                   "b n x y w1 w2 d -> b n (w1 x) (w2 y) d"),
}


def as_window_size(window_size: WindowSize) -> Tuple[int, int]:
    """Normalise an int or (w1, w2) pair to a pair.

    Non-square windows are real: the OPV2V configs are square, but the
    nuScenes SinBEVT config uses ``feat_win_size: [[6, 12], ...]`` because
    the image feature maps are not square.

    >>> as_window_size(8)
    (8, 8)
    >>> as_window_size((6, 12))
    (6, 12)
    """
    if isinstance(window_size, int):
        return (window_size, window_size)
    w1, w2 = window_size
    return (int(w1), int(w2))


def _check(spatial: Tuple[int, int], window: Tuple[int, int],
           mode: str) -> None:
    height, width = spatial
    w1, w2 = window
    if w1 <= 0 or w2 <= 0:
        raise ValueError(f"window size must be positive, got {window}")
    if height % w1 or width % w2:
        raise ValueError(
            f"{mode} partition needs the feature map to divide evenly: "
            f"got {height}x{width} with window {w1}x{w2} "
            f"({height} % {w1} = {height % w1}, {width} % {w2} = {width % w2}). "
            "Either pad the feature map or pick a window size that divides it.")


def partition(x: torch.Tensor, window_size: WindowSize, mode: str = WINDOW,
              channels_last: bool = False) -> torch.Tensor:
    """Group a feature map into attention token groups.

    Purpose
        Produce the token grouping one FAX attention operates over. ``mode``
        selects local (``"window"``) or global (``"grid"``) sampling.

    Inputs
    ------
    x              (B, N, D, H, W), or (B, N, H, W, D) if ``channels_last``.
                   N is the agent axis in FuseBEVT and the camera axis in
                   SinBEVT -- the maths does not care which.
    window_size    int or (w1, w2).
    mode           ``"window"`` | ``"grid"``.
    channels_last  layout of ``x``.

    Outputs
    -------
    (B, N, X, Y, w1, w2, D) where X = H // w1 and Y = W // w2, always
    channels-last, because attention consumes channels-last.

    Example
    -------
    >>> import torch
    >>> x = torch.arange(2 * 1 * 1 * 4 * 4).float().reshape(2, 1, 1, 4, 4)
    >>> partition(x, 2, mode=WINDOW).shape
    torch.Size([2, 1, 2, 2, 2, 2, 1])
    >>> partition(x, 2, mode=GRID).shape
    torch.Size([2, 1, 2, 2, 2, 2, 1])

    The shapes match; the contents do not. The first window is a contiguous
    2x2 block, the first grid group samples every other row and column:

    >>> partition(x, 2, mode=WINDOW)[0, 0, 0, 0, :, :, 0].tolist()
    [[0.0, 1.0], [4.0, 5.0]]
    >>> partition(x, 2, mode=GRID)[0, 0, 0, 0, :, :, 0].tolist()
    [[0.0, 2.0], [8.0, 10.0]]
    """
    if x.dim() != 5:
        raise ValueError(
            f"expected a 5-D (B, N, D, H, W) tensor, got shape {tuple(x.shape)}")
    key = (mode, bool(channels_last))
    if key not in _PATTERNS:
        raise ValueError(
            f"unknown partition mode {mode!r}; expected {WINDOW!r} or {GRID!r}")
    w1, w2 = as_window_size(window_size)
    spatial = (x.shape[2], x.shape[3]) if channels_last else (x.shape[3], x.shape[4])
    _check(spatial, (w1, w2), mode)
    return rearrange(x, _PATTERNS[key][0], w1=w1, w2=w2)


def unpartition(x: torch.Tensor, window_size: WindowSize, mode: str = WINDOW,
                channels_last: bool = False) -> torch.Tensor:
    """Invert :func:`partition`.

    ``mode`` and ``window_size`` must match the call that produced ``x``;
    mixing them silently scrambles the feature map rather than raising, which
    is why ``test_partition`` checks the round trip for both modes and checks
    that crossing them does *not* round-trip.

    Inputs
    ------
    x  (B, N, X, Y, w1, w2, D)

    Outputs
    -------
    (B, N, D, H, W), or (B, N, H, W, D) if ``channels_last``.

    Example
    -------
    >>> import torch
    >>> x = torch.randn(2, 3, 8, 4, 4)
    >>> for mode in (WINDOW, GRID):
    ...     torch.equal(unpartition(partition(x, 2, mode), 2, mode), x)
    True
    True
    """
    if x.dim() != 7:
        raise ValueError(
            f"expected a 7-D (B, N, X, Y, w1, w2, D) tensor, "
            f"got shape {tuple(x.shape)}")
    key = (mode, bool(channels_last))
    if key not in _PATTERNS:
        raise ValueError(
            f"unknown partition mode {mode!r}; expected {WINDOW!r} or {GRID!r}")
    w1, w2 = as_window_size(window_size)
    return rearrange(x, _PATTERNS[key][1], w1=w1, w2=w2)


def pad_to_multiple(x: torch.Tensor, window_size: WindowSize,
                    channels_last: bool = False) -> torch.Tensor:
    """Zero-pad the spatial dims up to a multiple of the window.

    SinBEVT needs this: image feature maps do not generally divide evenly by
    the key/value window, and the reference implementation pads rather than
    constraining the backbone. FuseBEVT never needs it -- its BEV grid is
    chosen to divide -- so the eager config validation is the right place to
    catch a bad FuseBEVT window, not a silent pad here.

    >>> import torch
    >>> pad_to_multiple(torch.randn(1, 1, 4, 5, 7), 4).shape
    torch.Size([1, 1, 4, 8, 8])
    """
    w1, w2 = as_window_size(window_size)
    height, width = (x.shape[2], x.shape[3]) if channels_last else (x.shape[3], x.shape[4])
    pad_h = (w1 - height % w1) % w1
    pad_w = (w2 - width % w2) % w2
    if not pad_h and not pad_w:
        return x
    # F.pad works from the last dim backwards; the spatial dims sit in
    # different places depending on layout.
    pad = (0, 0, 0, pad_w, 0, pad_h) if channels_last else (0, pad_w, 0, pad_h)
    return torch.nn.functional.pad(x, pad)
