"""
partition_hints.py
------------------
Work out a workable window plan, so a misconfiguration reports its own fix.

SinBEVT's cross-attention couples four independently configured numbers --
BEV size, query window, image feature size and key window -- through one
constraint::

    (bev / q_win)^2  ==  (feat_h / k_win_h) * (feat_w / k_win_w)

Anyone changing image resolution alone will break it. A message that only
says "4 windows against 1" is correct and still leaves the reader solving a
small factorisation by hand at the point of maximum irritation. This module
solves it for them.
"""

from __future__ import annotations

from typing import Optional, Tuple


def suggest_feature_window(bev: int, q_win: Tuple[int, int], feat_h: int,
                           feat_w: int) -> Optional[Tuple[int, int]]:
    """A ``feat_win_size`` giving the same window count as the query grid.

    Returns None when no integer window divides the feature map into the
    required number of windows -- in which case the caller must change
    ``q_win_size``, ``bev_size`` or the image resolution instead, and saying
    so is more useful than suggesting something that will not work.

    Example
    -------
    >>> suggest_feature_window(bev=128, q_win=(16, 16), feat_h=64, feat_w=64)
    (8, 8)
    >>> suggest_feature_window(bev=64, q_win=(16, 16), feat_h=32, feat_w=32)
    (8, 8)
    >>> suggest_feature_window(bev=32, q_win=(32, 32), feat_h=16, feat_w=16)
    (16, 16)

    A 16x16 feature map cannot be split into 9 windows, so there is nothing
    honest to suggest:

    >>> suggest_feature_window(bev=48, q_win=(16, 16), feat_h=16, feat_w=16)
    """
    if bev % q_win[0] or bev % q_win[1]:
        return None
    n_x, n_y = bev // q_win[0], bev // q_win[1]
    if n_x <= 0 or n_y <= 0:
        return None
    # Match the window layout axis by axis rather than only the total count:
    # the same number of windows in a different arrangement would attend
    # across mismatched regions while satisfying the assertion.
    if feat_h % n_x or feat_w % n_y:
        return None
    return (feat_h // n_x, feat_w // n_y)
