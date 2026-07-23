"""
shrink.py
---------
The shrink header: one strided convolution between the BEV backbone and
everything transformer-shaped.

Why it exists
-------------
The pillar backbone hands over a 384-channel map at half the pillar-grid
resolution. V2X-ViT's transformer runs at a quarter of it: HMSA builds an
L x L attention problem *per BEV cell* and MSwin partitions the grid into
windows, so halving H and W once more quarters the attention cost -- and it
is also the map that crosses the V2X link, so the shrink is simultaneously a
compute and a bandwidth decision. The reference calls this ``DownsampleConv``
(``shrink_header``: 384 -> 256, kernel 3, stride 2).

The stride taken here is load-bearing far beyond this file: anchors, the box
decoder and the STTF warp are all built from the FUSION GridSpec, whose
``downsample`` must equal ``backbone stride x shrink stride``. The model
validates that identity at construction (``V2XViT._validate_geometry``)
because getting it wrong is silent -- everything still runs, at the wrong
geometry, and only AP notices.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class ShrinkConv(nn.Module):
    """Strided conv + BN + ReLU projecting the backbone map to fusion geometry.

    Purpose
        Halve the spatial resolution and project 384 -> 256 channels, fixing
        the geometry and width every downstream module (RTE, STTF, HMSA,
        MSwin, the detection head) operates at.

    Inputs
    ------
    in_channels   backbone output width (reference: 384)
    out_channels  fusion width (reference: 256)
    kernel        conv kernel size (reference: 3)
    stride        spatial stride (reference: 2); 1 keeps the resolution, for
                  configs whose fusion grid equals the backbone grid

    Outputs
    -------
    The shrunk map, emitted at ``encoder/shrunk``.

    Shapes
    ------
    x  (N, in_channels, H, W)  ->  (N, out_channels, H//stride, W//stride)

    Example
    -------
    >>> import torch
    >>> shrink = ShrinkConv(in_channels=8, out_channels=4, stride=2)
    >>> shrink(torch.randn(3, 8, 16, 16)).shape
    torch.Size([3, 4, 8, 8])
    """

    def __init__(self, in_channels: int = 384, out_channels: int = 256,
                 kernel: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel,
                      stride=self.stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        out = self.conv(x)
        emit(taps, out, module="ShrinkConv", location="encoder/shrunk")
        return out

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, stride={self.stride}")
