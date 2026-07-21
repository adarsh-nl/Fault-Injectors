"""
decoder.py
----------
Upsample the fused 32x32 BEV map back to the 256x256 segmentation grid.

Three blocks, each ``conv -> BN -> ReLU -> upsample x2 -> conv -> BN ->
ReLU``, widening the spatial extent while narrowing channels::

    128 @ 32x32  ->  128 @ 64x64  ->  64 @ 128x128  ->  32 @ 256x256

Assumption A4
-------------
The paper says bilinear upsampling; ``NaiveDecoder`` in the released code
uses ``mode="nearest"``. The code wins by default (it produced the published
weights) and ``upsample_mode`` is exposed. The difference is not academic
for a fault benchmark: nearest-neighbour upsampling preserves blocky
artefacts that bilinear would smooth away, so a corruption visible at 32x32
stays visible at 256x256 rather than being partially hidden by the decoder.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class NaiveDecoder(nn.Module):
    """Upsample a BEV feature map to segmentation resolution.

    Purpose
        Bridge the 32x32 grid FuseBEVT works on and the 256x256 grid IoU is
        computed at.

    Inputs
    ------
    input_dim      channels arriving from FuseBEVT (CoBEVT: 128)
    num_ch_dec     per-stage widths, coarse-last (CoBEVT: [32, 64, 128])
    upsample_mode  ``"nearest"`` (released code) or ``"bilinear"`` (paper);
                   assumption A4

    Outputs
    -------
    ``(B, num_ch_dec[0], H * 2^n, W * 2^n)``.

    Shapes
    ------
    x       (B, input_dim, H, W)
    return  (B, 32, H*8, W*8) for CoBEVT's three stages

    Example
    -------
    >>> import torch
    >>> dec = NaiveDecoder(input_dim=16, num_ch_dec=[4, 8, 16])
    >>> dec(torch.randn(2, 16, 4, 4)).shape
    torch.Size([2, 4, 32, 32])
    """

    def __init__(self, input_dim: int = 128,
                 num_ch_dec: Sequence[int] = (32, 64, 128),
                 upsample_mode: str = "nearest") -> None:
        super().__init__()
        if upsample_mode not in ("nearest", "bilinear"):
            raise ValueError(
                f"unknown upsample_mode {upsample_mode!r}; "
                "expected 'nearest' (released code) or 'bilinear' (paper)")
        self.num_ch_dec = [int(c) for c in num_ch_dec]
        self.num_layer = len(self.num_ch_dec)
        self.upsample_mode = upsample_mode

        # Built in the order they run: coarsest stage first.
        self.stages = nn.ModuleList()
        for i in reversed(range(self.num_layer)):
            in_channels = (input_dim if i == self.num_layer - 1
                           else self.num_ch_dec[i + 1])
            self.stages.append(nn.ModuleList([
                _conv_bn_relu(in_channels, self.num_ch_dec[i]),
                _conv_bn_relu(self.num_ch_dec[i], self.num_ch_dec[i])]))

    def forward(self, x: torch.Tensor, taps: Optional[TapProtocol] = None,
                location_prefix: str = "decoder") -> torch.Tensor:
        align = None if self.upsample_mode == "nearest" else False
        for index, (before, after) in enumerate(self.stages):
            x = before(x)
            x = nn.functional.interpolate(x, scale_factor=2.0,
                                          mode=self.upsample_mode,
                                          align_corners=align)
            x = after(x)
            emit(taps, x, module="NaiveDecoder",
                 location=f"{location_prefix}/up{index}")
        return x

    def extra_repr(self) -> str:
        return (f"num_ch_dec={self.num_ch_dec}, mode={self.upsample_mode}, "
                f"scale=x{2 ** self.num_layer}")
