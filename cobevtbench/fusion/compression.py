"""
compression.py
--------------
The bandwidth knob: a 1x1 convolutional bottleneck on the transmitted BEV map.

CoBEVT transmits 32x32x128 per agent -- 524 KB uncompressed. The paper's
compression ablation trades that against IoU:

    0x  ->  524 KB  ->  60.4 IoU
    8x  ->   66 KB  ->  60.1
    16x ->   33 KB  ->  58.9
    32x ->   16 KB  ->  56.2
    64x ->    8 KB  ->  54.8

Reproducing that curve is one of the benchmark's targets, and it is also the
cleanest available check that the communication path is wired correctly: a
compressor that was constructed but never applied would leave the IoU flat
across every factor.

Why this is a fusion concern and not a data concern
---------------------------------------------------
It sits between SinBEVT and FuseBEVT -- after the feature exists, before it
crosses the link. That is also why it composes with, rather than duplicates,
the fault plane: compression is a *designed* bandwidth reduction, while
``BandwidthLimitInjector`` is an *unintended* one. Both can be active, and
telling them apart in the results is the point of keeping them separate.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class NaiveCompressor(nn.Module):
    """1x1 convolutional autoencoder on the channel axis.

    Purpose
        Shrink the transmitted BEV map by a configurable factor, reproducing
        the paper's compression ablation.

    Inputs
    ------
    dim     channel width of the BEV map (CoBEVT: 128)
    factor  compression ratio. ``0`` or ``1`` disables compression entirely
            and the module becomes a verified identity -- the reference uses
            ``compression: 0`` for that, and treating 0 as "divide by zero"
            rather than "off" is an easy misreading of the config.

    Outputs
    -------
    Same shape as the input; only the information content is reduced.

    Shapes
    ------
    x       (N, dim, H, W)  ->  (N, dim, H, W)
    latent  (N, dim // factor, H, W) internally

    Example
    -------
    >>> import torch
    >>> off = NaiveCompressor(dim=128, factor=0)
    >>> x = torch.randn(2, 128, 32, 32)
    >>> bool(torch.equal(off(x), x))          # exactly identity when disabled
    True
    >>> lossy = NaiveCompressor(dim=128, factor=16)
    >>> lossy.latent_dim, lossy(x).shape
    (8, torch.Size([2, 128, 32, 32]))
    >>> lossy.payload_bytes(32, 32)           # 8 channels x 32 x 32 x 4 bytes
    32768
    """

    def __init__(self, dim: int = 128, factor: int = 0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.factor = int(factor)
        self.enabled = self.factor > 1
        if self.enabled and self.dim % self.factor:
            raise ValueError(
                f"compression factor {self.factor} does not divide the channel "
                f"width {self.dim}; the bottleneck width would be fractional")
        self.latent_dim = self.dim // self.factor if self.enabled else self.dim

        if self.enabled:
            self.encoder = nn.Sequential(
                nn.Conv2d(self.dim, self.latent_dim, 1, bias=False),
                nn.BatchNorm2d(self.latent_dim), nn.ReLU(inplace=True))
            self.decoder = nn.Sequential(
                nn.Conv2d(self.latent_dim, self.dim, 1, bias=False),
                nn.BatchNorm2d(self.dim), nn.ReLU(inplace=True))
        else:
            self.encoder = None
            self.decoder = None

    def payload_bytes(self, height: int, width: int,
                      bytes_per_element: int = 4) -> int:
        """Bytes one agent transmits per frame at this compression factor."""
        return self.latent_dim * int(height) * int(width) * bytes_per_element

    def forward(self, x: torch.Tensor, taps: Optional[TapProtocol] = None
                ) -> torch.Tensor:
        if not self.enabled:
            return x
        latent = self.encoder(x)
        emit(taps, latent, module="NaiveCompressor", location="compress/encoded")
        out = self.decoder(latent)
        emit(taps, out, module="NaiveCompressor", location="compress/decoded")
        return out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, factor={self.factor}, "
                f"latent_dim={self.latent_dim}, enabled={self.enabled}")
