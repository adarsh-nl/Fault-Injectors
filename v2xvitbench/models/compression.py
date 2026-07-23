"""
compression.py
--------------
The bandwidth knob: a 1x1 convolutional bottleneck on the transmitted BEV map.

V2X-ViT's reference config carries a ``compression`` rate (0 in the released
setting, up to 32 in the staged-training recipe) that squeezes the 256-channel
map through a narrow autoencoder before it crosses the link. It models a
*designed* bandwidth reduction, which is exactly why it stays separate from
the fault plane's ``BandwidthLimitInjector`` -- an *unintended* one. Both can
be active, and telling them apart in the results is the point.

Contract-identical to the compressor in cobevtbench; re-implemented because
paper packages must not import each other (see ``fusion/geometry.py`` for the
same note). One deliberate difference: the output is emitted at
``encoder/compressed`` even when compression is off, so clean and compressed
runs join on the same location in layer-wise analyses.
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
        the reference's ``compression`` setting.

    Inputs
    ------
    dim     channel width of the BEV map (V2X-ViT: 256)
    factor  compression ratio. ``0`` or ``1`` disables compression entirely
            and the module becomes a verified identity -- the reference uses
            ``compression: 0`` for that, and treating 0 as "divide by zero"
            rather than "off" is an easy misreading of the config.

    Outputs
    -------
    Same shape as the input; only the information content is reduced. Emitted
    at ``encoder/compressed`` whether or not compression is enabled.

    Shapes
    ------
    x       (N, dim, H, W)  ->  (N, dim, H, W)
    latent  (N, dim // factor, H, W) internally

    Example
    -------
    >>> import torch
    >>> off = NaiveCompressor(dim=32, factor=0)
    >>> x = torch.randn(2, 32, 8, 8)
    >>> bool(torch.equal(off(x), x))          # exactly identity when disabled
    True
    >>> lossy = NaiveCompressor(dim=32, factor=16)
    >>> lossy.latent_dim, lossy(x).shape
    (2, torch.Size([2, 32, 8, 8]))
    >>> lossy.payload_bytes(8, 8)             # 2 channels x 8 x 8 x 4 bytes
    512
    """

    def __init__(self, dim: int = 256, factor: int = 0) -> None:
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
        if self.enabled:
            latent = self.encoder(x)
            x = self.decoder(latent)
        emit(taps, x, module="NaiveCompressor", location="encoder/compressed")
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, factor={self.factor}, "
                f"latent_dim={self.latent_dim}, enabled={self.enabled}")
