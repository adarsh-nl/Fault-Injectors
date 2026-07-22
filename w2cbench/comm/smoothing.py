"""
smoothing.py
------------
Gaussian smoothing of the spatial confidence map (assumption A9).

Why the paper smooths at all
----------------------------
Selection picks the highest-confidence cells. Without smoothing it will
happily pick a single cell whose neighbours carry no support at all -- a
detector artefact rather than an object. Those cells cost bandwidth and
contribute nothing, and worse, under a fault they are exactly what a degraded
encoder produces most of. Convolving with a small Gaussian makes a cell's
selection score depend on its neighbourhood, so isolated spikes fall below the
threshold while genuine object footprints, which are several cells across,
survive.

No parameters, by construction
------------------------------
The kernel is a registered buffer and the convolution is functional. The
released implementation uses an ``nn.Conv2d`` whose weights are overwritten
and never trained, which works but leaves a tensor in ``state_dict`` and in
``parameters()`` that looks trainable and is not. A buffer makes "this filter
is fixed" structural rather than a convention someone has to remember.

Normalisation, and a discrepancy worth naming (A16)
---------------------------------------------------
The released kernel is built from the continuous Gaussian density and is
never renormalised, so its discrete weights do not sum to 1 -- at the default
``k_size=3, c_sigma=1.0`` they sum to roughly 0.78. The filter therefore does
not only smooth, it *attenuates* the whole confidence map by about 22% before
the threshold is applied, which silently turns a configured threshold of 0.01
into an effective 0.0128.

``normalize=True`` (the default here) divides by the kernel sum, so a uniform
map passes through unchanged, which is what "smoothing" means and what any
reader would assume the number in a config does. ``normalize=False``
reproduces the released filter exactly for comparison. This departs from the
package's usual "default to the released behaviour" rule (A1) because there is
no released checkpoint for the datasets this package trains on, so the
threshold is ours to choose anyway -- and choosing it against an attenuating
filter would bake the discrepancy into every configured value.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from cpbench.observation import TapProtocol, emit

logger = logging.getLogger(__name__)


def gaussian_kernel_2d(k_size: int = 3, sigma: float = 1.0,
                       normalize: bool = True) -> torch.Tensor:
    """A square 2-D Gaussian kernel.

    Inputs
    ------
    k_size     odd kernel width in cells.
    sigma      standard deviation in cells.
    normalize  divide by the kernel sum so the weights sum to 1 (A16).

    Outputs
    -------
    ``(k_size, k_size)`` float tensor.

    Example
    -------
    >>> k = gaussian_kernel_2d(3, 1.0)
    >>> bool(torch.isclose(k.sum(), torch.tensor(1.0)))
    True
    >>> bool(k[1, 1] == k.max())                 # centre is the mode
    True
    >>> bool(torch.allclose(k, k.flip(0))) and bool(torch.allclose(k, k.flip(1)))
    True

    The un-normalised form is the released implementation's, and it does not
    sum to 1 -- which is the whole point of A16:

    >>> raw = gaussian_kernel_2d(3, 1.0, normalize=False)
    >>> bool(raw.sum() < 0.8)
    True
    """
    if k_size < 1 or k_size % 2 == 0:
        raise ValueError(f"k_size must be a positive odd integer, got {k_size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    center = k_size // 2
    coords = torch.arange(k_size, dtype=torch.float32) - center
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = (1.0 / (2.0 * math.pi * sigma ** 2)) * torch.exp(
        -(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum() if normalize else kernel


class GaussianSmoother(nn.Module):
    """Fixed low-pass filter over a confidence map (A9).

    Purpose
        Make a cell's selection score depend on its neighbourhood, so
        isolated high-confidence artefacts do not win bandwidth.

    Inputs
    ------
    k_size        odd kernel width (released default 3).
    sigma         standard deviation in cells (released default 1.0).
    normalize     see A16 above; True by default.
    padding_mode  ``"zeros"`` (released) or any mode ``F.pad`` accepts.
                  Zero padding pulls border cells down, because the kernel
                  averages in cells that do not exist -- so the map edge is
                  systematically less likely to be selected. That is a real
                  behavioural bias and it interacts with faults: a pose error
                  that shifts objects toward the boundary is then penalised
                  twice. ``"replicate"`` removes it. The default matches the
                  released code; the alternative is one config key away.

    Outputs
    -------
    Same shape as the input; each channel filtered independently.

    Shapes
    ------
    in/out  ``(N, C, H, W)`` -- normally ``(L, 1, H, W)``, one confidence map
            per agent.

    Example
    -------
    >>> smoother = GaussianSmoother(k_size=3, sigma=1.0)
    >>> flat = torch.full((2, 1, 8, 8), 0.5)
    >>> out = smoother(flat)
    >>> out.shape
    torch.Size([2, 1, 8, 8])
    >>> bool(torch.allclose(out[:, :, 3:5, 3:5], torch.tensor(0.5)))
    True

    An isolated spike is attenuated far more than a solid block, which is the
    behaviour selection depends on:

    >>> spike = torch.zeros(1, 1, 8, 8); spike[0, 0, 4, 4] = 1.0
    >>> block = torch.zeros(1, 1, 8, 8); block[0, 0, 3:6, 3:6] = 1.0
    >>> bool(smoother(spike)[0, 0, 4, 4] < smoother(block)[0, 0, 4, 4])
    True
    """

    def __init__(self, k_size: int = 3, sigma: float = 1.0,
                 normalize: bool = True, padding_mode: str = "zeros") -> None:
        super().__init__()
        self.k_size = int(k_size)
        self.sigma = float(sigma)
        self.padding_mode = str(padding_mode)
        self.pad = self.k_size // 2
        # A buffer, not a Parameter: this filter is fixed by construction and
        # must never appear to an optimiser as something it could train.
        self.register_buffer(
            "kernel", gaussian_kernel_2d(k_size, sigma, normalize))

    def forward(self, confidence: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> torch.Tensor:
        """Smooth every channel independently.

        `round_index` only names the tap location; the filter is stateless
        across rounds.
        """
        if self.k_size == 1:
            # A 1x1 normalised kernel is the identity. Short-circuiting keeps
            # "smoothing disabled" a genuinely free code path rather than a
            # convolution that happens to change nothing.
            return confidence
        channels = confidence.shape[1]
        weight = self.kernel.to(confidence.dtype).expand(channels, 1,
                                                         self.k_size,
                                                         self.k_size)
        padded = F.pad(confidence, (self.pad,) * 4, mode=self.padding_mode) \
            if self.padding_mode != "zeros" else F.pad(confidence,
                                                       (self.pad,) * 4)
        out = F.conv2d(padded, weight, groups=channels)
        emit(taps, out, module="GaussianSmoother",
             location=f"confidence/r{round_index}/smoothed")
        return out

    def extra_repr(self) -> str:
        return (f"k_size={self.k_size}, sigma={self.sigma}, "
                f"padding_mode={self.padding_mode!r}")
