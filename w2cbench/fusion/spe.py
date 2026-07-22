"""
spe.py
------
Sensor positional encoding: a prior that near observations are more reliable.

    SPE(dis, 2p)   = sin(dis / 10000^(2p/D))
    SPE(dis, 2p+1) = cos(dis / 10000^(2p/D))            (paper section 4.4)

The transformer's sinusoidal encoding with *physical distance* substituted for
sequence position. What it gives attention is the one thing the feature vector
cannot say for itself: how far the sensor that produced this cell was from the
cell. A LiDAR return at 8 m and one at 70 m produce comparable feature
magnitudes and wildly different reliability, and without this the model has no
way to prefer the near one except by learning proxies from the features.

Why it matters more under faults than in the clean case
-------------------------------------------------------
On clean data the encoding is a mild prior. Under a fault it is one of the few
signals that stays honest: a corrupted sensor produces corrupted *features*,
but the geometry of where the sender was standing is unaffected by fog, by
sensor noise, or by point-cloud thinning. It is only wrong when the fault is a
pose error -- which is precisely the fault this encoding cannot help with, and
one the benchmark should therefore expect to be the hardest of the set.

Off by default (A4)
-------------------
The released configuration runs ``with_spe: false`` for the aggregator this
package defaults to, so the encoding ships available and unused. Turning it on
is a config key and an ablation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class SensorPositionalEncoding(nn.Module):
    """Encode sender-to-cell distance as a feature-width sinusoid.

    Purpose
        Give attention a geometric prior it cannot read off the features.

    Inputs
    ------
    dim         D, the feature width; must be even, since the encoding pairs
                a sine and a cosine per frequency.
    base        the ``10000`` of the transformer formula, exposed because the
                useful frequency range depends on the metric scale of the
                scene -- a 40 m BEV grid and a 200 m one are not the same
                problem.

    Inputs (to :meth:`forward`)
    ---------------------------
    distances   ``(B, L, H, W)`` metres from each sender to each BEV cell.

    Outputs
    -------
    ``(B, L, D, H, W)``, ready to add to the warped feature maps.

    Example
    -------
    >>> import torch
    >>> spe = SensorPositionalEncoding(dim=8)
    >>> spe(torch.zeros(1, 2, 4, 4)).shape
    torch.Size([1, 2, 8, 4, 4])

    Distance zero is the encoding's origin -- all sines vanish, all cosines
    are one -- so a sender standing on a cell is encoded distinctly from one
    far away:

    >>> near = spe(torch.zeros(1, 1, 1, 1))[0, 0, :, 0, 0]
    >>> far = spe(torch.full((1, 1, 1, 1), 50.0))[0, 0, :, 0, 0]
    >>> float(near[0]), float(near[1])
    (0.0, 1.0)
    >>> bool((near - far).abs().sum() > 0.5)
    True
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(
                f"dim must be even (the encoding pairs sin and cos per "
                f"frequency), got {dim}")
        self.dim = int(dim)
        self.base = float(base)
        exponent = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        # A buffer: these frequencies are fixed by the formula, not learned.
        self.register_buffer("inv_frequency", 1.0 / (base ** exponent))

    def forward(self, distances: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> torch.Tensor:
        if distances.dim() != 4:
            raise ValueError(
                f"expected distances (B, L, H, W), got {tuple(distances.shape)}")
        batch, agents, height, width = distances.shape
        scaled = distances.unsqueeze(-1) * self.inv_frequency.to(distances.dtype)
        encoding = torch.stack((scaled.sin(), scaled.cos()), dim=-1)
        encoding = encoding.reshape(batch, agents, height, width, self.dim)
        encoding = encoding.permute(0, 1, 4, 2, 3).contiguous()
        emit(taps, encoding, module="SensorPositionalEncoding",
             location=f"fusion/r{round_index}/spe")
        return encoding

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}"


def sensor_distances(T_agent_to_ego: torch.Tensor, feature_hw: Tuple[int, int],
                     x_min: float, y_min: float, stride_x: float,
                     stride_y: float) -> torch.Tensor:
    """Metres from each sender's sensor to each BEV cell, in the ego frame.

    The sender's sensor sits at the translation column of its agent-to-ego
    transform, so this is where a *pose* error corrupts the encoding as well
    as the warp -- the one fault that makes the geometric prior lie rather
    than merely become less useful.

    Inputs
    ------
    T_agent_to_ego  ``(B, L, 4, 4)``
    feature_hw      ``(H, W)`` of the BEV feature grid
    x_min, y_min, stride_x, stride_y   the grid's metric layout, matching
                                       :class:`~w2cbench.fusion.align.SpatialTransform`

    Outputs
    -------
    ``(B, L, H, W)`` in metres.

    Example
    -------
    >>> import torch
    >>> T = torch.eye(4).expand(1, 1, 4, 4).contiguous()
    >>> d = sensor_distances(T, (2, 2), x_min=-1.0, y_min=-1.0,
    ...                      stride_x=1.0, stride_y=1.0)
    >>> d.shape
    torch.Size([1, 1, 2, 2])
    >>> bool(torch.allclose(d, torch.full((1, 1, 2, 2), 2.0 ** 0.5 / 2)))
    True
    """
    height, width = feature_hw
    device, dtype = T_agent_to_ego.device, T_agent_to_ego.dtype
    # Cell centres, matching align.SpatialTransform's align_corners convention.
    xs = x_min + (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride_x
    ys = y_min + (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride_y
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    sensor = T_agent_to_ego[..., :2, 3]                    # (B, L, 2)
    dx = grid_x[None, None] - sensor[..., 0, None, None]
    dy = grid_y[None, None] - sensor[..., 1, None, None]
    return torch.sqrt(dx ** 2 + dy ** 2)
