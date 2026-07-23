"""
prior.py
--------
The metadata the features travel with: velocity, time delay, agent type --
and the delay-aware positional encoding (DPE) that consumes the delay.

Why delay is an *input* and not just a nuisance
-----------------------------------------------
V2X communication is asynchronous: by the time a collaborator's features
arrive, they describe a scene ``dt`` frames old. V2X-ViT's answer (paper
section 4.2, "delay-aware positional encoding"; the reference code's
``RelTemporalEncoding``, adapted from heterogeneous graph transformers) is to
*tell the model* how stale each agent's features are, as a learned embedding
added uniformly across that agent's map, and let attention learn to discount
accordingly.

That design has a failure mode the paper never tests and this package exists
to measure: the encoding is only as good as the reported delay. The
``delay_encoding`` fault plane feeds the DPE a delay that disagrees with the
features' actual staleness -- ``rte/embedding`` is the tensor the lie enters
through, and it is tapped for exactly that reason.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class PriorEncoder(nn.Module):
    """Assemble the per-agent metadata triple the reference carries.

    Purpose
        Normalise and stack [velocity / 30, time delay, infrastructure flag]
        into the ``(B, L, 3)`` prior the reference code appends to the
        feature channels. Here the triple is kept *out* of the feature map --
        the consumers (DPE for delay, HMSA for type) receive their fields
        explicitly -- but the assembled prior is still built and emitted so
        the metadata entering the model is observable in one place, after
        every fault plane has acted on it.

    Inputs
    ------
    velocity    (B, L) float, m/s; the reference normalises by 30
    time_delay  (B, L) integer-valued, frames
    infra       (B, L) integer-valued, 0 = vehicle, 1 = infrastructure

    Outputs
    -------
    (B, L, 3) float prior, emitted at ``input/prior_encoding``.

    Example
    -------
    >>> import torch
    >>> prior = PriorEncoder()(velocity=torch.tensor([[15.0, 30.0]]),
    ...                        time_delay=torch.tensor([[0, 2]]),
    ...                        infra=torch.tensor([[0, 1]]))
    >>> prior.shape, prior[0, 1].tolist()
    (torch.Size([1, 2, 3]), [1.0, 2.0, 1.0])
    """

    #: the reference's velocity normaliser (30 m/s ~ urban speed ceiling)
    VELOCITY_SCALE = 30.0

    def forward(self, velocity: torch.Tensor, time_delay: torch.Tensor,
                infra: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        prior = torch.stack([velocity.float() / self.VELOCITY_SCALE,
                             time_delay.float(), infra.float()], dim=-1)
        emit(taps, prior, module="PriorEncoder",
             location="input/prior_encoding")
        return prior


class DelayPositionalEncoding(nn.Module):
    """The DPE / RTE: a learned embedding of each agent's time delay.

    Purpose
        Compensate for asynchronous arrival by adding a per-agent embedding
        of its delay to every cell of its feature map, before fusion. The
        embedding is a fixed sinusoidal table row passed through a learned
        linear map (the reference's ``RelTemporalEncoding``), so nearby
        delays get nearby encodings and the linear layer learns what the
        fusion stack should do about them.

    Inputs
    ------
    dim        feature channels the embedding is added to
    max_delay  largest representable delay, frames; larger delays clamp
    ratio      the reference's ``RTE_ratio``: table rows per frame of delay,
               spreading adjacent delays further apart in the table

    Outputs
    -------
    Features with the delay embedding broadcast-added, same shape as input.

    Shapes
    ------
    x    (B, L, C, H, W)
    dts  (B, L) integer-valued delays in frames
    ->   (B, L, C, H, W)

    Emits ``rte/embedding`` (B, L, C) and ``rte/output``.

    Example
    -------
    >>> import torch
    >>> rte = DelayPositionalEncoding(dim=8, max_delay=10, ratio=2)
    >>> x = torch.zeros(1, 2, 8, 4, 4)
    >>> out = rte(x, torch.tensor([[0, 3]]))
    >>> out.shape
    torch.Size([1, 2, 8, 4, 4])
    >>> bool((out[0, 0, :, 0, 0] == out[0, 0, :, 3, 3]).all())  # uniform
    True
    """

    def __init__(self, dim: int, max_delay: int = 100, ratio: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_delay = int(max_delay)
        self.ratio = int(ratio)

        rows = self.max_delay * self.ratio + 1
        position = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / self.dim))
        table = torch.zeros(rows, self.dim)
        table[:, 0::2] = torch.sin(position * div)
        table[:, 1::2] = torch.cos(position * div[: (self.dim + 1) // 2])
        # Fixed sinusoids, learned readout: the table is a buffer, the linear
        # map is the only trainable part (reference behaviour).
        self.register_buffer("table", table)
        self.linear = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor, dts: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "rte") -> torch.Tensor:
        if x.shape[:2] != dts.shape[:2]:
            raise ValueError(
                f"features are {tuple(x.shape[:2])} but delays are "
                f"{tuple(dts.shape)} on (batch, agent)")
        index = (dts.long() * self.ratio).clamp_(0, self.table.shape[0] - 1)
        embedding = self.linear(self.table[index])          # (B, L, C)
        emit(taps, embedding, module="DelayPositionalEncoding",
             location=f"{location_prefix}/embedding")

        out = x + embedding.unsqueeze(-1).unsqueeze(-1)
        emit(taps, out, module="DelayPositionalEncoding",
             location=f"{location_prefix}/output")
        return out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, max_delay={self.max_delay}, "
                f"ratio={self.ratio}")
