"""
mlp.py
------
The transformer feed-forward block: per-cell channel mixing between
attention layers.

Pre-norm construction (LayerNorm inside, residual added by the caller),
matching the reference. Kept as its own module rather than inlined in the
encoder because it is an injection point in its own right: the hidden
activation is the widest tensor in the fusion stack, and layer-wise
robustness sweeps need it addressable by name.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class FeedForward(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear, applied per BEV cell per agent.

    Inputs
    ------
    dim      feature channels
    mlp_dim  hidden width (reference: 256)
    dropout  applied after each linear

    Outputs
    -------
    (B, L, H, W, C) update; the caller adds the residual.

    Example
    -------
    >>> import torch
    >>> ffn = FeedForward(dim=8, mlp_dim=16)
    >>> ffn(torch.randn(1, 2, 4, 4, 8)).shape
    torch.Size([1, 2, 4, 4, 8])
    """

    def __init__(self, dim: int, mlp_dim: int = 256,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fusion/l0/ffn") -> torch.Tensor:
        hidden = self.drop(self.act(self.fc1(self.norm(x))))
        emit(taps, hidden, module="FeedForward",
             location=f"{location_prefix}/hidden")
        out = self.drop(self.fc2(hidden))
        emit(taps, out, module="FeedForward",
             location=f"{location_prefix}/out")
        return out
