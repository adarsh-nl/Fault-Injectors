"""
mlp.py
------
The position-wise feed-forward network of a FAX block.

Paper Eq. 7-8 pair each attention with ``MLP(LN(x))`` under a residual::

    x <- x + Attention(LN(x))
    x <- x + MLP(LN(x))

There is deliberately **no** ``PreNormResidual`` wrapper here, although the
reference implementation has one and it would save three lines per use.

A wrapper of the form ``fn(LayerNorm(x)) + x`` hides two tensors -- the
normalised input and the attention output before the residual add -- inside
a call the caller cannot reach into. In the reference those wrappers are then
stacked in an ``nn.Sequential``, which is precisely why the attention scores
and softmax in that implementation are unobservable. Reintroducing the
wrapper here would recreate the problem one layer up.

So the FAX blocks write the residual out::

    normed = self.norm(x)
    emit(taps, normed, ..., location=f"{prefix}/normed")
    delta = self.attention(normed, ..., taps=taps)
    x = x + delta

Three lines instead of one, and every tensor has a name.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class FeedForward(nn.Module):
    """Position-wise MLP: Linear -> GELU -> Dropout -> Linear -> Dropout.

    Purpose
        The channel-mixing half of a transformer block. Operates on the last
        dimension only, so it is agnostic to how tokens are grouped -- the
        same module serves window and grid branches.

    Inputs
    ------
    dim          input and output channel dim
    hidden_dim   inner width. CoBEVT uses 256 for dim 128 in FuseBEVT
                 (``mlp_dim``) and ``2 * dim`` in SinBEVT's cross-view MLPs.
    dropout      applied after each Linear, per the reference.

    Outputs
    -------
    Same shape as the input.

    Shapes
    ------
    x  (..., dim)  ->  (..., dim)

    Example
    -------
    >>> import torch
    >>> mlp = FeedForward(dim=128, hidden_dim=256)
    >>> mlp(torch.randn(2, 320, 128)).shape
    torch.Size([2, 320, 128])
    >>> mlp(torch.randn(2, 5, 8, 8, 128)).shape
    torch.Size([2, 5, 8, 8, 128])
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, taps: Optional[TapProtocol] = None,
                location_prefix: str = "mlp") -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        emit(taps, h, module="FeedForward", location=f"{location_prefix}/hidden")
        h = self.drop1(h)
        out = self.fc2(h)
        out = self.drop2(out)
        emit(taps, out, module="FeedForward", location=f"{location_prefix}/out")
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"
