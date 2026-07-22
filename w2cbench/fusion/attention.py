"""
attention.py
------------
The attention primitives, instrumented.

Where2comm fuses per BEV cell: each of the ``H x W`` cells is its own tiny
attention problem whose sequence is the **agent axis**, whose query is the ego
and whose keys and values are the messages that arrived. That is an unusual
shape -- a very large batch of very short sequences -- and it is the reason
this module exists rather than ``nn.MultiheadAttention`` being called directly:
``fusion/r{k}/softmax`` is the tensor the whole benchmark is built to observe,
and PyTorch's fused implementation does not hand it back.

What ``fusion/r{k}/softmax`` answers
------------------------------------
For every BEV cell, how much weight did the ego give each collaborator? That
is the question a fault benchmark ultimately asks: when a collaborator's
sensor is degraded, or its pose is wrong, does attention *down-weight* it --
or does it integrate the corruption at full strength? A model can lose accuracy
either way, and the two failures call for completely different fixes, but they
are indistinguishable from the output alone.

Masking, and the one case that produces NaN
-------------------------------------------
Two things remove a key: the warp's validity mask (this collaborator does not
cover this cell) and the communication graph (this collaborator sent nothing).
Both are *absences*, and the distinction that matters is that neither is the
same as a reading of zero -- an unmasked zero is read by attention as a
confident observation of empty space.

Masked entries are pushed to ``finfo.min`` before the softmax. If every key for
a cell were masked, that softmax would return a uniform distribution over
entries that are all meaningless, which is silently worse than returning
nothing. In practice the ego is always its own valid key -- A6 leaves the
self-link unmasked and the identity warp is always in bounds -- but "in
practice" is not a guarantee, so the weights are multiplied by the mask and
renormalised afterwards. When nothing was valid the output is then exactly
zero, which is the honest answer.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cpbench.observation import TapProtocol, emit

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with every intermediate exposed.

    Purpose
        Compute the attention distribution over agents for every BEV cell,
        and emit the scores before masking, after masking, after the softmax,
        and the weighted sum -- so a fault can be traced to whichever of those
        it actually perturbed.

    Inputs
    ------
    dim   feature width used for the ``1/sqrt(d)`` scaling.

    Inputs (to :meth:`forward`)
    ---------------------------
    query  ``(N, nH, Tq, d)``  -- normally ``Tq = 1``, the ego.
    key    ``(N, nH, Tk, d)``  -- ``Tk = L``, one per agent.
    value  ``(N, nH, Tk, d)``
    mask   ``(N, 1, 1, Tk)`` or broadcastable; True/1 marks a *usable* key.
    gate   optional ``(N, 1, Tq, Tk)`` multiplied into the post-softmax
           weights -- the paper's confidence weighting (A5). Applied after
           renormalisation, because it is a gate rather than a redistribution:
           the weights deliberately stop summing to 1.

    Outputs
    -------
    ``(context, weights)`` with ``context`` ``(N, nH, Tq, d)`` and ``weights``
    ``(N, nH, Tq, Tk)``.

    Example
    -------
    >>> import torch
    >>> attn = ScaledDotProductAttention(dim=4)
    >>> q = torch.randn(6, 2, 1, 4)
    >>> k = v = torch.randn(6, 2, 3, 4)
    >>> context, weights = attn(q, k, v)
    >>> context.shape, weights.shape
    (torch.Size([6, 2, 1, 4]), torch.Size([6, 2, 1, 3]))
    >>> bool(torch.allclose(weights.sum(-1), torch.ones(6, 2, 1), atol=1e-6))
    True

    A masked key receives no weight:

    >>> mask = torch.ones(6, 1, 1, 3, dtype=torch.bool)
    >>> mask[:, :, :, 2] = False
    >>> _, weights = attn(q, k, v, mask=mask)
    >>> float(weights[..., 2].abs().max())
    0.0
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.scale = float(dim) ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                gate: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = (query @ key.transpose(-2, -1)) * self.scale
        emit(taps, scores, module="ScaledDotProductAttention",
             location=f"fusion/r{round_index}/scores")

        if mask is not None:
            usable = mask.to(torch.bool)
            scores = scores.masked_fill(~usable, torch.finfo(scores.dtype).min)
            emit(taps, scores, module="ScaledDotProductAttention",
                 location=f"fusion/r{round_index}/scores_masked")

        weights = F.softmax(scores, dim=-1)
        if mask is not None:
            # Belt and braces for the all-masked cell: softmax over entries
            # that are all finfo.min returns a uniform distribution over
            # meaningless keys, which is worse than returning nothing.
            weights = weights * usable.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        emit(taps, weights, module="ScaledDotProductAttention",
             location=f"fusion/r{round_index}/softmax")

        if gate is not None:
            weights = weights * gate

        context = weights @ value
        emit(taps, context, module="ScaledDotProductAttention",
             location=f"fusion/r{round_index}/attn_out")
        return context, weights

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class MultiHeadAttention(nn.Module):
    """Learned Q/K/V projections around :class:`ScaledDotProductAttention`.

    Purpose
        The parameterised attention used by the transformer aggregator. The
        released ``AttenFusion`` has no projections at all (see
        :mod:`w2cbench.fusion.aggregators`), so this is the only place
        ``fusion/r{k}/q``, ``/k`` and ``/v`` exist as tensors distinct from
        the input.

    Inputs
    ------
    dim      model width D.
    heads    number of attention heads; ``dim`` must divide by it.

    Inputs (to :meth:`forward`)
    ---------------------------
    query    ``(N, Tq, D)``
    key_value  ``(N, Tk, D)``
    mask     ``(N, 1, 1, Tk)``
    gate     ``(N, 1, Tq, Tk)`` -- see :class:`ScaledDotProductAttention`.

    Outputs
    -------
    ``(context, weights)`` with ``context`` ``(N, Tq, D)``.

    Example
    -------
    >>> import torch
    >>> mha = MultiHeadAttention(dim=8, heads=2)
    >>> context, weights = mha(torch.randn(5, 1, 8), torch.randn(5, 3, 8))
    >>> context.shape, weights.shape
    (torch.Size([5, 1, 8]), torch.Size([5, 2, 1, 3]))
    """

    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(
                f"dim {dim} is not divisible by heads {heads}; the head width "
                "is derived as dim // heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.dim_head = self.dim // self.heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.project = nn.Linear(dim, dim, bias=False)
        self.attend = ScaledDotProductAttention(self.dim_head)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        n, tokens, _ = x.shape
        return x.reshape(n, tokens, self.heads, self.dim_head).transpose(1, 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                gate: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._split(self.to_q(query))
        k = self._split(self.to_k(key_value))
        v = self._split(self.to_v(key_value))
        emit(taps, q, module="MultiHeadAttention",
             location=f"fusion/r{round_index}/q")
        emit(taps, k, module="MultiHeadAttention",
             location=f"fusion/r{round_index}/k")
        emit(taps, v, module="MultiHeadAttention",
             location=f"fusion/r{round_index}/v")

        context, weights = self.attend(q, k, v, mask=mask, gate=gate,
                                       taps=taps, round_index=round_index)
        n, _, tokens, _ = context.shape
        merged = context.transpose(1, 2).reshape(n, tokens, self.dim)
        return self.project(merged), weights

    def extra_repr(self) -> str:
        return f"dim={self.dim}, heads={self.heads}"


class FeedForward(nn.Module):
    """The transformer aggregator's position-wise MLP.

    Inputs
    ------
    dim          model width.
    hidden_dim   inner width; defaults to ``2 * dim``.
    dropout      applied after the activation and after the output.

    Shapes
    ------
    in/out ``(N, T, D)``; the hidden activation is ``(N, T, hidden_dim)``.

    Example
    -------
    >>> import torch
    >>> ff = FeedForward(dim=8).eval()
    >>> ff(torch.randn(4, 1, 8)).shape
    torch.Size([4, 1, 8])
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim if hidden_dim is not None else 2 * dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> torch.Tensor:
        hidden = self.dropout(F.relu(self.up(x)))
        emit(taps, hidden, module="FeedForward",
             location=f"fusion/r{round_index}/ffn_hidden")
        out = self.dropout(self.down(hidden))
        emit(taps, out, module="FeedForward",
             location=f"fusion/r{round_index}/ffn_out")
        return out
