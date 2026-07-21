"""
attention.py
------------
Scaled dot-product attention with every intermediate exposed.

This is the module the whole benchmark is built to observe. The reference
implementation computes::

    sim = einsum(...) ; sim = sim + bias ; sim = sim.masked_fill(...)
    attn = Softmax(dim=-1)(sim) ; out = einsum(...)

inside an ``nn.Sequential``-wrapped block, so none of ``sim``, ``bias`` or
``attn`` survives the call. Here each step is a named, tapped tensor:

    scores          raw Q K^T / sqrt(d)
    scores_biased   after the learned relative position bias
    scores_masked   after absent agents are driven to -inf
    softmax         the attention distribution itself
    attn_out        the weighted sum of values

(``attn_out``, not ``output``: the enclosing FAX half reports its own result
as ``half_out``, and two locations differing only by three characters is a
join key waiting to be mistyped in an analysis script.)

``softmax`` is the one that answers the question this benchmark exists to
ask: when a collaborator's pose is corrupted, does attention down-weight it,
or does it integrate the corrupted map anyway? That question has no answer
you can read off an accuracy drop.

Masking uses ``finfo.min``, not ``-inf``
----------------------------------------
A row that is entirely masked softmaxes ``-inf`` to NaN, which then
propagates silently through the rest of the forward pass and shows up as a
loss that is NaN for reasons nobody can trace. ``finfo.min`` softmaxes to a
uniform distribution instead -- still wrong, but finite and debuggable, and
identical to ``-inf`` in every non-degenerate case (its softmax weight is 0
to well under fp32 resolution). CoBEVT never fully masks a row in practice,
because the ego agent is always present; this is insurance against a fault
condition that makes it happen.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class ScaledDotProductAttention(nn.Module):
    """Attention over pre-projected Q/K/V, with all intermediates tapped.

    Purpose
        The single attention primitive shared by FAX self-attention
        (FuseBEVT), FAX cross-attention (SinBEVT) and SinBEVT's terminal
        dense self-attention. One implementation means one place where the
        scaling, the bias addition and the mask semantics are defined.

    Inputs
    ------
    dim_head  channels per head; sets the ``1 / sqrt(d)`` scale
    dropout   applied to the attention weights after the softmax. The
              reference applies dropout only in the output projection, so
              this defaults to 0.0 to stay faithful; it is exposed because
              attention dropout is the more common choice and someone will
              want to ablate it.
    retain_softmax  keep the last attention distribution on the module as
              ``last_softmax``. Off by default and deliberately so: at
              CoBEVT's FuseBEVT shapes one such tensor is
              ``(B*16, 4, 320, 320)`` -- about 26 MB at batch 1 -- and
              holding it across training steps is a slow memory leak that
              looks like a framework problem rather than a choice made here.
              Turn it on for tests and interactive inspection; use taps for
              anything running at scale.

    Outputs
    -------
    ``(B, H, Tq, dim_head)``.

    Shapes
    ------
    q      (B, H, Tq, d)
    k      (B, H, Tk, d)
    v      (B, H, Tk, d)
    bias   (H, Tq, Tk) or (1, H, Tq, Tk), broadcast over batch
    mask   broadcastable to (B, H, Tq, Tk); zero/False = masked out
    return (B, H, Tq, d)

    Example
    -------
    >>> import torch
    >>> attn = ScaledDotProductAttention(dim_head=32)
    >>> q = k = v = torch.randn(2, 4, 16, 32)
    >>> attn(q, k, v).shape
    torch.Size([2, 4, 16, 32])

    The softmax is a distribution over keys, so its rows sum to one:

    >>> inspectable = ScaledDotProductAttention(dim_head=32, retain_softmax=True)
    >>> _ = inspectable(q, k, v)
    >>> weights = inspectable.last_softmax
    >>> bool(torch.allclose(weights.sum(-1), torch.ones(2, 4, 16), atol=1e-5))
    True
    """

    def __init__(self, dim_head: int, dropout: float = 0.0,
                 retain_softmax: bool = False) -> None:
        super().__init__()
        if dim_head < 1:
            raise ValueError(f"dim_head must be positive, got {dim_head}")
        self.dim_head = int(dim_head)
        self.scale = float(dim_head) ** -0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.retain_softmax = bool(retain_softmax)
        self.last_softmax: Optional[torch.Tensor] = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "attention") -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        emit(taps, scores, module="ScaledDotProductAttention",
             location=f"{location_prefix}/scores")

        if bias is not None:
            scores = scores + bias
            emit(taps, scores, module="ScaledDotProductAttention",
                 location=f"{location_prefix}/scores_biased")

        if mask is not None:
            fill = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask.to(torch.bool), fill)
            emit(taps, scores, module="ScaledDotProductAttention",
                 location=f"{location_prefix}/scores_masked")

        weights = scores.softmax(dim=-1)
        emit(taps, weights, module="ScaledDotProductAttention",
             location=f"{location_prefix}/softmax")
        if self.retain_softmax:
            self.last_softmax = weights.detach()

        weights = self.dropout(weights)
        output = torch.matmul(weights, v)
        emit(taps, output, module="ScaledDotProductAttention",
             location=f"{location_prefix}/attn_out")
        return output

    def extra_repr(self) -> str:
        return f"dim_head={self.dim_head}, scale={self.scale:.6f}"
