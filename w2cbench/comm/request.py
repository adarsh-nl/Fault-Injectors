"""
request.py
----------
The request map: where an agent is asking others to look.

    R_i^(k) = 1 - C_i^(k)  in  R^(H x W)      (paper section 4.3)

Why the complement of confidence is the right question
------------------------------------------------------
A cell where an agent is confident needs no help. A cell where it is not is
either empty road or something it cannot see -- occluded, too distant, too
sparsely sampled -- and the agent has no way to tell those apart from its own
observation alone. That ambiguity is exactly what a collaborator can resolve,
so uncertainty is the correct broadcast signal even though it is also true of
empty road: a partner that is confident there is nothing there sends nothing,
and a partner that is confident there is a vehicle there sends it.

The selection rule downstream is what makes this pay off. A cell is worth
transmitting only when ``C_i (X) R_j`` is high -- the sender is confident AND
the receiver is not -- which selects for complementarity rather than for
confidence, and is what stops round 2 from re-sending round 1.

Why one line gets its own module
--------------------------------
Three reasons, none of them about the arithmetic.

*It is the control payload.* Everything else on the wire is features. The
request map is the only message that steers the protocol rather than feeding
perception, which makes it the thing a protocol-plane fault acts on
(``RequestLossInjector``, design doc section 6.2). A fault needs a named seam.

*It is an observation point.* ``comm/r{k}/request_map`` is where you look to
answer "did this agent know it was blind?" -- the question that separates a
model degrading gracefully from one degrading silently.

*It is the natural place for a variant.* A learned request head, or one that
accounts for expected occlusion rather than raw uncertainty, replaces this
module and nothing else.

With ``rounds=1`` the map is computed and transmitted but never consumed:
nobody gets a chance to answer it. That is faithful to the released
single-round configuration, and it is why ``RequestLossInjector`` is provably
a no-op at K=1 -- a fact the fault suite asserts rather than assumes.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class RequestMapGenerator(nn.Module):
    """Turn a confidence map into a request map.

    Purpose
        Produce the control payload each agent broadcasts, saying where it
        would like collaborators to contribute.

    Inputs
    ------
    confidence  ``(L, 1, H, W)`` in [0, 1] -- ``C_i`` from the spatial
                confidence generator, after smoothing.

    Outputs
    -------
    ``(L, 1, H, W)`` in [0, 1] -- ``R_i = 1 - C_i``.

    Shapes
    ------
    Shape-preserving; the channel axis is kept at 1 so the map broadcasts
    against features without a reshape at the call site.

    Example
    -------
    >>> import torch
    >>> gen = RequestMapGenerator()
    >>> confidence = torch.tensor([[[[0.9, 0.1], [0.5, 0.0]]]])
    >>> gen(confidence)
    tensor([[[[0.1000, 0.9000],
              [0.5000, 1.0000]]]])

    A blind agent asks for everything, a confident one asks for nothing:

    >>> float(gen(torch.zeros(1, 1, 4, 4)).mean())
    1.0
    >>> float(gen(torch.ones(1, 1, 4, 4)).mean())
    0.0
    """

    def forward(self, confidence: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> torch.Tensor:
        request = 1.0 - confidence
        emit(taps, request, module="RequestMapGenerator",
             location=f"comm/r{round_index}/request_map")
        return request
