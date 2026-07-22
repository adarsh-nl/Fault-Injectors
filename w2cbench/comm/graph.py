"""
graph.py
--------
The communication graph: which links exist at all.

    A_{i,j}^(0) = 1                                   (round 0: broadcast)
    A_{i,j}^(k) = max_{h,w} (M_{i->j}^(k))_{h,w}       (k > 0: request-driven)

Why round 0 is complete regardless of the mask
----------------------------------------------
At round 0 nobody has requested anything yet, so every agent broadcasts on the
basis of its own confidence alone. The link exists because the broadcast
happened; whether that particular sender had anything worth putting in it is a
property of the *message*, not of the *link*. From round 1 the two coincide,
because a sender only transmits where a receiver asked and it can answer -- so
an empty mask genuinely means no connection was needed.

Keeping the distinction matters for one reported number. ``comm_graph_density``
is meant to answer "how much of the possible V2X topology was actually used",
and conflating "broadcast carrying nothing" with "no broadcast" would report a
round-0 density that falls whenever the scene is empty, which has nothing to do
with connectivity.

What this is not
----------------
Not a physical connectivity model. Range limits, packet loss and agent dropout
are faults applied upstream on the raw data, and by the time a message reaches
this module it is already whatever the ego really received. This module reports
the *protocol's* graph, and the two differ in a way worth being able to see: a
dropped agent leaves the graph via ``input/agent_mask``, whereas an agent that
is present but has nothing to say leaves it via an empty selection mask.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class CommunicationGraph(nn.Module):
    """Derive the round's adjacency matrix from the selection mask.

    Purpose
        Say which ordered pairs of agents communicated, so fusion can ignore
        links that carry nothing and the accountant can report topology use.

    Inputs
    ------
    broadcast_round_zero  treat round 0 as fully connected (the paper's
                          ``A^(0) = 1``). Disable to derive every round from
                          its mask, which is the stricter reading and is kept
                          reachable for the ablation.

    Inputs (to :meth:`forward`)
    ---------------------------
    mask          ``(..., L, L, H, W)`` binary selection matrix.
    agent_mask    optional ``(L,)`` or ``(..., L)`` marking real agents; slots
                  a drop fault emptied are removed from the graph even at
                  round 0, because a broadcast to an absent agent is not a
                  link.
    round_index   selects the rule above, and names the tap.

    Outputs
    -------
    ``(..., L, L)`` float in {0, 1}.

    Example
    -------
    >>> import torch
    >>> graph = CommunicationGraph()
    >>> mask = torch.zeros(3, 3, 4, 4)
    >>> mask[1, 0, 0, 0] = 1.0                  # only agent 1 -> 0 selected
    >>> graph(mask, round_index=1)
    tensor([[0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 0.]])

    Round 0 is complete regardless, because the broadcast happened:

    >>> int(graph(mask, round_index=0).sum())
    9

    An agent that was dropped is not reachable even by a broadcast:

    >>> present = torch.tensor([True, True, False])
    >>> graph(mask, agent_mask=present, round_index=0)
    tensor([[1., 1., 0.],
            [1., 1., 0.],
            [0., 0., 0.]])
    """

    def __init__(self, broadcast_round_zero: bool = True) -> None:
        super().__init__()
        self.broadcast_round_zero = bool(broadcast_round_zero)

    def forward(self, mask: torch.Tensor,
                agent_mask: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> torch.Tensor:
        if mask.dim() < 4 or mask.shape[-4] != mask.shape[-3]:
            raise ValueError(
                f"expected a square agent block (..., L, L, H, W); got "
                f"{tuple(mask.shape)}")

        if round_index == 0 and self.broadcast_round_zero:
            graph = torch.ones(mask.shape[:-2], dtype=mask.dtype,
                               device=mask.device)
        else:
            graph = (mask.amax(dim=(-2, -1)) > 0).to(mask.dtype)

        if agent_mask is not None:
            present = agent_mask.to(torch.bool)
            # A link needs both ends: sender present AND receiver present.
            graph = graph * (present.unsqueeze(-1) * present.unsqueeze(-2)
                             ).to(graph.dtype)

        emit(taps, graph, module="CommunicationGraph",
             location=f"comm/r{round_index}/comm_graph")
        return graph

    def extra_repr(self) -> str:
        return f"broadcast_round_zero={self.broadcast_round_zero}"


def incoming_links(graph: torch.Tensor, receiver: int = 0) -> tuple:
    """``(realised, possible)`` incoming links for one receiver.

    This, not the full matrix, is what an ego-centric deployment should
    report. Where2comm fuses for one receiver at a time, so only column
    ``receiver`` is ever used -- and density over all ``L*(L-1)`` ordered pairs
    would be structurally capped at ``1/(L-1)``. With ``L=5`` a graph in which
    every collaborator reached the ego would report 0.2, reading as "80% of
    the topology unused" when in fact all of the *used* topology was active.

    Excludes the self-link, which is not a link.

    Example
    -------
    >>> import torch
    >>> graph = torch.zeros(4, 4)
    >>> graph[1, 0] = graph[2, 0] = 1.0          # two of three reached the ego
    >>> incoming_links(graph, receiver=0)
    (2.0, 3.0)
    """
    n_agents = graph.shape[-1]
    senders = [i for i in range(n_agents) if i != receiver]
    if not senders:
        return 0.0, 0.0
    return float(graph[..., senders, receiver].sum()), float(len(senders))


def graph_density(graph: torch.Tensor, include_self: bool = False) -> float:
    """Fraction of possible links that exist.

    Self-links are excluded by default: an agent is trivially connected to
    itself, and counting ``L`` guaranteed links would put a floor under the
    density that rises as the agent count falls -- so a scene losing
    collaborators would report *higher* connectivity.

    Example
    -------
    >>> import torch
    >>> full = torch.ones(3, 3)
    >>> graph_density(full)
    1.0
    >>> graph_density(torch.eye(3))          # only self-links: no real links
    0.0
    """
    n_agents = graph.shape[-1]
    if include_self:
        return float(graph.sum() / max(graph.numel(), 1))
    off_diagonal = graph * (1.0 - torch.eye(n_agents, dtype=graph.dtype,
                                            device=graph.device))
    possible = graph[..., 0, 0].numel() * n_agents * (n_agents - 1)
    return float(off_diagonal.sum() / possible) if possible else 0.0
