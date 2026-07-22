"""
packing.py
----------
Turn a selection mask into the message that crosses the link, and charge it.

    Z_{i->j}^(k) = M_{i->j}^(k) (X) F_i^(k)          (paper section 4.3)

Dense in storage, sparse on the wire
------------------------------------
``Z`` is a full-size feature map in which the unselected cells are zero. The
sparsity is *semantic*: only ``cpbench.comms.MessageChannel`` reads those zeros
as un-sent, and it charges accordingly (non-zero cells x channels x precision,
plus index bytes). Keeping the tensor dense means fusion is a dense op, which
is what a GPU wants; the paper's compression claim is about what crosses a
radio link, not about GPU memory.

Why packing is receiver-indexed and the mask is not
---------------------------------------------------
The paper writes the message set pairwise, ``Z_{i->j}`` for every ordered pair.
Materialising that is not viable at paper scale: at OPV2V's ``L=5, D=256,
100x252`` the pairwise tensor is **615 MB per round** in fp32, forward-only,
against **123 MB** for the messages one receiver actually consumes -- and the
released implementation is ego-centric for exactly this reason, computing one
mask per agent rather than one per pair.

So the two are split by cost. The *selection matrix* stays fully pairwise
because it is cheap (2.4 MB at the same scale) and because it is the tensor the
benchmark wants to observe -- ``comm/r{k}/selection_mask`` should show what
every agent would send to every other, including the links the ego never uses.
The *messages* are packed one receiver at a time, which is all fusion consumes.
Nothing is lost analytically; a caller that genuinely wants every pair loops
over receivers.

What is charged, and what is not
--------------------------------
*The self-link is free.* ``Z_{i->i}`` never crosses a link: the receiver's own
features are already local. Charging them would inflate every reported volume
by one agent's worth of features and would make the paper's compression ratio
look worse than it is. A6 forces the self-link mask to ones precisely because
those cells are free, so counting them would be doubly wrong.

*The request map is charged once per sender, not once per link.* ``R_i`` does
not depend on the receiver, so a real radio broadcasts it once. Charging it
``L-1`` times would scale the control-plane bytes with the agent count and make
"request maps are cheap" look false when it is not. It is charged densely at
the configured precision, which is the conservative reading -- a deployed
system would quantise a soft mask hard -- and it is charged only when a later
round will actually consume it, because an unconsumed message is not a message.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import nn

from cpbench.comms.channel import MessageChannel
from cpbench.observation import TapProtocol, emit

logger = logging.getLogger(__name__)


class MessagePacker(nn.Module):
    """Apply a selection mask to features and account for the result.

    Purpose
        Produce the tensor fusion consumes, and -- when given a channel --
        make the bytes it would cost a measured quantity rather than an
        estimate.

    Inputs (to :meth:`forward`)
    ---------------------------
    mask       ``(L, L, H, W)`` binary selection matrix from a
               :class:`~w2cbench.comm.selection.Selector`.
    features   ``(L, D, H, W)`` per-agent BEV maps, ``F^(k)``.
    receiver   index of the receiving agent; 0 by convention, the ego slot in
               every collator in this repository.
    channel    optional :class:`MessageChannel`. Passed per call rather than
               held as state because a channel accumulates run-scoped byte
               counts, and the benchmark runner reuses one model across every
               fault condition -- an accumulator living on the module would
               have each condition's volume contaminated by the last.
    graph      optional ``(L, L)`` adjacency; links it marks absent are not
               charged. Without it every non-self link is charged.
    request    optional ``(L, 1, H, W)`` request maps to broadcast, or None
               when no later round will consume them.

    Outputs
    -------
    ``(L, D, H, W)`` -- the messages arriving at `receiver`, with the
    receiver's own row left unmasked (A6).

    Taps emitted
    ------------
    ``comm/r{k}/message_sparse``; the channel additionally emits
    ``comm/r{k}/sent`` per message and ``comm/r{k}/request_sent`` per
    broadcast request.

    Example
    -------
    >>> import torch
    >>> from cpbench.comms.channel import MessageChannel
    >>> packer = MessagePacker()
    >>> features = torch.ones(3, 4, 2, 2)
    >>> mask = torch.zeros(3, 3, 2, 2)
    >>> mask[:, 0, 0, 0] = 1.0            # every agent sends one cell to agent 0
    >>> channel = MessageChannel(bytes_per_element=4)
    >>> messages = packer(mask, features, receiver=0, channel=channel)
    >>> messages.shape
    torch.Size([3, 4, 2, 2])
    >>> int(messages[1].sum())            # one cell of four channels survived
    4

    Two senders crossed the link; the receiver's own row was free:

    >>> channel.log.messages
    2
    >>> channel.log.total_bytes           # 2 x (1 cell x 4 ch x 4 B + 4 B idx)
    40
    """

    def forward(self, mask: torch.Tensor, features: torch.Tensor,
                receiver: int = 0,
                channel: Optional[MessageChannel] = None,
                graph: Optional[torch.Tensor] = None,
                request: Optional[torch.Tensor] = None,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> torch.Tensor:
        """Pack one receiver's incoming messages and charge for them."""
        if mask.dim() != 4 or features.dim() != 4:
            raise ValueError(
                f"expected mask (L, L, H, W) and features (L, D, H, W); got "
                f"{tuple(mask.shape)} and {tuple(features.shape)}")
        if mask.shape[0] != features.shape[0]:
            raise ValueError(
                f"mask covers {mask.shape[0]} agents but features hold "
                f"{features.shape[0]}")

        link = mask[:, receiver].unsqueeze(1)          # (L, 1, H, W)
        messages = link * features                     # (L, D, H, W)
        emit(taps, messages, module="MessagePacker",
             location=f"comm/r{round_index}/message_sparse")

        if channel is not None:
            self._charge(messages, receiver, channel, graph, request,
                         round_index)
        return messages

    @staticmethod
    def _charge(messages: torch.Tensor, receiver: int,
                channel: MessageChannel, graph: Optional[torch.Tensor],
                request: Optional[torch.Tensor], round_index: int) -> None:
        """Send every real message through the byte counter.

        Skips the self-link (already local) and any link the graph marks
        absent (nothing was selected for it, so nothing was transmitted).
        """
        for sender in range(messages.shape[0]):
            if sender == receiver:
                continue
            if graph is not None and float(graph[sender, receiver]) == 0.0:
                continue
            channel.send(messages[sender], sender=f"agent{sender}",
                         receiver=f"agent{receiver}",
                         location=f"comm/r{round_index}/sent", sparse=True,
                         round_index=round_index)

        if request is not None:
            # One broadcast per sender: R_i is identical for every receiver.
            for sender in range(request.shape[0]):
                channel.send(request[sender], sender=f"agent{sender}",
                             receiver="broadcast",
                             location=f"comm/r{round_index}/request_sent",
                             round_index=round_index)


def message_statistics(mask: torch.Tensor, receiver: int = 0) -> Dict[str, float]:
    """Per-frame selection statistics for one receiver, for the accountant.

    Excludes the self-link, which A6 forces to ones and which never crosses a
    link -- including it would report a selection ratio inflated by a full map
    and would hide a collapse in what collaborators actually sent.

    Inputs
    ------
    mask  ``(L, L, H, W)`` binary selection matrix.

    Outputs
    -------
    ``{"selected_cells": mean cells per incoming link,
       "cells_per_map": H*W, "n_links": incoming links counted}``

    Example
    -------
    >>> import torch
    >>> mask = torch.zeros(3, 3, 4, 4)
    >>> mask[1, 0, :2, :2] = 1.0          # agent 1 -> 0 sends 4 cells
    >>> mask[2, 0, 0, 0] = 1.0            # agent 2 -> 0 sends 1 cell
    >>> mask[0, 0] = 1.0                  # the self-link, which is free
    >>> stats = message_statistics(mask, receiver=0)
    >>> stats["selected_cells"], stats["cells_per_map"], stats["n_links"]
    (2.5, 16.0, 2.0)
    """
    n_agents, _, height, width = mask.shape
    senders = [i for i in range(n_agents) if i != receiver]
    if not senders:
        return {"selected_cells": 0.0, "cells_per_map": float(height * width),
                "n_links": 0.0}
    incoming = mask[senders, receiver]                 # (L-1, H, W)
    return {"selected_cells": float(incoming.sum(dim=(-2, -1)).mean()),
            "cells_per_map": float(height * width),
            "n_links": float(len(senders))}
