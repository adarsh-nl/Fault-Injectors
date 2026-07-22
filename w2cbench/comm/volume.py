"""
volume.py
---------
Per-frame communication bookkeeping: the bridge from the model's decisions to
the ``comm_*`` columns of an ``EvalRecord``.

The division of labour
----------------------
``cpbench.comms.MessageChannel`` counts bytes at the moment of transmission.
``cpbench.metrics.CommVolumeMetrics`` aggregates frames into the reported row.
This module sits between them and knows the one thing neither can: the shape of
a Where2comm frame, which is *several rounds*, each with its own selection mask
and adjacency, all belonging to one frame's volume.

Why it is not an ``nn.Module``
------------------------------
It holds run-scoped mutable state -- accumulating bytes, cell counts, round
counts. The benchmark runner reuses one model across every fault condition, so
state living on the model would have each condition's volume contaminated by
the last, silently and in a direction that always looks like more traffic. An
accountant is constructed per run and passed in.

It refuses to report in training mode (A17)
-------------------------------------------
The released selector ignores its configured rule during training and keeps a
random fraction of the map instead -- the paper's bandwidth curriculum. A
volume measured under it is a draw from that curriculum, not a model decision,
and it would be wrong by a random factor between roughly 0.1 and 1.0 with
nothing to indicate it. ``lgcpbench``'s OpenCOOD adapter refuses to measure in
train mode for the same reason; this one raises rather than returning a
plausible wrong number.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

from cpbench.comms.channel import MessageChannel
from cpbench.metrics import CommVolumeMetrics, FrameComms
from cpbench.observation import TapProtocol, emit

from .graph import incoming_links
from .packing import message_statistics

logger = logging.getLogger(__name__)


class CommVolumeAccountant:
    """Accumulate one run's communication volume, frame by frame.

    Purpose
        Own the channel, tally per-round protocol statistics, and hand the
        benchmark a ``comm_*`` dict that sits beside AP in the same row.

    Inputs
    ------
    bytes_per_element  transmission precision. Defaults to 4 (A8), matching
                       the paper's ``log2(|M| * D * 32/8)`` -- not the
                       ``MessageChannel`` default of 2, which suits papers
                       assuming half-precision links. Getting this wrong
                       shifts every point on the log2 axis by exactly 1.0.
    taps               forwarded to the channel, so in-flight messages are
                       observable at ``comm/r{k}/sent``.

    Usage
    -----
    ::

        accountant = CommVolumeAccountant(bytes_per_element=4)
        for frame in frames:
            accountant.start_frame()
            out = model(batch, accountant=accountant)     # rounds record here
            accountant.end_frame(frame, training=model.training)
        record.comms = accountant.compute()

    Example
    -------
    >>> import torch
    >>> acc = CommVolumeAccountant(bytes_per_element=4)
    >>> acc.start_frame()
    >>> mask = torch.zeros(3, 3, 4, 4); mask[1, 0, 0, 0] = 1.0
    >>> graph = torch.ones(3, 3)
    >>> acc.record_round(mask, graph, receiver=0)
    >>> _ = acc.channel.send(torch.ones(8, 4, 4) * mask[1, 0], sender="a",
    ...                      receiver="b", location="comm/r0/sent", sparse=True)
    >>> _ = acc.end_frame(0)
    >>> out = acc.compute()
    >>> out["bytes_per_frame"], out["rounds"], out["n_frames"]
    (36.0, 1.0, 1.0)
    >>> round(out["rate"], 4)              # 0.5 cells of 16, averaged over 2 links
    0.0312
    """

    def __init__(self, bytes_per_element: int = 4,
                 taps: Optional[TapProtocol] = None) -> None:
        self.channel = MessageChannel(bytes_per_element=bytes_per_element,
                                      taps=taps)
        self.metrics = CommVolumeMetrics()
        self.taps = taps
        self._frame_open = False
        self._reset_frame()
        logger.info("CommVolumeAccountant(bytes_per_element=%d)",
                    bytes_per_element)

    # -- per-frame lifecycle ------------------------------------------------

    def _reset_frame(self) -> None:
        self._selected: list = []
        self._cells_per_map = 0.0
        self._links = 0.0
        self._possible = 0.0
        self._rounds = 0

    def start_frame(self) -> None:
        """Begin a frame; clears the channel so bytes are per-frame.

        Without the reset every frame after the first would re-count its
        predecessors' bytes, and the per-frame mean would grow linearly with
        the frame index -- a bug that looks exactly like a memory leak in the
        protocol.
        """
        self.channel.log.reset()
        self.channel.new_frame()
        self._reset_frame()
        self._frame_open = True

    def record_round(self, mask: torch.Tensor, graph: torch.Tensor,
                     receiver: int = 0, round_index: int = 0) -> None:
        """Tally one communication round's protocol statistics.

        Byte counting is the channel's job and happens during packing; this
        records the things bytes alone cannot express -- how many cells were
        selected, and how much of the possible topology carried them.
        """
        if not self._frame_open:
            raise RuntimeError(
                "record_round called outside a frame; call start_frame() "
                "first, or every round would be attributed to the previous "
                "frame's volume")
        stats = message_statistics(mask, receiver=receiver)
        self._selected.append(stats["selected_cells"])
        self._cells_per_map = stats["cells_per_map"]
        # Incoming links only. Density over the full pairwise matrix would be
        # structurally capped at 1/(L-1) for an ego-centric model, so a graph
        # in which every collaborator reached the ego would report 0.2 at L=5.
        realised, possible = incoming_links(graph, receiver=receiver)
        self._links += realised
        self._possible += possible
        self._rounds += 1

        emit(self.taps, torch.as_tensor(stats["selected_cells"]),
             module="CommVolumeAccountant",
             location=f"comm/r{round_index}/comm_rate")
        emit(self.taps, torch.as_tensor(float(self.channel.log.total_bytes)),
             module="CommVolumeAccountant",
             location=f"comm/r{round_index}/bytes")

    def end_frame(self, frame_index: int, training: bool = False) -> FrameComms:
        """Close the frame and fold it into the run's metrics.

        `training` is the model's mode, and a True value raises: see A17 in
        the module docstring. It is a parameter rather than something sniffed
        from a model because the accountant never holds a model reference --
        which is what keeps it usable in tests with no model at all.
        """
        if training:
            raise RuntimeError(
                "refusing to record communication volume in training mode: "
                "the selector keeps a random fraction of the map during "
                "training (the paper's bandwidth curriculum, A17), so the "
                "measured volume is a draw from that curriculum rather than "
                "a model decision. Call model.eval() before measuring.")
        if not self._frame_open:
            raise RuntimeError("end_frame called without a matching start_frame")

        selected = (sum(self._selected) / len(self._selected)
                    if self._selected else None)
        frame = FrameComms(
            frame=frame_index,
            bytes_by_location=dict(self.channel.log.bytes_by_location),
            messages=self.channel.log.messages,
            selected_cells=selected,
            cells_per_map=int(self._cells_per_map) or None,
            graph_links=int(self._links) or None,
            graph_possible=int(self._possible) or None,
            rounds=max(self._rounds, 1))
        self.metrics.add_frame(frame)
        self._frame_open = False
        return frame

    # -- reporting ----------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        """The ``comm_*`` payload for an ``EvalRecord`` (before the prefix)."""
        return self.metrics.compute()

    def reset(self) -> None:
        """Forget every frame; used between benchmark conditions."""
        self.channel.log.reset()
        self.metrics.reset()
        self._reset_frame()
        self._frame_open = False

    def __repr__(self) -> str:
        return (f"CommVolumeAccountant(bytes_per_element="
                f"{self.channel.bytes_per_element}, "
                f"frames={len(self.metrics.frames)})")
