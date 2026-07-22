"""
comms.py
--------
Communication-volume metrics: how much crossed the V2X link, and how that
number moves under faults.

Why this is a metric and not profiling
--------------------------------------
For most collaborative-perception models the transmitted volume is fixed by
the architecture and is a footnote. For bandwidth-aware models -- Where2comm
and its descendants -- it is decided *per frame, by the model, from the
input*, which makes it a result rather than a machine property. It also makes
it fault-sensitive in a way that is easy to misread: a sensor degradation that
flattens a confidence map causes fewer cells to be selected, so the fault
*lowers* measured bandwidth while lowering accuracy. Reported alone, that
column says the system got more efficient. Reported next to AP, it says the
system started failing.

That is the whole reason this module exists: so a benchmark can put the two
numbers in the same row.

Where the bytes come from
-------------------------
``cpbench.comms.MessageChannel`` counts them at the moment of transmission,
already accounting for sparsity (non-zero cells x channels + indices), binary
request masks (1 bit per cell) and the transmission precision assumed by the
paper. This module never re-derives a byte count; it aggregates what the
channel measured. Precision assumptions therefore live in exactly one place.

log2 of the mean, not the mean of the log2
------------------------------------------
The Where2comm-family x-axis is ``log2(bytes)``. Because log is concave,
``mean(log2(b_i)) <= log2(mean(b_i))`` (Jensen), and the two disagree by more
the more variable the per-frame volume is -- which is precisely the regime a
fault benchmark operates in. The published figures plot one point per
configuration against its *average* communication volume, so the average is
taken first and the log second. ``log2_bytes_of_mean`` is that number and is
what belongs on a reproduction plot; ``mean_log2_bytes`` is also reported,
because the gap between them is itself a useful signal that a condition has
made the per-frame volume erratic.

Example
-------
>>> m = CommVolumeMetrics()
>>> m.add_frame(FrameComms(frame=0, bytes_by_location={"comm/r0/sent": 1024}))
>>> m.add_frame(FrameComms(frame=1, bytes_by_location={"comm/r0/sent": 3072}))
>>> out = m.compute()
>>> out["bytes_per_frame"], out["log2_bytes"]
(2048.0, 11.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

_BYTES_PER_MB = 2 ** 20


def log2_bytes(nbytes: float) -> float:
    """log2(bytes), with zero reported as NaN rather than -inf or 0.

    The Where2comm-family x-axis. Module-level so plotting code, the metrics
    accumulator and any paper package share one definition -- a second
    implementation would be a second convention for the zero case.

    Zero bytes has no logarithm, and both plausible substitutes lie: -inf
    poisons every downstream mean and breaks CSV round-tripping, while 0.0 is
    indistinguishable from "one byte was sent". NaN is the honest answer and
    is what plotting libraries already skip; the raw count survives alongside
    it in ``bytes_per_frame``.

    >>> log2_bytes(1024)
    10.0
    >>> math.isnan(log2_bytes(0))
    True
    """
    return math.log2(nbytes) if nbytes > 0 else float("nan")


@dataclass
class FrameComms:
    """One frame's transmitted volume and the protocol state that produced it.

    Purpose
        The communication analogue of ``FramePair``: a per-frame record the
        accumulator consumes, so the tester does no arithmetic of its own.

    Attributes
    ----------
    frame               frame index, for traceability.
    bytes_by_location   bytes per tap location, exactly as
                        ``CommLog.bytes_by_location`` holds them. Keys are
                        full location names; this module aggregates them by
                        their final path segment, so ``comm/r0/sent`` and
                        ``comm/r1/sent`` sum into one ``mb_sent`` column
                        rather than producing a column per round.
    messages            number of messages sent this frame.
    selected_cells      mean non-zero cells per transmitting link. Optional:
                        only models with an explicit selection step have it.
    cells_per_map       H*W, the denominator of the selection ratio.
    graph_links         communication links that actually existed.
    graph_possible      links that could have existed (L*(L-1), or L*L if the
                        model counts self-links).
    rounds              communication rounds executed for this frame.

    ``None`` is used rather than 0.0 for the optional fields because "this
    model has no selection step" and "this model selected nothing" are
    different facts, and averaging the second into a mean is correct while
    averaging the first is not.

    Shapes
    ------
    All scalars; the tensors they summarise never reach this module.
    """

    frame: int
    bytes_by_location: Mapping[str, int] = field(default_factory=dict)
    messages: int = 0
    selected_cells: Optional[float] = None
    cells_per_map: Optional[int] = None
    graph_links: Optional[int] = None
    graph_possible: Optional[int] = None
    rounds: int = 1

    @property
    def total_bytes(self) -> int:
        return int(sum(self.bytes_by_location.values()))

    @classmethod
    def from_comm_log(cls, log: Any, frame: int, **extras: Any) -> "FrameComms":
        """Build from a ``cpbench.comms.CommLog`` snapshot.

        Duck-typed on ``bytes_by_location`` and ``messages`` rather than
        importing ``cpbench.comms``, so the metrics package stays independent
        of the transport package and either can be used without the other.

        The log must cover ONE frame: reset it (or construct a fresh channel)
        per frame, or every frame after the first will re-count its
        predecessors' bytes.

        Example
        -------
        >>> class _Log:                       # stands in for CommLog
        ...     bytes_by_location = {"comm/r0/sent": 512}
        ...     messages = 2
        >>> FrameComms.from_comm_log(_Log(), frame=3).total_bytes
        512
        """
        return cls(frame=frame,
                   bytes_by_location=dict(log.bytes_by_location),
                   messages=int(getattr(log, "messages", 0)),
                   **extras)


class CommVolumeMetrics:
    """Accumulate per-frame volumes; compute the reported communication row.

    Purpose
        Turn a run's worth of :class:`FrameComms` into the ``comm_*`` columns
        of an ``EvalRecord``.

    Inputs
        :meth:`add_frame` per evaluated frame.

    Outputs
        :meth:`compute` returns a flat ``Dict[str, float]``. Keys, before the
        ``comm_`` prefix the record adds:

        ============================  ==================================
        ``bytes_total``               every byte the run transmitted
        ``bytes_per_frame``           mean bytes per frame
        ``log2_bytes``                log2 of the mean -- the paper's axis
        ``mean_log2_bytes``           mean of the per-frame logs (see above)
        ``mb_total`` / ``mb_per_frame``   the same in mebibytes
        ``mb_<suffix>``               per message type, summed over rounds
        ``n_messages``                messages sent
        ``messages_per_frame``        mean messages per frame
        ``selected_cells_mean``       mean selected cells per link
        ``rate``                      selected cells / (H*W); the ratio the
                                      Where2comm reference reports
        ``graph_density``             realised links / possible links
        ``rounds``                    mean rounds executed
        ``n_frames``                  frames accumulated
        ============================  ==================================

        Optional keys are omitted entirely when no frame supplied them,
        rather than emitted as zero: a model without a selection step should
        produce no ``comm_rate`` column, not a column of zeros that reads as
        "selected nothing".

    Example
    -------
    >>> m = CommVolumeMetrics()
    >>> m.add_frame(FrameComms(0, {"comm/r0/sent": 800, "comm/r0/request_sent": 224},
    ...                        messages=4, selected_cells=64, cells_per_map=256,
    ...                        graph_links=6, graph_possible=12))
    >>> out = m.compute()
    >>> out["bytes_per_frame"], out["rate"], out["graph_density"]
    (1024.0, 0.25, 0.5)
    >>> round(out["mb_sent"], 6), round(out["mb_request_sent"], 6)
    (0.000763, 0.000214)

    An empty run is reported honestly rather than as a zero-byte run:

    >>> CommVolumeMetrics().compute()
    {'n_frames': 0.0}
    """

    def __init__(self) -> None:
        self.frames: List[FrameComms] = []

    def add_frame(self, frame: FrameComms) -> None:
        self.frames.append(frame)

    def reset(self) -> None:
        self.frames = []

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _suffix(location: str) -> str:
        """Message type from a tap location: ``comm/r1/sent`` -> ``sent``.

        Collapsing the round index is deliberate. A run with K=3 would
        otherwise produce three ``mb_sent`` columns whose names encode a
        config value, so two runs at different K could not be compared with a
        single CSV read. Per-round detail is available in ``taps.csv``, which
        is where per-layer breakdowns belong.

        >>> CommVolumeMetrics._suffix("comm/r1/request_sent")
        'request_sent'
        >>> CommVolumeMetrics._suffix("sent")
        'sent'
        """
        return location.rsplit("/", 1)[-1]

    def _mean(self, attr: str) -> Optional[float]:
        """Mean of an optional per-frame field, over the frames that have it."""
        values = [getattr(f, attr) for f in self.frames
                  if getattr(f, attr) is not None]
        return sum(values) / len(values) if values else None

    def _ratio(self, numerator: str, denominator: str) -> Optional[float]:
        """Sum-over-sum ratio, not mean-of-ratios.

        A frame in which nothing could have been transmitted (no
        collaborators, so no possible links) would otherwise contribute a 0/0
        that either crashes or, if guarded to zero, drags the average down as
        though the model had failed to communicate when it had nothing to
        communicate with.
        """
        num = sum(getattr(f, numerator) for f in self.frames
                  if getattr(f, numerator) is not None
                  and getattr(f, denominator))
        den = sum(getattr(f, denominator) for f in self.frames
                  if getattr(f, numerator) is not None
                  and getattr(f, denominator))
        return num / den if den else None

    # -- computation --------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        n_frames = len(self.frames)
        if n_frames == 0:
            return {"n_frames": 0.0}

        by_type: Dict[str, int] = {}
        for frame in self.frames:
            for location, nbytes in frame.bytes_by_location.items():
                key = self._suffix(location)
                by_type[key] = by_type.get(key, 0) + int(nbytes)

        total_bytes = sum(by_type.values())
        per_frame = total_bytes / n_frames
        n_messages = sum(f.messages for f in self.frames)

        out: Dict[str, float] = {
            "bytes_total": float(total_bytes),
            "bytes_per_frame": per_frame,
            "log2_bytes": log2_bytes(per_frame),
            "mean_log2_bytes": sum(log2_bytes(f.total_bytes)
                                   for f in self.frames) / n_frames,
            "mb_total": total_bytes / _BYTES_PER_MB,
            "mb_per_frame": per_frame / _BYTES_PER_MB,
            "n_messages": float(n_messages),
            "messages_per_frame": n_messages / n_frames,
            "rounds": sum(f.rounds for f in self.frames) / n_frames,
            "n_frames": float(n_frames),
        }
        for key, nbytes in sorted(by_type.items()):
            out[f"mb_{key}"] = nbytes / _BYTES_PER_MB

        optional = {
            "selected_cells_mean": self._mean("selected_cells"),
            "rate": self._ratio("selected_cells", "cells_per_map"),
            "graph_density": self._ratio("graph_links", "graph_possible"),
        }
        out.update({k: v for k, v in optional.items() if v is not None})
        return out
