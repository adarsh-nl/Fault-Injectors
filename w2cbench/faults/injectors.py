"""
injectors.py
------------
The protocol plane: faults on Where2comm's own messages.

Why a third plane exists at all
-------------------------------
This repository's standing rule is that faults are *physical* and are applied
to raw data upstream of the model -- one corruption path, in
``src.pipeline.FaultPipeline``, reached through ``cpbench.faults``. That rule
is not relaxed here.

But Where2comm's messages carry two payloads. Features are one. The other is
the **request map**, a small control packet that steers the next round of
communication, plus the confidence map a receiver weights its fusion by. A V2X
stack that delivers the large feature packet and drops the small control packet
is a routine, physically real event, and no sensor-level injector can express
it. ``lgcpbench`` established the precedent with its control plane; this
follows it, and confines itself to the same boundary: these injectors act on a
*message*, at the moment of transmission, and every action becomes a
``FaultRecord`` exactly like a physical one.

What is deliberately not here
-----------------------------
An agent that computes one confidence internally and reports a *different* one
is an attack, not a fault. Modelling it would let an adversarial result be read
as a reliability result, and the two call for entirely different conclusions.
:class:`ConfidenceReportInjector` therefore models a *miscalibrated* agent: its
confidence is wrong, and it is wrong consistently -- the agent believes it,
selects on it, and reports it. That is a sensor-degradation consequence, not a
lie.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from ..comm.selection import INDEX_BYTES_PER_CELL, top_k_mask

logger = logging.getLogger(__name__)


class ProtocolInjector(ABC):
    """One corruption applied at one point in the message protocol.

    Attributes
    ----------
    stage  which boundary this injector acts at:

           ``"confidence"``  the map a sender selects on and reports
           ``"request"``     the control packet a receiver broadcasts
           ``"selection"``   the mask, after the strategy has chosen

    Subclasses implement :meth:`apply`, returning the corrupted tensor and a
    parameter dict for the audit trail (or None when nothing fired, so a
    probabilistic injector does not record a no-op as a fault).
    """

    stage: str = ""
    name: str = ""

    @abstractmethod
    def apply(self, tensor: torch.Tensor, *, generator: torch.Generator,
              round_index: int, **context: Any
              ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Corrupt `tensor`; return ``(tensor, params or None)``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(stage={self.stage!r})"


class RequestLossInjector(ProtocolInjector):
    """Drop a receiver's request map before senders can condition on it.

    Purpose
        Model the loss of the protocol's control packet -- the small message
        a V2X stack is most likely to drop and least likely to retransmit.

    How a lost request is represented
        By setting the affected map to **ones**, not zeros. ``R = 1``
        everywhere means "I need everything, everywhere", which makes the
        selection score ``C_i (X) R_j`` collapse to ``C_i`` -- exactly the
        unconditioned, round-0 broadcast a sender falls back to when no
        request arrived. Zeroing it would instead say "I need nothing", which
        would silence the sender completely and model a different fault.

    Provably a no-op at K=1
        With one round nobody ever consumes a request map, so this injector
        cannot change anything -- and the fault suite asserts that rather than
        assuming it. The protocol fault family only becomes meaningful under
        the multi-round config, which is a real finding about where
        Where2comm's stated mechanism is actually exercised.

    Inputs
    ------
    p_loss  probability, per agent per round, that the request is lost.

    Example
    -------
    >>> import torch
    >>> injector = RequestLossInjector(p_loss=1.0)
    >>> request = torch.full((3, 1, 4, 4), 0.2)
    >>> out, params = injector.apply(
    ...     request, generator=torch.Generator().manual_seed(0), round_index=1)
    >>> float(out.min()), params["n_lost"]
    (1.0, 3)
    """

    stage = "request"
    name = "request_loss"

    def __init__(self, p_loss: float = 0.25) -> None:
        if not 0.0 <= float(p_loss) <= 1.0:
            raise ValueError(f"p_loss must be in [0, 1], got {p_loss}")
        self.p_loss = float(p_loss)

    def apply(self, tensor, *, generator, round_index, **context):
        n_agents = tensor.shape[0]
        draw = torch.rand(n_agents, generator=generator)
        lost = draw < self.p_loss
        if not bool(lost.any()):
            return tensor, None
        out = tensor.clone()
        out[lost] = 1.0                      # "send me everything you have"
        return out, {"p_loss": self.p_loss, "n_lost": int(lost.sum()),
                     "agents": lost.nonzero().flatten().tolist(),
                     "round": round_index}


class ConfidenceReportInjector(ProtocolInjector):
    """Miscalibrate an agent's confidence map.

    Purpose
        Where2comm's whole design trusts each sender's self-assessment: it
        decides what that sender transmits and how strongly the receiver
        weights it. This measures what that trust costs when the assessment is
        wrong.

    Modes
    -----
    ``inflate``  the agent over-rates itself. It claims cells it cannot
                 actually resolve, wins bandwidth for them, and -- under a
                 confidence-weighting aggregator -- wins attention weight too,
                 injecting noise into the fused map. The realistic cause is a
                 detector that has not seen this domain, not malice.
    ``deflate``  the agent under-rates itself, withholding cells it can
                 genuinely see. Costs recall and *saves* bandwidth, which is
                 the second way this benchmark's efficiency column can improve
                 while perception degrades.

    Applied to the map the agent both selects on and reports, because a
    miscalibrated agent believes its own numbers. An agent reporting something
    different from what it computed is an attack rather than a fault; see the
    module docstring.

    Inputs
    ------
    mode        ``"inflate"`` or ``"deflate"``.
    magnitude   additive shift, clamped into [0, 1].
    p_affected  fraction of agents affected, drawn per round.

    Example
    -------
    >>> import torch
    >>> injector = ConfidenceReportInjector(mode="inflate", magnitude=0.3,
    ...                                     p_affected=1.0)
    >>> out, params = injector.apply(
    ...     torch.full((2, 1, 4, 4), 0.5),
    ...     generator=torch.Generator().manual_seed(0), round_index=0)
    >>> round(float(out.max()), 4), params["mode"]
    (0.8, 'inflate')
    """

    stage = "confidence"
    name = "confidence_report"

    def __init__(self, mode: str = "inflate", magnitude: float = 0.3,
                 p_affected: float = 0.3) -> None:
        if mode not in ("inflate", "deflate"):
            raise ValueError(
                f"mode must be 'inflate' or 'deflate', got {mode!r}")
        if not 0.0 <= float(p_affected) <= 1.0:
            raise ValueError(
                f"p_affected must be in [0, 1], got {p_affected}")
        self.mode = mode
        self.magnitude = float(magnitude)
        self.p_affected = float(p_affected)

    def apply(self, tensor, *, generator, round_index, **context):
        n_agents = tensor.shape[0]
        draw = torch.rand(n_agents, generator=generator)
        affected = draw < self.p_affected
        if not bool(affected.any()):
            return tensor, None
        shift = self.magnitude if self.mode == "inflate" else -self.magnitude
        out = tensor.clone()
        out[affected] = (out[affected] + shift).clamp(0.0, 1.0)
        return out, {"mode": self.mode, "magnitude": self.magnitude,
                     "p_affected": self.p_affected,
                     "n_affected": int(affected.sum()),
                     "agents": affected.nonzero().flatten().tolist(),
                     "round": round_index}


class BandwidthCapInjector(ProtocolInjector):
    """Truncate each link's selected cells to fit a hard byte cap.

    Purpose
        Model a congested link. Distinct from ``src``'s ``BandwidthInjector``,
        which thins the raw point cloud upstream: this one acts on the wire,
        *after* the model has decided what to send, which is the failure mode
        of a real network rather than of a sensor.

    The interaction with A1 is the point
        Under ``ThresholdSelector`` the model believes it sent a message that
        was in fact truncated -- it has no feedback and no chance to
        re-prioritise, so the cells lost are arbitrary with respect to what it
        would have chosen to keep. Under ``BudgetSelector`` the model planned
        around the same limit and this injector does nothing, because the
        selection already fits. The paper claims graceful degradation under
        varying bandwidth; this is the condition that tests the claim in the
        case where the model was *not* told.

    Cells are dropped lowest-priority first, which is the most favourable
    possible truncation. A pessimistic model would drop arbitrarily; keeping
    the optimistic one means a poor result here cannot be blamed on the
    injector.

    Inputs
    ------
    max_bytes  cap per link per round.
    channels   feature width, for the byte-to-cell conversion. Left None it is
               taken from the message tensor at call time.

    Example
    -------
    >>> import torch
    >>> injector = BandwidthCapInjector(max_bytes=132)     # exactly one cell
    >>> mask = torch.ones(2, 2, 4, 4)
    >>> priority = torch.rand(2, 2, 4, 4)
    >>> out, params = injector.apply(
    ...     mask, generator=torch.Generator().manual_seed(0), round_index=0,
    ...     priority=priority, channels=32, receiver=0)
    >>> int(out[1, 0].sum()), params["cells_allowed"]
    (1, 1)
    """

    stage = "selection"
    name = "bandwidth_cap"

    def __init__(self, max_bytes: int = 16384,
                 channels: Optional[int] = None) -> None:
        if int(max_bytes) < 0:
            raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
        self.max_bytes = int(max_bytes)
        self.channels = channels

    def apply(self, tensor, *, generator, round_index, priority=None,
              channels=None, receiver=0, bytes_per_element: int = 4,
              **context):
        width = channels if channels is not None else self.channels
        if width is None:
            raise ValueError(
                "BandwidthCapInjector needs the feature width to convert a "
                "byte cap into a cell count; pass channels=")
        per_cell = int(width) * int(bytes_per_element) + INDEX_BYTES_PER_CELL
        allowed = self.max_bytes // per_cell

        *lead, height, width_cells = tensor.shape
        n_cells = height * width_cells
        current = tensor.reshape(*lead, n_cells)
        if int(current.sum(dim=-1).max()) <= allowed:
            return tensor, None              # already fits; nothing truncated

        # Rank by the selection priority so the truncation is the most
        # favourable possible; a poor result then cannot be blamed on it.
        scores = (priority.reshape(*lead, n_cells) if priority is not None
                  else current)
        capped = top_k_mask(scores * current, allowed) * current
        out = capped.reshape(*tensor.shape)
        # The self-link never crosses a link, so it is never capped (A6).
        n_agents = out.shape[-4]
        eye = torch.eye(n_agents, dtype=torch.bool, device=out.device)
        out = torch.where(eye[:, :, None, None], tensor, out)
        return out, {"max_bytes": self.max_bytes, "cells_allowed": int(allowed),
                     "bytes_per_cell": per_cell, "round": round_index}


_INJECTORS = {
    "request_loss": RequestLossInjector,
    "confidence_report": ConfidenceReportInjector,
    "bandwidth_cap": BandwidthCapInjector,
}


def make_protocol_injector(name: str, **kwargs) -> ProtocolInjector:
    """Build a protocol injector by config name.

    >>> make_protocol_injector("request_loss", p_loss=0.5).p_loss
    0.5
    >>> make_protocol_injector("nope")
    Traceback (most recent call last):
    KeyError: "unknown protocol injector 'nope'; expected one of ['bandwidth_cap', 'confidence_report', 'request_loss']"
    """
    try:
        cls = _INJECTORS[name]
    except KeyError:
        raise KeyError(
            f"unknown protocol injector {name!r}; expected one of "
            f"{sorted(_INJECTORS)}") from None
    return cls(**kwargs)


def available_protocol_injectors() -> list:
    """Names accepted by :func:`make_protocol_injector`."""
    return sorted(_INJECTORS)
