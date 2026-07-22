"""
protocol.py
-----------
The protocol-plane bridge: one place messages get corrupted, mirroring the
physical plane's ``cpbench.faults.DataFaultBridge``.

Structural parity with the physical bridge is deliberate. Both are constructed
from a config dict, both are provably identity when unconfigured, both
accumulate ``FaultRecord``s that drain into ``injection_summary.csv``, and both
expose ``is_clean``. A reader who understands one understands the other, and
more importantly a *result* does not distinguish them: a protocol fault appears
in the same audit trail, with the same columns, as a fogged LiDAR.

The boundary this is confined to
--------------------------------
Three hooks, called from :class:`~w2cbench.models.where2comm.Where2comm` and
nowhere else:

    confidence  after the generator, before selection reads it
    request     after ``R = 1 - C``, before senders condition on it
    selection   after the strategy has chosen, before packing

Every one of them is a *message* in the protocol: something an agent computed
in order to transmit it. None of them is an arbitrary activation, which is the
line this plane does not cross.

Determinism
-----------
One ``torch.Generator``, seeded once, threaded through every injector. Two runs
of the same config corrupt the same agents in the same rounds -- without which
a clean-versus-faulted comparison would be measuring the difference between two
random draws as much as the effect of the fault.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from cpbench.faults import FaultRecord

from .injectors import ProtocolInjector, make_protocol_injector

logger = logging.getLogger(__name__)

STAGES = ("confidence", "request", "selection")


class ProtocolFaultBridge:
    """Apply protocol-plane faults at the message boundary.

    Purpose
        Corrupt the control payloads Where2comm's protocol depends on, with
        the same bookkeeping discipline as a physical fault.

    Inputs
    ------
    injectors  a sequence of :class:`ProtocolInjector`, or None/empty for a
               provably clean bridge -- no injector exists to fire, rather
               than injectors configured to do nothing.
    seed       master seed for the shared generator.

    Outputs
    -------
    ``apply(stage, tensor, ...)`` returns the (possibly corrupted) tensor.
    ``drain_records()`` yields the audit trail.

    Example
    -------
    >>> import torch
    >>> bridge = ProtocolFaultBridge.from_config(
    ...     {"request_loss": {"p_loss": 1.0}}, seed=0)
    >>> request = torch.full((2, 1, 4, 4), 0.2)
    >>> float(bridge.apply("request", request, round_index=1).min())
    1.0
    >>> [r.fault_type for r in bridge.drain_records()]
    ['request_loss']

    An unconfigured bridge is the identity, and says so:

    >>> clean = ProtocolFaultBridge.from_config(None)
    >>> clean.is_clean
    True
    >>> clean.apply("request", request) is request
    True
    """

    def __init__(self, injectors: Optional[Sequence[ProtocolInjector]] = None,
                 seed: int = 0) -> None:
        self.injectors: List[ProtocolInjector] = list(injectors or [])
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.records: List[FaultRecord] = []
        for injector in self.injectors:
            if injector.stage not in STAGES:
                raise ValueError(
                    f"{type(injector).__name__} declares stage "
                    f"{injector.stage!r}; expected one of {STAGES}")
        logger.info("ProtocolFaultBridge(seed=%d, injectors=%s)", self.seed,
                    [i.name for i in self.injectors] or "clean")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]],
                    seed: int = 0) -> "ProtocolFaultBridge":
        """Build from a ``protocol_pipeline`` config block.

        ``None`` or ``{}`` produces a bridge with no injectors at all. That
        distinction matters for the same reason it does on the physical plane:
        a reference condition that quietly injected something would make every
        comparison against it meaningless.
        """
        config = dict(config or {})
        injectors = [make_protocol_injector(name, **dict(params or {}))
                     for name, params in config.items()]
        return cls(injectors, seed=seed)

    @property
    def is_clean(self) -> bool:
        """True when no injector exists to fire."""
        return not self.injectors

    # -- the boundary -------------------------------------------------------

    def apply(self, stage: str, tensor: torch.Tensor, *,
              round_index: int = 0, frame: int = -1,
              **context: Any) -> torch.Tensor:
        """Run every injector registered for `stage`.

        Returns `tensor` unchanged -- the same object, not a copy -- when
        nothing is configured, so a clean run costs one list check per hook.
        """
        if stage not in STAGES:
            raise ValueError(f"unknown protocol stage {stage!r}; "
                             f"expected one of {STAGES}")
        for injector in self.injectors:
            if injector.stage != stage:
                continue
            tensor, params = injector.apply(
                tensor, generator=self.generator, round_index=round_index,
                **context)
            if params is not None:
                self.records.append(FaultRecord(
                    frame=frame, agent_id="*", fault_type=injector.name,
                    target=stage, params=params))
        return tensor

    def drain_records(self) -> List[FaultRecord]:
        """Return accumulated records and reset the buffer."""
        out, self.records = self.records, []
        return out

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed and clear, so a condition starts from a known state."""
        self.generator = torch.Generator().manual_seed(
            self.seed if seed is None else int(seed))
        self.records = []

    def __repr__(self) -> str:
        return (f"ProtocolFaultBridge(injectors="
                f"{[i.name for i in self.injectors] or 'clean'}, "
                f"seed={self.seed})")
