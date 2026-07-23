"""
metadata.py
-----------
The metadata-plane bridge: one place batch metadata gets corrupted,
mirroring the physical plane's ``cpbench.faults.DataFaultBridge`` and
w2cbench's ``ProtocolFaultBridge``.

Structural parity with those bridges is deliberate. All are constructed from
a config dict, all are provably identity when unconfigured, all accumulate
``FaultRecord``s that drain into ``injection_summary.csv``, and all expose
``is_clean``. A reader who understands one understands the others -- and a
*result* does not distinguish them: a metadata fault appears in the same
audit trail, with the same columns, as a fogged LiDAR.

The boundary this is confined to
--------------------------------
One hook, ``apply_to_batch``, called by the evaluation tester after collation
and before ``model.forward`` -- and nowhere else. That placement is a
documented deviation from w2cbench, whose protocol hooks live inside the
model's forward: Where2comm's messages only *exist* mid-forward, while
V2X-ViT's corruptible metadata all sits in the collated batch dict, so
corrupting it post-collate reaches exactly the same tensors without giving
the model any fault-aware code path. Evaluation-only by construction:
training never sees this bridge, so a benchmarked model was never fitted to
its own fault distribution.

The ego row is restored centrally after every injector: the ego's metadata
never crossed a link, so no transport fault can corrupt it, and enforcing
that here means a buggy injector cannot silently break the invariant.

Determinism
-----------
One ``torch.Generator``, seeded once, threaded through every injector. Two
runs of the same config corrupt the same agents in the same frames --
without which a clean-versus-faulted comparison would be measuring the
difference between two random draws as much as the effect of the fault.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from cpbench.faults import FaultRecord

from v2xvitbench.faults.injectors import (STAGES, MetadataInjector,
                                          make_metadata_injector)

logger = logging.getLogger(__name__)

#: stage name -> the collated-batch key it corrupts
STAGE_KEYS = {"time_delay": "time_delay", "agent_types": "infra",
              "poses": "T_agent_to_ego", "prior": "velocity"}


class MetadataFaultBridge:
    """Apply metadata-plane faults to a collated batch.

    Purpose
        Corrupt the V2X metadata V2X-ViT's robustness mechanisms consume,
        with the same bookkeeping discipline as a physical fault.

    Inputs
    ------
    injectors  a sequence of :class:`MetadataInjector`, or None/empty for a
               provably clean bridge -- no injector exists to fire, rather
               than injectors configured to do nothing.
    seed       master seed for the shared generator.

    Outputs
    -------
    ``apply_to_batch(batch, frame)`` returns the batch with corrupted
    metadata (a shallow copy; corrupted fields are fresh tensors, the rest
    are shared). ``drain_records()`` yields the audit trail.

    Example
    -------
    >>> import torch
    >>> bridge = MetadataFaultBridge.from_config(
    ...     {"delay_encoding": {"mode": "stale", "magnitude_frames": 2}},
    ...     seed=0)
    >>> batch = {"time_delay": torch.tensor([[0, 1, 0]]),
    ...          "infra": torch.tensor([[0, 0, 1]])}
    >>> out = bridge.apply_to_batch(batch, frame=7)
    >>> out["time_delay"].tolist()          # ego untouched, rest + 2
    [[0, 3, 2]]
    >>> [r.fault_type for r in bridge.drain_records()]
    ['delay_encoding']

    An unconfigured bridge is the identity, and says so:

    >>> clean = MetadataFaultBridge.from_config(None)
    >>> clean.is_clean
    True
    >>> clean.apply_to_batch(batch, frame=0) is batch
    True
    """

    def __init__(self, injectors: Optional[Sequence[MetadataInjector]] = None,
                 seed: int = 0) -> None:
        self.injectors: List[MetadataInjector] = list(injectors or [])
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.records: List[FaultRecord] = []
        for injector in self.injectors:
            if injector.stage not in STAGES:
                raise ValueError(
                    f"{type(injector).__name__} declares stage "
                    f"{injector.stage!r}; expected one of {STAGES}")
        logger.info("MetadataFaultBridge(seed=%d, injectors=%s)", self.seed,
                    [i.name for i in self.injectors] or "clean")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]],
                    seed: int = 0) -> "MetadataFaultBridge":
        """Build from a ``metadata_pipeline`` config block.

        ``None`` or ``{}`` produces a bridge with no injectors at all. That
        distinction matters for the same reason it does on the physical
        plane: a reference condition that quietly injected something would
        make every comparison against it meaningless.
        """
        config = dict(config or {})
        injectors = [make_metadata_injector(name, **dict(params or {}))
                     for name, params in config.items()]
        return cls(injectors, seed=seed)

    @property
    def is_clean(self) -> bool:
        """True when no injector exists to fire."""
        return not self.injectors

    # -- the boundary -------------------------------------------------------

    def apply_to_batch(self, batch: Dict[str, Any],
                       frame: int = -1) -> Dict[str, Any]:
        """Run every injector on its batch field; ego rows are restored.

        Returns `batch` unchanged -- the same object, not a copy -- when
        nothing is configured, so a clean run costs one list check.
        """
        if not self.injectors:
            return batch
        out = dict(batch)
        for injector in self.injectors:
            key = STAGE_KEYS[injector.stage]
            original = out[key]
            corrupted, params = injector.apply(original,
                                               generator=self.generator,
                                               frame=frame)
            if params is None:
                continue
            if corrupted is original:
                corrupted = original.clone()
            corrupted[:, 0] = original[:, 0]        # the ego never lied
            out[key] = corrupted
            self.records.append(FaultRecord(
                frame=frame, agent_id="*", fault_type=injector.name,
                target=key, params=params))
        return out

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
        return (f"MetadataFaultBridge(injectors="
                f"{[i.name for i in self.injectors] or 'clean'}, "
                f"seed={self.seed})")
