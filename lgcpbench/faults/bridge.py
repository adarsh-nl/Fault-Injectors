"""
bridge.py
---------
The control plane's fault surface (plane 3).

The contract, mirroring plane 1's discipline one level up
    Plane 1's rule is "no model code corrupts a tensor" -- corruption happens
    once, upstream, on the CooperativeSample. Plane 3's rule is the exact
    analogue for decisions:

        Control-plane faults are applied ONLY at the RSU/CAV message
        boundary, by ControlPlaneFaultBridge, BETWEEN protocol stages.
        Algorithm code -- selection/, network/, roi/ -- is never fault-aware.

    So Algorithm 1 receives a possibly-falsified confidence matrix and runs
    exactly as published on it. Algorithm 2 receives a possibly-corrupted
    group set and schedules it exactly as published. A measured degradation
    is therefore attributable to the fault, never to fault-handling code that
    would not exist in a real deployment.

Why this plane exists at all
    LGCP's contribution is not tensors. Its observables and its failure modes
    are decisions: who reports what confidence, which CAVs form a group, who
    leads, who transmits when, what the RSU broadcasts. None of that is
    reachable by tensor-level fault injection, and as far as I can find it is
    unexplored in the collaborative-perception robustness literature. This is
    where the benchmark earns its keep.

Design mirrors DataFaultBridge deliberately
    Same config shape (``{"pipeline": {...}, "seed": ...}``), same
    ``is_clean`` property, same ``drain_records()`` audit trail. One mental
    model and one config grammar for both planes.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical injection points, matching the design doc's map (section 6.1).
# Each injector declares which one it acts on; the pipeline calls the bridge
# at each boundary, and unmatched locations pass through untouched.
STAGE_OF_LOCATION: Dict[str, str] = {
    "lgcp/roi/areas": "1-initiation",
    "lgcp/confidence/reports": "1-initiation",
    "lgcp/selection/groups": "2-assignment",
    "lgcp/network/schedule": "3-sharing",
    "lgcp/rsu/global_view": "4-aggregation",
}


@dataclass
class ControlFaultRecord:
    """One injected control-plane fault, for injection_summary.csv.

    Deliberately shaped like ``cpbench.faults.FaultRecord`` so both planes
    land in the same audit table with the same column conventions.

    Attributes
    ----------
    frame      frame index the fault was applied at.
    injector   e.g. 'confidence_report', 'leader_failure'.
    stage      protocol stage, from STAGE_OF_LOCATION.
    location   canonical injection point.
    target     what was altered: 'confidence' | 'group' | 'leader' |
               'packet' | 'area' | 'detections'.
    n_altered  how many decisions changed. Zero means the injector fired but
               had no effect -- worth logging, because it distinguishes "the
               fault did nothing" from "the fault never ran".
    params     the injector's parameters.
    """

    frame: int
    injector: str
    stage: str
    location: str
    target: str
    n_altered: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "frame": self.frame,
            "agent_id": "*",              # column-compatible with plane 1
            "fault_type": self.injector,
            "plane": "control",
            "stage": self.stage,
            "location": self.location,
            "target": self.target,
            "n_altered": self.n_altered,
        }
        for key, val in self.params.items():
            row[f"param_{key}"] = (
                val if isinstance(val, (str, int, float, bool)) else repr(val)
            )
        return row


class ControlPlaneFaultBridge:
    """Apply control-plane faults between protocol stages.

    Purpose
        The single place a decision can be corrupted. The pipeline calls
        ``apply`` at each message boundary; injectors registered for that
        location transform the payload; everything else passes through.

    Inputs
    ------
    config  ``{"pipeline": {<injector>: {<params>}}, "seed": int}``, or None
            for a clean run. Unknown injector names raise rather than being
            ignored, because a typo'd fault name in a sweep config would
            otherwise produce a silently clean condition labelled as faulty.
    seed    base seed; each injector gets an independent stream derived from
            it, so adding one injector does not perturb another's draws.

    Outputs
    -------
    ``apply(location, payload, frame=...)`` -> possibly-transformed payload
    ``drain_records()`` -> the audit trail since the last drain

    Example
    -------
    >>> bridge = ControlPlaneFaultBridge(None)
    >>> bridge.is_clean
    True
    >>> bridge.apply("lgcp/selection/groups", "untouched", frame=0)
    'untouched'
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, seed: int = 0) -> None:
        from .registry import build_injector, available_injectors

        cfg = dict(config or {})
        self.seed = int(cfg.pop("seed", seed))
        pipeline_cfg = cfg.pop("pipeline", None) or {}
        cfg.pop("name", None)
        cfg.pop("sweep", None)          # expanded by the benchmark runner
        if cfg:
            raise ValueError(
                f"unknown control-fault config keys {sorted(cfg)}; "
                f"expected 'pipeline', 'seed', 'name', 'sweep'"
            )

        self.injectors = []
        for index, (name, params) in enumerate(sorted(pipeline_cfg.items())):
            if name not in available_injectors():
                raise ValueError(
                    f"unknown control-plane injector {name!r}; expected one of "
                    f"{sorted(available_injectors())}. A typo here would produce "
                    f"a silently clean condition labelled as faulty."
                )
            self.injectors.append(build_injector(name, **(params or {})))

        # Independent stream per injector, keyed on the injector's NAME rather
        # than its position in the list. Spawning by position would mean that
        # adding a second injector shifts the first one's draws, so a
        # single-fault condition and a combined condition would not share a
        # baseline -- and any interaction effect measured between them would
        # be partly RNG drift.
        self._rngs = {
            inj.name: np.random.default_rng(self._stream_seed(inj.name))
            for inj in self.injectors
        }
        self._records: List[ControlFaultRecord] = []

        if self.injectors:
            logger.info(
                "ControlPlaneFaultBridge(seed=%d, injectors=%s)",
                self.seed, [i.name for i in self.injectors],
            )

    def _stream_seed(self, name: str) -> int:
        """A stable per-injector seed derived from the run seed and the name.

        ``hashlib`` rather than ``hash()``: Python salts string hashing per
        process, so ``hash()`` would make runs irreproducible across
        invocations -- the one property this whole design exists to preserve.
        """
        digest = hashlib.sha256(f"{self.seed}|{name}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    @property
    def is_clean(self) -> bool:
        """True if no injector is configured -- the reference condition."""
        return not self.injectors

    def apply(self, location: str, payload: Any, *, frame: int = 0) -> Any:
        """Run every injector registered for ``location``.

        Payloads for unmatched locations are returned unchanged and untouched,
        so wiring the bridge into a new boundary costs nothing until an
        injector claims it.
        """
        if not self.injectors:
            return payload

        stage = STAGE_OF_LOCATION.get(location, "unknown")
        for injector in self.injectors:
            if injector.location != location:
                continue
            payload, n_altered = injector.apply(
                payload, rng=self._rngs[injector.name], frame=frame
            )
            self._records.append(
                ControlFaultRecord(
                    frame=frame,
                    injector=injector.name,
                    stage=stage,
                    location=location,
                    target=injector.target,
                    n_altered=n_altered,
                    params=injector.params,
                )
            )
            if n_altered == 0:
                logger.debug(
                    "frame %d: %s fired at %s but altered nothing",
                    frame, injector.name, location,
                )
        return payload

    def drain_records(self) -> List[ControlFaultRecord]:
        """Return and clear the audit trail (mirrors DataFaultBridge)."""
        records, self._records = self._records, []
        return records
