"""
recorders.py
------------
Read-only taps for the CONTROL plane.

Why corabench's recorders are not enough
    ``corabench.observation.recorders.StatsTap`` computes tensor statistics
    and, by its own contract, "non-tensors are ignored". That is correct for
    CoRA, whose every observable is a tensor.

    LGCP's contribution is not tensors. Its observables are decisions --
    groups, leaders, load dictionaries, packet schedules. Routed through
    StatsTap they vanish silently, which would leave the entire control plane
    unobservable while every test still passed. This module closes that gap.

Contract (identical to plane 2's)
    ``observe`` returns None and never mutates the observed object. Payloads
    are summarised into flat, CSV-friendly rows by default; ``retain=True``
    additionally keeps a reference for in-process assertions and replay.
    Retention is off by default because a schedule for 30 CAVs over hundreds
    of frames is large, and a benchmark run should not silently accumulate it.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ControlRecord:
    """One observation of one control-plane decision.

    Attributes
    ----------
    module    the class that emitted it, e.g. ``"SelectionAlgorithm"``.
    location  canonical name, e.g. ``"lgcp/selection/groups"``.
    kind      payload type name, e.g. ``"list[Group]"``.
    size      element count (len) where meaningful, else 1.
    summary   scalar aggregates derived from the payload.
    context   free-form call-site context (frame, delta_g, ...).
    payload   the object itself, only when the tap was built with retain=True.
    """

    module: str
    location: str
    kind: str
    size: int
    summary: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    payload: Any = None

    def as_row(self) -> Dict[str, Any]:
        """Flatten to a CSV-friendly dict (never includes ``payload``)."""
        row: Dict[str, Any] = {
            "module": self.module,
            "location": self.location,
            "kind": self.kind,
            "size": self.size,
        }
        row.update({str(k): v for k, v in self.summary.items()})
        row.update({f"ctx_{k}": v for k, v in self.context.items()})
        return row


def _summarise(payload: Any) -> Dict[str, Any]:
    """Derive scalar aggregates from a control-plane payload.

    Deliberately duck-typed rather than switching on concrete classes: new
    decision types get useful summaries without editing this function, and
    ``observation`` never has to import ``selection`` or ``network`` (which
    would invert the dependency graph).
    """
    summary: Dict[str, Any] = {}

    if isinstance(payload, dict):
        numeric = [v for v in payload.values() if isinstance(v, (int, float))]
        if numeric:
            summary["value_max"] = max(numeric)
            summary["value_min"] = min(numeric)
            summary["value_sum"] = sum(numeric)
            summary["value_mean"] = sum(numeric) / len(numeric)
        return summary

    if isinstance(payload, (list, tuple)):
        items = list(payload)
        if not items:
            return summary
        # group-like: anything exposing `size` and `is_orphaned`
        if all(hasattr(i, "size") and hasattr(i, "is_orphaned") for i in items):
            sizes = [int(i.size) for i in items]
            summary["n_orphaned"] = sum(1 for i in items if i.is_orphaned)
            summary["size_max"] = max(sizes)
            summary["size_mean"] = sum(sizes) / len(sizes)
            if all(hasattr(i, "confidence") for i in items):
                confs = [float(i.confidence) for i in items]
                summary["confidence_mean"] = sum(confs) / len(confs)
            if all(hasattr(i, "leader") for i in items):
                summary["n_with_leader"] = sum(1 for i in items if i.leader is not None)
        elif all(isinstance(i, (int, float)) for i in items):
            summary["value_max"] = max(items)
            summary["value_min"] = min(items)
            summary["value_mean"] = sum(items) / len(items)
        return summary

    if is_dataclass(payload) and not isinstance(payload, type):
        for name, value in vars(payload).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                summary[name] = value
        return summary

    if isinstance(payload, (int, float, str, bool)):
        summary["value"] = payload

    return summary


def _kind(payload: Any) -> str:
    """Readable type name, e.g. ``list[Group]``."""
    if isinstance(payload, (list, tuple)) and payload:
        inner = type(payload[0]).__name__
        return f"{type(payload).__name__}[{inner}]"
    return type(payload).__name__


def _size(payload: Any) -> int:
    try:
        return len(payload)
    except TypeError:
        return 1


class ControlPlaneTap:
    """Record control-plane decisions -- groups, leaders, loads, schedules.

    Purpose
        Make LGCP's own contribution observable. Without this, a fault that
        changes which CAV leads an area would be invisible until it moved AP,
        which is far downstream and confounded by everything else.

    Inputs
    ------
    retain   keep a reference to each payload. Off by default: a 30-CAV
             schedule over hundreds of frames is large, and a benchmark run
             must not silently accumulate it. Turn on for tests and replay.
    include  optional location globs; None records everything.

    Outputs
    -------
    ``self.records`` (list of ControlRecord); ``to_rows()``; ``to_csv(path)``.

    Example
    -------
    >>> tap = ControlPlaneTap(retain=True)
    >>> tap.observe({"a": 2.0, "b": 0.0}, module="M",
    ...             location="lgcp/selection/loads", frame=3)
    >>> r = tap.records[0]
    >>> r.location, r.size, r.summary["value_max"], r.context["frame"]
    ('lgcp/selection/loads', 2, 2.0, 3)
    """

    def __init__(
        self, retain: bool = False, include: Optional[Sequence[str]] = None
    ) -> None:
        self.retain = retain
        self.include = list(include) if include is not None else None
        self.records: List[ControlRecord] = []

    def _wants(self, location: str) -> bool:
        if self.include is None:
            return True
        from fnmatch import fnmatch

        return any(fnmatch(location, pat) for pat in self.include)

    def observe(
        self, payload: Any, *, module: str, location: str, **context: Any
    ) -> None:
        """Record one decision. Returns None; never mutates ``payload``."""
        if not self._wants(location):
            return
        self.records.append(
            ControlRecord(
                module=module,
                location=location,
                kind=_kind(payload),
                size=_size(payload),
                summary=_summarise(payload),
                context=dict(context),
                payload=payload if self.retain else None,
            )
        )

    # ------------------------------------------------------------------ #
    # access / export
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Drop all records (called between frames by long runs)."""
        self.records.clear()

    def locations(self) -> List[str]:
        """Distinct locations seen, in first-observation order."""
        seen: List[str] = []
        for r in self.records:
            if r.location not in seen:
                seen.append(r.location)
        return seen

    def latest(self, location: str) -> Optional[ControlRecord]:
        """Most recent record at a location, or None."""
        for r in reversed(self.records):
            if r.location == location:
                return r
        return None

    def to_rows(self) -> List[Dict[str, Any]]:
        """All records as flat dicts."""
        return [r.as_row() for r in self.records]

    def to_csv(self, path: Path) -> Path:
        """Write records to CSV, unioning columns across heterogeneous rows.

        Different locations produce different summary keys, so the header is
        the union and missing cells are blank -- the same convention the
        corabench logbook uses.
        """
        path = Path(path)
        rows = self.to_rows()
        if not rows:
            logger.warning("ControlPlaneTap: no records to write to %s", path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            return path

        columns: List[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("ControlPlaneTap: wrote %d records to %s", len(rows), path)
        return path
