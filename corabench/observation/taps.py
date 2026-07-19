"""
taps.py
-------
The read-only observation protocol of corabench.

Purpose
    Expose every intermediate tensor of a model at a named location WITHOUT
    giving observers any way to alter the forward pass. Corruption never
    happens here; it happens upstream on raw data (see `corabench.faults`).
    Taps exist for measurement: statistics, tensor dumps for information-
    quality estimation, drift-vs-clean analysis.

Contract
    * ``observe`` returns ``None`` -- there is nothing to feed back.
    * ``emit`` (the helper every module calls) hands the tap a DETACHED
      tensor, so autograd cannot be rerouted and in-place edits by a buggy
      tap cannot reach gradients. Mutating the handed tensor is still a
      programming error; ``TapSet(strict=True)`` clones defensively.
    * With ``taps=None`` the hooks cost one ``is None`` check.

Example
    >>> stats = StatsTap()                            # doctest: +SKIP
    >>> taps = TapSet([stats])
    >>> out = model(batch, taps=taps)                 # forward unchanged
    >>> stats.records[0].location
    'encoder/bev_features'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Sequence

import torch

try:  # Protocol is 3.8+; keep a soft fallback for exotic interpreters
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object          # type: ignore[assignment]

    def runtime_checkable(cls):  # type: ignore[misc]
        return cls


@dataclass
class TapRecord:
    """One observation of one tensor at one location.

    Attributes
    ----------
    module / location : where the observation was made (canonical names from
        `corabench.observation.locations`).
    shape, dtype      : tensor metadata.
    stats             : summary statistics (mean, std, l2, sparsity, ...).
    context           : free-form call-site context (agent_id, frame, ...).
    """

    module: str
    location: str
    shape: tuple
    dtype: str
    stats: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        """Flatten to a CSV-friendly dict (stats and context inlined)."""
        row: Dict[str, Any] = {
            "module": self.module,
            "location": self.location,
            "shape": "x".join(map(str, self.shape)),
            "dtype": self.dtype,
        }
        row.update(self.stats)
        row.update({f"ctx_{k}": v for k, v in self.context.items()})
        return row


@runtime_checkable
class TapProtocol(Protocol):
    """Anything with a read-only ``observe`` method.

    Inputs
    ------
    tensor   : detached ``torch.Tensor`` (never modify it).
    module   : class name of the observing module, e.g. ``"LCModule"``.
    location : canonical location name, e.g. ``"lc/z_fused"``.
    context  : extra keyword context (agent_id, frame_index, ...).

    Output: ``None`` -- observations cannot influence the forward pass.
    """

    def observe(self, tensor: torch.Tensor, *, module: str, location: str,
                **context: Any) -> None: ...


class NullTap:
    """No-op tap; useful as an explicit default and in tests."""

    def observe(self, tensor: torch.Tensor, *, module: str, location: str,
                **context: Any) -> None:
        return None


class TapSet:
    """Route observations to several taps, optionally filtered by location.

    Purpose
        Lets one run record statistics everywhere while dumping full tensors
        only at a handful of locations.

    Inputs
    ------
    taps      : sequence of TapProtocol objects.
    include   : optional glob patterns; a tap-set level filter. A location is
                observed iff it matches ANY pattern (default: all).
    strict    : clone tensors before handing them to taps. Slower, but makes
                even a misbehaving tap physically unable to touch the
                forward tensor's storage.

    Example
    -------
    >>> ts = TapSet([StatsTap()], include=["lc/*", "pac/attention_map"])
    """

    def __init__(self, taps: Sequence[TapProtocol],
                 include: Optional[Sequence[str]] = None,
                 strict: bool = False) -> None:
        self.taps: List[TapProtocol] = list(taps)
        self.include = list(include) if include is not None else None
        self.strict = strict

    def wants(self, location: str) -> bool:
        """True if `location` passes this tap-set's include filter."""
        if self.include is None:
            return True
        return any(fnmatch(location, pat) for pat in self.include)

    def observe(self, tensor: torch.Tensor, *, module: str, location: str,
                **context: Any) -> None:
        if not self.wants(location):
            return
        handed = tensor.clone() if (self.strict and torch.is_tensor(tensor)) \
            else tensor
        for tap in self.taps:
            tap.observe(handed, module=module, location=location, **context)


def emit(taps: Optional[TapProtocol], tensor: Any, *, module: str,
         location: str, **context: Any) -> None:
    """The single hook every model module calls.

    Detaches the tensor (observation must not create autograd edges) and
    forwards it to ``taps.observe``. With ``taps=None`` this is one branch.

    Non-tensor payloads (e.g. lists of boxes) are passed through unchanged;
    taps that only understand tensors should ignore them.
    """
    if taps is None:
        return
    if torch.is_tensor(tensor):
        tensor = tensor.detach()
    taps.observe(tensor, module=module, location=location, **context)
