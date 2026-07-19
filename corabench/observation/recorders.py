"""
recorders.py
------------
Concrete read-only taps: statistics, tensor dumps, drift-vs-clean.

All recorders honour the tap contract: they never modify the observed tensor
and return nothing. `TensorDumpTap` writes `.npz` files consumable by the
information-quality tooling in `src/info_quality` (RQ2 analysis).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .taps import TapRecord

logger = logging.getLogger(__name__)


def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    """Summary statistics of a tensor (float-cast, NaN-safe)."""
    if t.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "l2": 0.0, "amax": 0.0,
                "sparsity": 1.0, "n_nan": 0.0, "n_inf": 0.0}
    f = t.detach().float()
    finite = torch.isfinite(f)
    safe = torch.where(finite, f, torch.zeros_like(f))
    n = f.numel()
    return {
        "mean": safe.sum().item() / n,
        "std": safe.std().item() if n > 1 else 0.0,
        "l2": safe.norm().item(),
        "amax": safe.abs().max().item(),
        "sparsity": (safe == 0).sum().item() / n,
        "n_nan": torch.isnan(f).sum().item(),
        "n_inf": torch.isinf(f).sum().item(),
    }


class StatsTap:
    """Record summary statistics of every observed tensor.

    Purpose   cheap always-on measurement: layer health (NaN/Inf), energy,
              sparsity -- the per-layer signal used for propagation analysis.
    Inputs    tensors of any shape; non-tensors are ignored.
    Output    ``self.records`` (list of TapRecord); ``to_csv(path)``.

    Example
    -------
    >>> tap = StatsTap()
    >>> emit(TapSet([tap]), torch.ones(2, 3), module="M", location="lc/gate")
    >>> tap.records[0].stats["l2"]                    # doctest: +SKIP
    """

    def __init__(self) -> None:
        self.records: List[TapRecord] = []

    def observe(self, tensor: Any, *, module: str, location: str,
                **context: Any) -> None:
        if not torch.is_tensor(tensor):
            return
        self.records.append(TapRecord(
            module=module, location=location, shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            stats=_tensor_stats(tensor),
            context={k: v for k, v in context.items()
                     if isinstance(v, (str, int, float, bool))},
        ))

    def clear(self) -> None:
        self.records.clear()

    def to_rows(self) -> List[Dict[str, Any]]:
        return [r.as_row() for r in self.records]

    def to_csv(self, path: "str | Path") -> Path:
        """Write all records to CSV (union of columns across records)."""
        path = Path(path)
        rows = self.to_rows()
        if not rows:
            logger.warning("StatsTap.to_csv: no records to write to %s", path)
            return path
        cols: List[str] = []
        for row in rows:
            for key in row:
                if key not in cols:
                    cols.append(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        return path


class TensorDumpTap:
    """Persist observed tensors as ``.npz`` for offline analysis.

    Purpose   feed `src/info_quality` (mutual-information estimators) and any
              other offline measurement with the actual intermediate tensors.
    Inputs    every_n: keep one frame in `every_n` (contexts must carry
              ``frame``); locations observed are whatever the enclosing
              TapSet routes here.
    Output    files ``<out_dir>/<location with '/'->'__'>/frame<k>[_<agent>].npz``
              each holding array ``tensor`` plus scalar metadata.
    """

    def __init__(self, out_dir: "str | Path", every_n: int = 1,
                 max_dumps: Optional[int] = None) -> None:
        self.out_dir = Path(out_dir)
        self.every_n = max(1, int(every_n))
        self.max_dumps = max_dumps
        self.n_dumped = 0

    def observe(self, tensor: Any, *, module: str, location: str,
                **context: Any) -> None:
        if not torch.is_tensor(tensor):
            return
        frame = context.get("frame", 0)
        if isinstance(frame, int) and frame % self.every_n != 0:
            return
        if self.max_dumps is not None and self.n_dumped >= self.max_dumps:
            return
        loc_dir = self.out_dir / location.replace("/", "__")
        loc_dir.mkdir(parents=True, exist_ok=True)
        agent = context.get("agent_id")
        stem = f"frame{frame}" + (f"_{agent}" if agent is not None else "")
        np.savez_compressed(
            loc_dir / f"{stem}.npz",
            tensor=tensor.detach().cpu().float().numpy(),
            module=module, location=location,
            **{k: v for k, v in context.items()
               if isinstance(v, (str, int, float, bool))})
        self.n_dumped += 1


class DriftTap:
    """Measure divergence of observed tensors from a cached clean run.

    Purpose   layer-wise robustness: after a CleanBenchmarkRunner dumps
              reference tensors with TensorDumpTap, a faulted run with a
              DriftTap pointed at that dump directory yields, per location,
              L2 and cosine drift of each intermediate representation.
    Inputs    reference_dir: the TensorDumpTap out_dir of the clean run.
    Output    ``self.records``: TapRecords whose stats hold
              ``drift_l2`` (relative) and ``drift_cos`` (1 - cosine sim).
              Locations/frames without a reference are skipped silently.
    """

    def __init__(self, reference_dir: "str | Path") -> None:
        self.reference_dir = Path(reference_dir)
        self.records: List[TapRecord] = []

    def observe(self, tensor: Any, *, module: str, location: str,
                **context: Any) -> None:
        if not torch.is_tensor(tensor):
            return
        frame = context.get("frame", 0)
        agent = context.get("agent_id")
        stem = f"frame{frame}" + (f"_{agent}" if agent is not None else "")
        ref_path = self.reference_dir / location.replace("/", "__") / f"{stem}.npz"
        if not ref_path.exists():
            return
        ref = np.load(ref_path)["tensor"]
        cur = tensor.detach().cpu().float().numpy()
        if ref.shape != cur.shape:
            logger.warning("DriftTap: shape mismatch at %s frame %s (%s vs %s)",
                           location, frame, ref.shape, cur.shape)
            return
        ref_f, cur_f = ref.ravel(), cur.ravel()
        ref_norm = float(np.linalg.norm(ref_f))
        diff = float(np.linalg.norm(cur_f - ref_f))
        denom = float(np.linalg.norm(cur_f)) * ref_norm
        cos = float(np.dot(cur_f, ref_f) / denom) if denom > 0 else 1.0
        self.records.append(TapRecord(
            module=module, location=location, shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            stats={"drift_l2": diff / ref_norm if ref_norm > 0 else diff,
                   "drift_cos": 1.0 - cos},
            context={k: v for k, v in context.items()
                     if isinstance(v, (str, int, float, bool))},
        ))
