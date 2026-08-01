"""
collate.py
----------
Batch scenes with ragged agent counts, carrying the V2X metadata along.

The one thing that must not go wrong
------------------------------------
The agent index in ``coords`` is **per-scene** on the way in and must become
**global** on the way out. ``PointPillarScatter`` writes each row of
``coords`` onto canvas slot ``coords[:, 0]``, so two scenes whose agent 0
both say ``0`` would be scattered onto the same canvas -- silently summing
two vehicles' LiDAR into one feature map. Nothing raises; the model trains,
and every number afterwards is wrong.

Metadata padding
----------------
``time_delay``, ``infra`` and ``velocity`` become dense ``(B, max_cav)``
tensors because the model indexes them by agent slot alongside the padded
feature stack. Padding values are the BENIGN ones -- delay 0, vehicle,
speed 0 -- because a padded slot is masked out of attention anyway, and a
sentinel like -1 would flow through the DPE's table lookup before the mask
ever applies. ``T_agent_to_ego`` pads with identity for the same reason: a
zero matrix is singular and the warp inverts the rotation block.

(Contract-identical to w2cbench's ``collate_lidar`` plus the metadata; local
copy because paper packages must not import each other.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch


def _pad_transforms(samples: Sequence[Dict[str, Any]],
                    max_cav: int) -> torch.Tensor:
    """``(B, max_cav, 4, 4)``, identity in the unused slots."""
    batch = len(samples)
    out = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, max_cav, 1, 1)
    for index, sample in enumerate(samples):
        transforms = sample["T_agent_to_ego"]
        kept = min(transforms.shape[0], max_cav)
        out[index, :kept] = transforms[:kept].to(out.dtype)
    return out


def _pad_metadata(samples: Sequence[Dict[str, Any]], key: str, max_cav: int,
                  dtype: torch.dtype) -> torch.Tensor:
    """``(B, max_cav)`` metadata, zero (the benign value) in unused slots."""
    out = torch.zeros(len(samples), max_cav, dtype=dtype)
    for index, sample in enumerate(samples):
        values = sample[key]
        kept = min(values.shape[0], max_cav)
        out[index, :kept] = values[:kept].to(dtype)
    return out


def collate_v2xvit(samples: Sequence[Dict[str, Any]],
                   max_cav: int = 5) -> Dict[str, Any]:
    """Batch scenes into the model's input dict.

    Outputs
    -------
    ``features`` (P_total, T, 10), ``coords`` (P_total, 3) with a **global**
    agent index, ``num_points`` (P_total,), ``record_len`` list of agents
    per sample, ``T_agent_to_ego`` (B, max_cav, 4, 4), ``time_delay`` /
    ``infra`` (B, max_cav) long, ``velocity`` (B, max_cav) float,
    ``gt_boxes`` list, plus the fault bookkeeping (``fault_records``,
    ``n_faults``) the benchmark writes to ``injection_summary.csv``.

    Example
    -------
    >>> import torch
    >>> def scene(n_agents, n_pillars):
    ...     return {"features": torch.zeros(n_pillars, 4, 10),
    ...             "coords": torch.zeros(n_pillars, 3, dtype=torch.long),
    ...             "num_points": torch.zeros(n_pillars, dtype=torch.long),
    ...             "T_agent_to_ego": torch.eye(4).expand(n_agents, 4, 4),
    ...             "time_delay": torch.arange(n_agents),
    ...             "infra": torch.ones(n_agents, dtype=torch.long),
    ...             "velocity": torch.full((n_agents,), 5.0),
    ...             "gt_boxes": None, "n_agents": n_agents, "frame": 0}
    >>> batch = collate_v2xvit([scene(2, 5), scene(1, 3)], max_cav=3)
    >>> batch["features"].shape, batch["record_len"]
    (torch.Size([8, 4, 10]), [2, 1])
    >>> batch["coords"][5:, 0].tolist()   # second scene's agents offset by 2
    [2, 2, 2]
    >>> batch["time_delay"].tolist()      # padded slots report delay 0
    [[0, 1, 0], [0, 0, 0]]
    >>> batch["infra"].tolist()           # padded slots read as vehicles
    [[1, 1, 0], [1, 0, 0]]
    """
    record_len = [min(int(s["n_agents"]), max_cav) for s in samples]
    faults: List[Any] = []
    for sample in samples:
        faults.extend(sample.get("fault_records", []))

    features, coords, num_points = [], [], []
    offset = 0
    for sample, kept in zip(samples, record_len):
        agent_index = sample["coords"][:, 0]
        keep = agent_index < kept                  # drop truncated agents
        shifted = sample["coords"][keep].clone()
        shifted[:, 0] += offset
        coords.append(shifted)
        features.append(sample["features"][keep])
        num_points.append(sample["num_points"][keep])
        offset += kept

    return {
        "features": torch.cat(features) if features else torch.zeros(0, 0, 10),
        "coords": (torch.cat(coords) if coords
                   else torch.zeros(0, 3, dtype=torch.long)),
        "num_points": (torch.cat(num_points) if num_points
                       else torch.zeros(0, dtype=torch.long)),
        "record_len": record_len,
        "T_agent_to_ego": _pad_transforms(samples, max_cav),
        "time_delay": _pad_metadata(samples, "time_delay", max_cav,
                                    torch.long),
        "infra": _pad_metadata(samples, "infra", max_cav, torch.long),
        "velocity": _pad_metadata(samples, "velocity", max_cav,
                                  torch.float32),
        "gt_boxes": [s["gt_boxes"] for s in samples],
        "frame": [int(s["frame"]) for s in samples],
        "fault_records": faults,
        "n_faults": len(faults),
    }


def v2xvit_collator(max_cav: int = 5):
    """A ``collate_fn`` for ``DataLoader`` with ``max_cav`` bound.

    >>> callable(v2xvit_collator(3))
    True
    """
    def _collate(samples):
        return collate_v2xvit(samples, max_cav)
    return _collate
