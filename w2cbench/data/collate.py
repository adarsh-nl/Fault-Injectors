"""
collate.py
----------
Batch scenes with ragged agent counts.

The one thing that must not go wrong
------------------------------------
The agent index in ``coords`` is **per-scene** on the way in and must become
**global** on the way out. ``PointPillarScatter`` writes each row of ``coords``
onto canvas slot ``coords[:, 0]``, so two scenes whose agent 0 both say ``0``
would be scattered onto the same canvas -- silently summing two vehicles'
LiDAR into one feature map. Nothing raises; the model trains, and every
number afterwards is wrong.

Why ``T_agent_to_ego`` is padded but the pillars are not
--------------------------------------------------------
The transforms become a dense ``(B, max_cav, 4, 4)`` tensor because the model
indexes them by sample and slices to that sample's agent count. The pillars
stay concatenated and ragged, indexed by ``record_len``, because padding them
would create all-zero agent slots -- and this architecture would then score
those slots' confidence, rank them in selection, and offer them as candidate
links. Each of those would need masking again, and a mask that is missed
produces a plausible number rather than an error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def _pad_transforms(samples: Sequence[Dict[str, Any]],
                    max_cav: int) -> torch.Tensor:
    """``(B, max_cav, 4, 4)``, identity in the unused slots.

    Identity rather than zeros: a zero matrix is singular, and the warp
    inverts the rotation block. An unused slot is never read, but a NaN
    produced by inverting a padding row would propagate through the whole
    batch and be attributed to whichever agent happened to be looked at next.
    """
    batch = len(samples)
    out = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, max_cav, 1, 1)
    for index, sample in enumerate(samples):
        transforms = sample["T_agent_to_ego"]
        kept = min(transforms.shape[0], max_cav)
        out[index, :kept] = transforms[:kept].to(out.dtype)
    return out


def collate_lidar(samples: Sequence[Dict[str, Any]],
                  max_cav: int = 5) -> Dict[str, Any]:
    """Batch LiDAR scenes into the model's input dict.

    Outputs
    -------
    ``features`` (P_total, T, 9), ``coords`` (P_total, 3) with a **global**
    agent index, ``num_points`` (P_total,), ``record_len`` list of agents per
    sample, ``T_agent_to_ego`` (B, max_cav, 4, 4), ``gt_boxes`` list, plus
    the fault bookkeeping (``fault_records``, ``n_faults``) the benchmark
    writes to ``injection_summary.csv``.

    Example
    -------
    >>> import torch
    >>> def scene(n_agents, n_pillars):
    ...     return {"features": torch.zeros(n_pillars, 4, 9),
    ...             "coords": torch.zeros(n_pillars, 3, dtype=torch.long),
    ...             "num_points": torch.zeros(n_pillars, dtype=torch.long),
    ...             "T_agent_to_ego": torch.eye(4).expand(n_agents, 4, 4),
    ...             "gt_boxes": None, "n_agents": n_agents, "frame": 0}
    >>> batch = collate_lidar([scene(2, 5), scene(1, 3)], max_cav=3)
    >>> batch["features"].shape, batch["record_len"]
    (torch.Size([8, 4, 9]), [2, 1])
    >>> batch["coords"][5:, 0].tolist()   # the second scene's agents offset by 2
    [2, 2, 2]
    >>> batch["T_agent_to_ego"].shape
    torch.Size([2, 3, 4, 4])
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
        "features": torch.cat(features) if features else torch.zeros(0, 0, 9),
        "coords": (torch.cat(coords) if coords
                   else torch.zeros(0, 3, dtype=torch.long)),
        "num_points": (torch.cat(num_points) if num_points
                       else torch.zeros(0, dtype=torch.long)),
        "record_len": record_len,
        "T_agent_to_ego": _pad_transforms(samples, max_cav),
        "gt_boxes": [s["gt_boxes"] for s in samples],
        "frame": [int(s["frame"]) for s in samples],
        "fault_records": faults,
        "n_faults": len(faults),
    }


def collate_camera(samples: Sequence[Dict[str, Any]],
                   max_cav: int = 5) -> Dict[str, Any]:
    """Batch camera scenes into the model's input dict.

    Simpler than the LiDAR case: images are already rectangular, so the agent
    axis is padded rather than concatenated and there is no per-scene index to
    make global. ``record_len`` still says how many slots are real, and
    :meth:`CameraEncoder._agent_views` slices to it before the backbone runs --
    a padded slot must never reach the lift, where it would splat an all-zero
    image into a genuine BEV map.

    Outputs
    -------
    ``images`` (B, max_cav, M, ...), ``intrinsics`` (B, max_cav, M, 3, 3),
    ``extrinsics`` (B, max_cav, M, 4, 4), plus ``record_len``,
    ``T_agent_to_ego`` (B, max_cav, 4, 4), ``gt_boxes`` and the fault
    bookkeeping.

    Example
    -------
    >>> import torch
    >>> def scene(n_agents):
    ...     return {"images": torch.zeros(n_agents, 2, 3, 8, 8),
    ...             "intrinsics": torch.eye(3).expand(n_agents, 2, 3, 3),
    ...             "extrinsics": torch.eye(4).expand(n_agents, 2, 4, 4),
    ...             "T_agent_to_ego": torch.eye(4).expand(n_agents, 4, 4),
    ...             "gt_boxes": None, "n_agents": n_agents, "frame": 0}
    >>> batch = collate_camera([scene(2), scene(1)], max_cav=3)
    >>> batch["images"].shape, batch["record_len"]
    (torch.Size([2, 3, 2, 3, 8, 8]), [2, 1])
    """
    record_len = [min(int(s["n_agents"]), max_cav) for s in samples]
    faults: List[Any] = []
    for sample in samples:
        faults.extend(sample.get("fault_records", []))

    def pad(key: str) -> torch.Tensor:
        reference = samples[0][key]
        shape = (len(samples), max_cav) + tuple(reference.shape[1:])
        out = torch.zeros(shape, dtype=reference.dtype)
        for index, sample in enumerate(samples):
            kept = min(sample[key].shape[0], max_cav)
            out[index, :kept] = sample[key][:kept]
        return out

    return {
        "images": pad("images"),
        "intrinsics": pad("intrinsics"),
        "extrinsics": pad("extrinsics"),
        "record_len": record_len,
        "T_agent_to_ego": _pad_transforms(samples, max_cav),
        "gt_boxes": [s["gt_boxes"] for s in samples],
        "frame": [int(s["frame"]) for s in samples],
        "fault_records": faults,
        "n_faults": len(faults),
    }


def camera_collator(max_cav: int = 5):
    """A ``collate_fn`` for ``DataLoader`` with ``max_cav`` bound.

    >>> callable(camera_collator(3))
    True
    """
    def _collate(samples):
        return collate_camera(samples, max_cav)
    return _collate


def lidar_collator(max_cav: int = 5):
    """A ``collate_fn`` for ``DataLoader`` with ``max_cav`` bound.

    >>> callable(lidar_collator(3))
    True
    """
    def _collate(samples):
        return collate_lidar(samples, max_cav)
    return _collate
