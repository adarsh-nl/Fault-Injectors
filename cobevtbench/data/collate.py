"""
collate.py
----------
Batch scenes with different agent counts into one fixed-shape batch.

Two axes need reconciling and they are handled differently on purpose:

* the **agent axis of the transforms** is zero-padded to ``max_cav``, because
  FuseBEVT's relative position bias table is allocated for a fixed extent;
* the **agent axis of the features** stays flat, with a ``record_len`` telling
  the model where each scene starts. Padding images or pillars to ``max_cav``
  would run the backbone over blank inputs for every absent agent -- pure
  waste, and at CoBEVT's settings the image backbone is the most expensive
  part of the forward pass.

``regroup`` inside the model bridges the two.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def _pad_transforms(samples: Sequence[Dict[str, Any]],
                    max_cav: int) -> torch.Tensor:
    """(B, max_cav, 4, 4), identity in the padded slots.

    Identity rather than zeros: a zero matrix is singular, and the warp
    inverts the rotation block. The padded slots are masked out downstream,
    but an inversion runs over the whole batch before the mask applies, so a
    singular padding would produce NaNs in tensors that are then discarded --
    the most confusing possible failure.
    """
    batch = len(samples)
    out = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, max_cav, 1, 1)
    for i, sample in enumerate(samples):
        transforms = sample["T_agent_to_ego"]
        kept = min(transforms.shape[0], max_cav)
        out[i, :kept] = transforms[:kept]
    return out


def _common(samples: Sequence[Dict[str, Any]], max_cav: int) -> Dict[str, Any]:
    record_len = [min(int(s["n_agents"]), max_cav) for s in samples]
    faults: List[Any] = []
    for sample in samples:
        faults.extend(sample.get("fault_records", []))
    return {
        "record_len": record_len,
        "T_agent_to_ego": _pad_transforms(samples, max_cav),
        "gt_boxes": [s["gt_boxes"] for s in samples],
        "frame": [int(s["frame"]) for s in samples],
        "fault_records": faults,
        "n_faults": len(faults),
    }


def collate_camera(samples: Sequence[Dict[str, Any]],
                   max_cav: int = 5) -> Dict[str, Any]:
    """Batch camera scenes.

    Inputs
    ------
    samples  list of :class:`CoBEVTCameraDataset` items
    max_cav  agent cap; scenes with more agents are truncated ego-first

    Outputs
    -------
    ``images`` (N_total, M, H, W, 3), ``intrinsics`` (N_total, M, 3, 3),
    ``extrinsics`` (N_total, M, 4, 4), ``target`` (B, H_bev, W_bev),
    plus ``record_len``, ``T_agent_to_ego`` (B, max_cav, 4, 4) and the
    fault bookkeeping.

    Example
    -------
    >>> import torch
    >>> def scene(n):
    ...     return {"images": torch.zeros(n, 4, 8, 8, 3),
    ...             "intrinsics": torch.zeros(n, 4, 3, 3),
    ...             "extrinsics": torch.zeros(n, 4, 4, 4),
    ...             "T_agent_to_ego": torch.zeros(n, 4, 4),
    ...             "target": torch.zeros(8, 8, dtype=torch.long),
    ...             "gt_boxes": None, "n_agents": n, "frame": 0}
    >>> batch = collate_camera([scene(2), scene(1)], max_cav=3)
    >>> batch["images"].shape, batch["record_len"]
    (torch.Size([3, 4, 8, 8, 3]), [2, 1])
    >>> batch["T_agent_to_ego"].shape, batch["target"].shape
    (torch.Size([2, 3, 4, 4]), torch.Size([2, 8, 8]))
    """
    out = _common(samples, max_cav)
    for key in ("images", "intrinsics", "extrinsics"):
        out[key] = torch.cat([s[key][:max_cav] for s in samples])
    out["target"] = torch.stack([s["target"] for s in samples])
    return out


def collate_lidar(samples: Sequence[Dict[str, Any]],
                  max_cav: int = 5) -> Dict[str, Any]:
    """Batch LiDAR scenes.

    The agent index in ``coords`` is per-scene on the way in and must become
    global on the way out, or ``PointPillarScatter`` writes every scene's
    agent 0 onto the same canvas.

    Example
    -------
    >>> import torch
    >>> def scene(n_agents, n_pillars):
    ...     return {"features": torch.zeros(n_pillars, 4, 10),
    ...             "coords": torch.zeros(n_pillars, 3, dtype=torch.long),
    ...             "num_points": torch.zeros(n_pillars, dtype=torch.long),
    ...             "T_agent_to_ego": torch.zeros(n_agents, 4, 4),
    ...             "gt_boxes": None, "n_agents": n_agents, "frame": 0}
    >>> batch = collate_lidar([scene(2, 5), scene(1, 3)], max_cav=3)
    >>> batch["features"].shape, batch["record_len"]
    (torch.Size([8, 4, 10]), [2, 1])
    >>> batch["coords"][5:, 0].tolist()      # second scene's agents offset by 2
    [2, 2, 2]
    """
    out = _common(samples, max_cav)
    features, coords, num_points = [], [], []
    offset = 0
    for sample, kept in zip(samples, out["record_len"]):
        agent_index = sample["coords"][:, 0]
        keep = agent_index < kept
        shifted = sample["coords"][keep].clone()
        shifted[:, 0] += offset
        coords.append(shifted)
        features.append(sample["features"][keep])
        num_points.append(sample["num_points"][keep])
        offset += kept
    out["features"] = torch.cat(features) if features else torch.zeros(0, 0, 10)
    out["coords"] = torch.cat(coords) if coords else torch.zeros(0, 3,
                                                                 dtype=torch.long)
    out["num_points"] = torch.cat(num_points) if num_points \
        else torch.zeros(0, dtype=torch.long)
    return out


def camera_collator(max_cav: int = 5):
    """A ``collate_fn`` for ``DataLoader`` with ``max_cav`` bound."""
    def _collate(samples):
        return collate_camera(samples, max_cav)
    return _collate


def lidar_collator(max_cav: int = 5):
    """A ``collate_fn`` for ``DataLoader`` with ``max_cav`` bound."""
    def _collate(samples):
        return collate_lidar(samples, max_cav)
    return _collate
