"""
samples.py
----------
Conversions between the fault toolkit's sample model and model tensors.

``src.datasets`` describes a scene in physical terms -- ``Box3D`` with yaw in
degrees, poses as 4x4 matrices, labels possibly in world or agent frame. Every
model in this repository works in one convention instead: ``(N, 7)`` arrays of
``x, y, z, l, w, h, yaw`` in **radians**, in the **ego** frame.

Every unit and frame conversion happens here, once. Scattering them through
the datasets is how a degrees-versus-radians bug ends up affecting one paper
package and not another -- and a silent 57x error in a yaw target looks like a
model that simply will not converge.

Extracted from ``cobevtbench.data.transforms`` when ``w2cbench`` became the
third package to need it (``corabench.data.cooperative`` holds a fourth,
divergent implementation of the box conversion). That package re-exports from
here, so the move is additive.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..utils import transform_boxes

EMPTY_BOXES = np.zeros((0, 7), dtype=np.float32)


def labels_to_array(labels: Sequence, T_world_to_ego: Optional[np.ndarray] = None,
                    categories: Optional[Sequence[str]] = None) -> np.ndarray:
    """``List[Box3D]`` -> ``(N, 7)`` ego-frame boxes with yaw in radians.

    Inputs
    ------
    labels          Box3D list from a CooperativeSample
    T_world_to_ego  applied to boxes whose ``frame`` is ``"world"``. Boxes
                    already in the agent frame are passed through -- mixing
                    the two is legal in the sample model, so the frame tag is
                    honoured per box rather than assumed for the list.
    categories      keep only these categories; None keeps everything

    Outputs
    -------
    ``(N, 7)`` float32 -- x, y, z, l, w, h, yaw[rad].

    Example
    -------
    >>> import numpy as np
    >>> from src.datasets.base import Box3D
    >>> box = Box3D(center=np.array([1.0, 2.0, 0.0]),
    ...             size=np.array([4.0, 2.0, 1.5]), yaw=90.0, frame="agent")
    >>> out = labels_to_array([box])
    >>> out.shape, round(float(out[0, 6]), 4)
    ((1, 7), 1.5708)
    """
    if not len(labels):
        return EMPTY_BOXES.copy()

    world_rows, agent_rows = [], []
    for box in labels:
        if categories is not None and box.category not in categories:
            continue
        row = [float(box.center[0]), float(box.center[1]), float(box.center[2]),
               float(box.size[0]), float(box.size[1]), float(box.size[2]),
               float(np.radians(box.yaw))]          # Box3D yaw is DEGREES
        (world_rows if box.frame == "world" else agent_rows).append(row)

    parts = []
    if world_rows:
        world = np.asarray(world_rows, dtype=np.float64)
        parts.append(transform_boxes(world, T_world_to_ego)
                     if T_world_to_ego is not None else world)
    if agent_rows:
        parts.append(np.asarray(agent_rows, dtype=np.float64))
    if not parts:
        return EMPTY_BOXES.copy()
    return np.concatenate(parts).astype(np.float32)


def world_to_ego_matrix(sample) -> np.ndarray:
    """``(4, 4)`` transform from world coordinates into the ego frame."""
    return np.linalg.inv(sample.ego.pose)


def agent_to_ego_matrix(sample, agent_id: str) -> np.ndarray:
    """``(4, 4)`` transform from one agent's frame into the ego frame.

    This is where a pose-error fault becomes a feature misalignment: the
    injector perturbs ``agent.pose``, and the error enters the model here as a
    wrong warp rather than as corrupted content.
    """
    return np.linalg.inv(sample.ego.pose) @ sample.agents[agent_id].pose


def ordered_agent_ids(sample, max_cav: Optional[int] = None) -> list:
    """Agent ids with the ego first, optionally truncated to ``max_cav``.

    Ego-first is a contract several things rely on: truncation drops the tail
    when a scene exceeds ``max_cav``, and losing the ego rather than a
    collaborator would be catastrophic instead of merely lossy.
    """
    others = sorted(a for a in sample.agents if a != sample.ego_id)
    ids = [sample.ego_id] + others
    return ids[:max_cav] if max_cav is not None else ids


def cooperative_gt_boxes(adapter, k: int, *, categories=None,
                         point_range=None, mode: str = "merge",
                         dedup_iou: float = 0.5) -> np.ndarray:
    """The answer key for ego frame ``k``: merged multi-agent ground truth.

    Scoring against the ego's own label file alone punishes cooperation: a
    collaborator-revealed object that the model correctly detects would be
    counted as a false positive because it is missing from the key. This
    builder therefore merges the labels of EVERY agent in the scene -- the
    OpenCOOD evaluation convention -- and it does so from a freshly loaded
    CLEAN sample, so fault injection (pose noise, latency, dropout) can
    never corrupt the ground truth it is being measured against.

    Inputs
    ------
    adapter      a ``src.datasets.BaseDataset`` (NOT a corrupted sample).
    k            ego frame index.
    categories   keep only these Box3D categories (None = all).
    point_range  ``(xmin, ymin, zmin, xmax, ymax, zmax)``: boxes whose centre
                 falls outside the x/y bounds are dropped, matching the
                 model's detection range (otherwise out-of-range objects
                 inflate the false-negative count of every model).
    mode         ``"merge"`` (all agents, deduplicated -- default) or
                 ``"ego"`` (ego labels only; the pre-fix behaviour, kept for
                 comparison studies).
    dedup_iou    two boxes closer than this BEV IoU are one object.

    Output: ``(G, 7)`` float32 ego-frame boxes, yaw in radians.

    Deduplication uses ``track_id`` when present (OPV2V ids are global CARLA
    actor ids, so identical ids across agents ARE the same object) and falls
    back to BEV IoU for id-less or cross-source labels (e.g. DAIR-V2X, where
    vehicle- and infrastructure-side ids are independent).
    """
    from ..utils import rotated_iou_bev

    if mode not in ("merge", "ego"):
        raise ValueError(f"mode must be 'merge'|'ego', got {mode!r}")
    sample = adapter.get_sample(k, load=("labels",))
    if sample.ego.pose is None:
        raise ValueError(f"frame {k}: ego agent {sample.ego_id!r} has no pose")
    T_we = np.linalg.inv(sample.ego.pose)

    order = [sample.ego_id] if mode == "ego" else ordered_agent_ids(sample)
    kept_rows: list = []
    seen_ids: set = set()
    for aid in order:
        agent = sample.agents[aid]
        T_ae = (T_we @ agent.pose) if (aid != sample.ego_id
                                       and agent.pose is not None) else None
        for box in agent.labels:
            if categories is not None and box.category not in categories:
                continue
            row = np.array([[float(box.center[0]), float(box.center[1]),
                             float(box.center[2]), float(box.size[0]),
                             float(box.size[1]), float(box.size[2]),
                             float(np.radians(box.yaw))]])
            if box.frame == "world":
                row = transform_boxes(row, T_we)
            elif aid != sample.ego_id:
                if T_ae is None:
                    continue              # agent-frame box, no pose to place it
                row = transform_boxes(row, T_ae)
            if point_range is not None:
                xmin, ymin, _, xmax, ymax, _ = point_range
                if not (xmin <= row[0, 0] < xmax and ymin <= row[0, 1] < ymax):
                    continue
            tid = str(getattr(box, "track_id", "") or "")
            if tid and tid in seen_ids:
                continue
            # IoU dedup runs even for tracked boxes: cross-source labels
            # (vehicle vs infrastructure side) carry independent ids
            if kept_rows and rotated_iou_bev(
                    row, np.asarray(kept_rows)).max() >= dedup_iou:
                continue
            if tid:
                seen_ids.add(tid)
            kept_rows.append(row[0])
    if not kept_rows:
        return EMPTY_BOXES.copy()
    return np.asarray(kept_rows, dtype=np.float32).reshape(-1, 7)
