"""
transforms.py
-------------
Conversions between the fault toolkit's sample model and the tensors the
models consume.

``src.datasets`` describes a scene in physical terms -- Box3D with yaw in
degrees, poses as 4x4 matrices, labels possibly in world or agent frame.
``cpbench`` and both CoBEVT models work in a single convention: ``(N, 7)``
arrays of ``x, y, z, l, w, h, yaw`` in **radians**, in the **ego** frame.

Every unit and frame conversion in this package happens here, once. Scattering
them through the datasets is how a degrees-vs-radians bug ends up affecting
one track and not the other.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from cpbench.utils import transform_boxes

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
    injector perturbs ``agent.pose``, and the error enters the model here as
    a wrong warp rather than as corrupted content.
    """
    return np.linalg.inv(sample.ego.pose) @ sample.agents[agent_id].pose


def ordered_agent_ids(sample, max_cav: Optional[int] = None) -> list:
    """Agent ids with the ego first, optionally truncated to ``max_cav``.

    Ego-first is a contract several things rely on: ``regroup`` drops the
    tail when a scene exceeds ``max_cav``, and losing the ego rather than a
    collaborator would be catastrophic instead of merely lossy.
    """
    others = sorted(a for a in sample.agents if a != sample.ego_id)
    ids = [sample.ego_id] + others
    return ids[:max_cav] if max_cav is not None else ids
