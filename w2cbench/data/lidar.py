"""
lidar.py
--------
Dataset for the LiDAR track: cooperative samples -> pillar batches.

The frame decision that defines the architecture
------------------------------------------------
Each agent's points are voxelised in **its own frame**, never the ego's. That
is what makes this intermediate fusion: the collaborator encodes locally,
transmits a compact BEV map, and the ego warps the *features*. Transforming
points into the ego frame first would be early fusion -- a different
architecture that produces very similar-looking code.

For Where2comm specifically the consequence runs deeper than for a fixed-fusion
model. Because each agent computes its confidence map from features in its own
frame, a pose error leaves *selection* completely intact: the collaborator was
rightly confident about a real object and transmitted exactly the right cells.
Only the warp is wrong. Under early fusion the same error would corrupt the
points, the confidence, the selection and the bandwidth all at once, and the
four effects would be impossible to separate in the results.

Where the fault bridge sits
---------------------------
``bridge.load`` is the only entry point, so every fault -- pose error, agent
dropout, latency, sensor corruption -- is applied to the raw sample before a
single tensor is built. The audit trail is drained per item and travels with
the batch, which is what lets ``injection_summary.csv`` record faults that a
frame's *output* gives no sign of.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from cpbench.data import (GridSpec, PillarVoxelizer, agent_to_ego_matrix,
                          labels_to_array, ordered_agent_ids,
                          world_to_ego_matrix)
from cpbench.faults import DataFaultBridge

logger = logging.getLogger(__name__)


class W2CLidarDataset(Dataset):
    """Cooperative LiDAR frames plus 3-D detection targets.

    Purpose
        Turn a ``src.datasets`` adapter into the batches
        :class:`~w2cbench.models.where2comm.Where2comm` consumes.

    Inputs
    ------
    adapter     any ``src.datasets.BaseDataset``
    grid        cpbench GridSpec; drives voxelisation, the warp and anchors
    max_cav     agent cap; the ego is always kept (ego-first ordering)
    bridge      DataFaultBridge; None means a provably clean run
    categories  label categories to keep as ground truth

    Outputs
    -------
    ``__getitem__`` returns one scene::

        features        (P, max_points, 9)  pillar features, agents stacked
        coords          (P, 3)  [agent index within the scene, row, col]
        num_points      (P,)
        T_agent_to_ego  (n_agents, 4, 4)
        gt_boxes        (G, 7) ego-frame ground truth
        n_agents        int
        frame           int
        fault_records   the bridge's audit trail for this frame

    Example
    -------
    >>> from cpbench.data import GridSpec, SyntheticCooperativeDataset
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> adapter = SyntheticCooperativeDataset(n_frames=2, n_agents=2)
    >>> ds = W2CLidarDataset(adapter, spec, max_cav=2)
    >>> item = ds[0]
    >>> item["features"].shape[1:], item["coords"].shape[1], item["n_agents"]
    (torch.Size([32, 9]), 3, 2)
    >>> ds.is_clean
    True
    """

    def __init__(self, adapter, grid: GridSpec, max_cav: int = 5,
                 bridge: Optional[DataFaultBridge] = None,
                 categories: Optional[Sequence[str]] = None,
                 max_points_per_pillar: int = 32,
                 max_pillars: int = 20000) -> None:
        self.adapter = adapter
        self.grid = grid
        self.max_cav = int(max_cav)
        self.categories = tuple(categories) if categories else None
        self.bridge = bridge or DataFaultBridge(
            None, fps=getattr(adapter, "fps", 10.0))
        self.voxelizer = PillarVoxelizer(grid, max_points_per_pillar,
                                         max_pillars)

    @property
    def is_clean(self) -> bool:
        """True when no injector exists at all -- the reference condition."""
        return self.bridge.is_clean

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.bridge.load(self.adapter, index,
                                  load=("lidar", "labels"))
        agent_ids = ordered_agent_ids(sample, self.max_cav)

        features, coords, num_points, transforms = [], [], [], []
        for agent_index, agent_id in enumerate(agent_ids):
            agent = sample.agents[agent_id]
            points = agent.lidar
            if points is None:
                # A sensor fault may remove an agent's returns entirely. It
                # still occupies a row: dropping it here would silently turn a
                # sensor fault into an agent-drop fault, and the two conditions
                # would become indistinguishable in the results.
                points = np.zeros((0, 4), dtype=np.float32)
            pillars = self.voxelizer(points)          # in the AGENT's frame
            n_pillars = pillars["coords"].shape[0]
            features.append(pillars["features"])
            num_points.append(pillars["num_points"])
            agent_column = torch.full((n_pillars, 1), agent_index,
                                      dtype=pillars["coords"].dtype)
            coords.append(torch.cat([agent_column, pillars["coords"]], dim=1))
            transforms.append(agent_to_ego_matrix(sample, agent_id))

        boxes = labels_to_array(sample.ego.labels, world_to_ego_matrix(sample),
                                self.categories)

        return {
            "features": torch.cat(features) if features
            else torch.zeros(0, self.voxelizer.max_points_per_pillar, 9),
            "coords": torch.cat(coords) if coords
            else torch.zeros(0, 3, dtype=torch.int64),
            "num_points": torch.cat(num_points) if num_points
            else torch.zeros(0, dtype=torch.int64),
            "T_agent_to_ego": torch.from_numpy(
                np.stack(transforms).astype(np.float32)),
            "gt_boxes": boxes,
            "n_agents": len(agent_ids),
            "frame": int(index),
            "fault_records": self.bridge.drain_records(),
        }
