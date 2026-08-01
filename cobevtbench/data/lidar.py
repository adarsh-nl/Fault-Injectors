"""
lidar.py
--------
Dataset for the LiDAR track: cooperative samples -> pillar batches.

The important frame decision
----------------------------
Each agent's points are voxelised in **its own frame**, not the ego's. That
is what makes this intermediate fusion: the collaborator encodes locally,
transmits a compact BEV map, and the ego warps the *features*. Transforming
points into the ego frame first would be early fusion -- a different
architecture that happens to produce similar-looking code.

The consequence for fault injection is the whole reason it matters: under
intermediate fusion a pose error perturbs the warp, so the features arrive
intact but land in the wrong place and attention has a chance to notice.
Under early fusion it corrupts the points and there is nothing to notice.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from cpbench.data import GridSpec, PillarVoxelizer
from cpbench.faults import DataFaultBridge

from .transforms import (agent_to_ego_matrix, cooperative_gt_boxes,
                         ordered_agent_ids, world_to_ego_matrix)

logger = logging.getLogger(__name__)


class CoBEVTLidarDataset(Dataset):
    """Cooperative LiDAR frames plus 3-D detection targets.

    Purpose
        Turn a ``src.datasets`` adapter into the batches
        :class:`~cobevtbench.models.cobevt_lidar.CoBEVTLidar` consumes.

    Inputs
    ------
    adapter     any ``src.datasets.BaseDataset``
    grid        cpbench GridSpec; drives voxelisation and the warp
    max_cav     agent cap (CoBEVT: 5); ego is always kept
    bridge      DataFaultBridge; None means a provably clean run
    categories  label categories to keep as ground truth

    Outputs
    -------
    ``__getitem__`` returns one scene:

    ``features``      (P, max_points, 10)  pillar features, all agents stacked
    ``coords``        (P, 3)  [agent index within the scene, row, col]
    ``num_points``    (P,)
    ``T_agent_to_ego`` (n_agents, 4, 4)
    ``gt_boxes``      (G, 7) ego-frame ground truth
    ``n_agents``      int

    Example
    -------
    >>> from cpbench.data import SyntheticCooperativeDataset, GridSpec
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> adapter = SyntheticCooperativeDataset(n_frames=2, n_agents=2)
    >>> ds = CoBEVTLidarDataset(adapter, spec, max_cav=2)
    >>> item = ds[0]
    >>> item["features"].shape[1:], item["coords"].shape[1], item["n_agents"]
    (torch.Size([32, 10]), 3, 2)
    """

    def __init__(self, adapter, grid: GridSpec, max_cav: int = 5,
                 bridge: Optional[DataFaultBridge] = None,
                 categories: Optional[Sequence[str]] = None,
                 max_points_per_pillar: int = 32,
                 max_pillars: int = 20000,
                 gt_mode: str = "merge") -> None:
        self.gt_mode = gt_mode
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
                points = np.zeros((0, 4), dtype=np.float32)
            # Voxelised in the AGENT's own frame -- see the module docstring.
            pillars = self.voxelizer(points)
            n_pillars = pillars["coords"].shape[0]
            features.append(pillars["features"])
            num_points.append(pillars["num_points"])
            # Prepend the agent index; PointPillarScatter needs 3 columns.
            agent_column = torch.full((n_pillars, 1), agent_index,
                                      dtype=pillars["coords"].dtype)
            coords.append(torch.cat([agent_column, pillars["coords"]], dim=1))
            transforms.append(agent_to_ego_matrix(sample, agent_id))

        boxes = cooperative_gt_boxes(self.adapter, index,
                                     categories=self.categories,
                                     point_range=self.grid.point_range,
                                     mode=self.gt_mode)

        return {
            "features": torch.cat(features) if features
            else torch.zeros(0, self.voxelizer.max_points_per_pillar, 10),
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
