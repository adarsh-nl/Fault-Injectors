"""
dataset.py
----------
Cooperative LiDAR frames -> pillar batches, plus the metadata V2X-ViT is
built around: each agent's time delay, agent type and speed.

The frame decision that defines the architecture
------------------------------------------------
Each agent's points are voxelised in **its own frame**, never the ego's. That
is what makes this intermediate fusion: the collaborator encodes locally,
transmits a compact BEV map, and the ego warps the *features* (the STTF).
Transforming points into the ego frame first would be early fusion -- a
different architecture that produces very similar-looking code.

Where the metadata comes from
-----------------------------
``time_delay`` is read from the latency injector's audit trail
(``agent.faults['comm_latency']['delta_frames']``): when the plane-1 latency
fault serves an agent a stale frame, it records the exact staleness, and this
dataset reports it to the model -- which is precisely the paper's
asynchronous setting, where the delay is *known* and the DPE compensates.
The metadata fault plane then corrupts the *report* (post-collate), never
this ground truth.

``infra`` comes from ``AgentFrame.agent_type`` (V2XSet marks negative cav
ids as infrastructure). Synthetic adapters have no infrastructure, so
``force_infra`` can mark agent slots as infra -- without it, no smoke test
would ever execute HMSA's second projection set.

``velocity`` comes from ``AgentFrame.speed`` when the adapter records it
(OPV2V-format yaml ``ego_speed``), else 0 (assumption A5).

Where the fault bridge sits
---------------------------
``bridge.load`` is the only entry point, so every plane-1 fault -- pose
error, agent dropout, latency, sensor corruption -- is applied to the raw
sample before a single tensor is built. The audit trail is drained per item
and travels with the batch, which is what lets ``injection_summary.csv``
record faults that a frame's *output* gives no sign of.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from cpbench.data import (GridSpec, PillarVoxelizer, agent_to_ego_matrix,
                          cooperative_gt_boxes, ordered_agent_ids)
from cpbench.faults import DataFaultBridge

logger = logging.getLogger(__name__)


class V2XVitLidarDataset(Dataset):
    """Cooperative LiDAR frames plus V2X metadata and detection targets.

    Purpose
        Turn a ``src.datasets`` adapter into the batches
        :class:`~v2xvitbench.models.V2XViT` consumes.

    Inputs
    ------
    adapter      any ``src.datasets`` BaseDataset (V2XSet, OPV2V, synthetic)
    grid         the FUSION GridSpec; drives voxelisation, warp and anchors
    max_cav      agent cap; the ego is always kept (ego-first ordering)
    bridge       DataFaultBridge; None means a provably clean run
    categories   label categories to keep as ground truth
    force_infra  agent SLOT indices (post-ordering, 0 = ego) to mark as
                 infrastructure regardless of the adapter -- for synthetic
                 smoke runs. Marking the ego is refused: V2XSet egos are
                 vehicles and the detection frame is the ego's.

    Outputs
    -------
    ``__getitem__`` returns one scene::

        features        (P, max_points, 9)  pillar features, agents stacked
        coords          (P, 3)  [agent index within the scene, row, col]
        num_points      (P,)
        T_agent_to_ego  (n_agents, 4, 4)
        time_delay      (n_agents,) long -- actual staleness, frames
        infra           (n_agents,) long -- 1 = infrastructure
        velocity        (n_agents,) float -- m/s, 0 when unrecorded
        gt_boxes        (G, 7) ego-frame ground truth
        n_agents, frame, fault_records

    Example
    -------
    >>> from cpbench.data import GridSpec, SyntheticCooperativeDataset
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
    ...                 downsample=4)
    >>> adapter = SyntheticCooperativeDataset(n_frames=2, n_agents=3)
    >>> ds = V2XVitLidarDataset(adapter, spec, max_cav=3, force_infra=[2])
    >>> item = ds[0]
    >>> item["features"].shape[1:], item["n_agents"]
    (torch.Size([32, 9]), 3)
    >>> item["time_delay"].tolist(), item["infra"].tolist()
    ([0, 0, 0], [0, 0, 1])
    >>> ds.is_clean
    True
    """

    def __init__(self, adapter, grid: GridSpec, max_cav: int = 5,
                 bridge: Optional[DataFaultBridge] = None,
                 categories: Optional[Sequence[str]] = None,
                 max_points_per_pillar: int = 32,
                 max_pillars: int = 32000,
                 force_infra: Optional[Sequence[int]] = None,
                 gt_mode: str = "merge") -> None:
        self.adapter = adapter
        self.grid = grid
        self.max_cav = int(max_cav)
        self.categories = tuple(categories) if categories else None
        self.gt_mode = gt_mode
        self.bridge = bridge or DataFaultBridge(
            None, fps=getattr(adapter, "fps", 10.0))
        self.voxelizer = PillarVoxelizer(grid, max_points_per_pillar,
                                         max_pillars)
        self.force_infra = frozenset(int(i) for i in (force_infra or ()))
        if 0 in self.force_infra:
            raise ValueError(
                "force_infra must not include slot 0: the ego is a vehicle "
                "and the detection frame is the ego's")

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
        time_delay, infra, velocity = [], [], []
        for agent_index, agent_id in enumerate(agent_ids):
            agent = sample.agents[agent_id]
            points = agent.lidar
            if points is None:
                # A sensor fault may remove an agent's returns entirely. It
                # still occupies a row: dropping it here would silently turn
                # a sensor fault into an agent-drop fault, and the two
                # conditions would become indistinguishable in the results.
                points = np.zeros((0, 4), dtype=np.float32)
            pillars = self.voxelizer(points)          # in the AGENT's frame
            n_pillars = pillars["coords"].shape[0]
            features.append(pillars["features"])
            num_points.append(pillars["num_points"])
            agent_column = torch.full((n_pillars, 1), agent_index,
                                      dtype=pillars["coords"].dtype)
            coords.append(torch.cat([agent_column, pillars["coords"]], dim=1))
            transforms.append(agent_to_ego_matrix(sample, agent_id))

            latency = agent.faults.get("comm_latency", {})
            time_delay.append(int(latency.get("delta_frames", 0)))
            infra.append(1 if (agent.agent_type == "infrastructure"
                               or agent_index in self.force_infra) else 0)
            speed = getattr(agent, "speed", None)
            velocity.append(float(speed) if speed is not None else 0.0)

        boxes = cooperative_gt_boxes(self.adapter, index,
                                     categories=self.categories,
                                     point_range=self.grid.point_range,
                                     mode=self.gt_mode)

        return {
            "features": torch.cat(features) if features
            else torch.zeros(0, self.voxelizer.max_points_per_pillar, 9),
            "coords": torch.cat(coords) if coords
            else torch.zeros(0, 3, dtype=torch.int64),
            "num_points": torch.cat(num_points) if num_points
            else torch.zeros(0, dtype=torch.int64),
            "T_agent_to_ego": torch.from_numpy(
                np.stack(transforms).astype(np.float32)),
            "time_delay": torch.tensor(time_delay, dtype=torch.long),
            "infra": torch.tensor(infra, dtype=torch.long),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
            "gt_boxes": boxes,
            "n_agents": len(agent_ids),
            "frame": int(index),
            "fault_records": self.bridge.drain_records(),
        }
