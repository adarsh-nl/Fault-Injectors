"""
cooperative.py
--------------
CoRADataset: bridge from `src.datasets` cooperative samples to CoRA batches.

Data flow per ego frame k:

    adapter.get_sample(k)  ──►  DataFaultBridge (POSE / LATENCY / DROP /
                                BANDWIDTH / sensor faults -- the only
                                corruption site)  ──►  per-agent LiDAR warped
    into the EGO frame using the (corrupted) shared poses  ──►  pillars.

Warping raw points with the shared poses is the OpenCOOD intermediate-fusion
convention and is what turns upstream pose error into feature-level
misalignment -- the exact failure mode CoRA is built to survive.

Ground truth comes from the ego agent's own annotations (ego pose is never
corrupted), so labels remain valid under every fault condition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.base import BaseDataset, Box3D

from ..faults.bridge import DataFaultBridge
from .preprocessing import (AnchorGenerator, GridSpec, PillarVoxelizer,
                            TargetAssigner)

logger = logging.getLogger(__name__)


def _boxes_from_labels(labels: Sequence[Box3D], T_world_to_ego: np.ndarray,
                       categories: Optional[Sequence[str]] = None
                       ) -> np.ndarray:
    """`src.datasets.Box3D` list -> (G, 7) [x,y,z,l,w,h,yaw(rad)] ego frame."""
    rows: List[List[float]] = []
    for box in labels:
        if categories and box.category and box.category not in categories:
            continue
        cx, cy, cz = [float(v) for v in np.asarray(box.center).ravel()[:3]]
        l, w, h = [float(v) for v in np.asarray(box.size).ravel()[:3]]
        yaw = float(np.radians(box.yaw))
        if box.frame == "world":
            pt = T_world_to_ego @ np.array([cx, cy, cz, 1.0])
            cx, cy, cz = pt[:3]
            yaw += float(np.arctan2(T_world_to_ego[1, 0], T_world_to_ego[0, 0]))
        rows.append([cx, cy, cz, l, w, h, yaw])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 7)


class CoRADataset(Dataset):
    """Torch dataset over a `src.datasets` adapter, fault bridge included.

    Parameters
    ----------
    adapter      : any `src.datasets.BaseDataset` (OPV2V, DAIR-V2X, ...).
    grid         : GridSpec shared with the model.
    bridge       : DataFaultBridge or None (clean).
    max_agents   : cap on agents per sample (ego always kept).
    comm_range_m : collaborators farther than this from the ego are excluded
                   (measured with the corrupted poses -- the ego can only
                   judge distance from what was transmitted).
    categories   : GT category filter (None = all).

    __getitem__ output (one ego frame)
    ----------------------------------
    dict with:
      pillars      list (per agent) of voxelizer dicts
      agent_ids    list[str], ego first
      ego_index    0
      gt_boxes     (G, 7) float32 ego frame
      cls_target   (H, W, A) float32
      reg_target   (H, W, A, 7) float32
      frame        int, n_agents int
      fault_records list -- audit trail drained from the bridge
    """

    def __init__(self, adapter: BaseDataset, grid: GridSpec,
                 bridge: Optional[DataFaultBridge] = None,
                 anchor_generator: Optional[AnchorGenerator] = None,
                 target_assigner: Optional[TargetAssigner] = None,
                 max_points_per_pillar: int = 32, max_pillars: int = 20000,
                 max_agents: int = 5, comm_range_m: float = 70.0,
                 categories: Optional[Sequence[str]] = None) -> None:
        self.adapter = adapter
        self.grid = grid
        self.bridge = bridge or DataFaultBridge(None, fps=getattr(adapter, "fps", 10.0))
        self.voxelizer = PillarVoxelizer(grid, max_points_per_pillar, max_pillars)
        self.anchor_generator = anchor_generator or AnchorGenerator(grid)
        self.target_assigner = target_assigner or TargetAssigner(self.anchor_generator)
        self.max_agents = int(max_agents)
        self.comm_range_m = float(comm_range_m)
        self.categories = list(categories) if categories else None

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, k: int) -> Dict[str, Any]:
        sample = self.bridge.load(self.adapter, k, load=("lidar", "labels"))
        ego = sample.ego
        if ego.pose is None:
            raise ValueError(f"frame {k}: ego agent {sample.ego_id!r} has no pose")
        T_world_to_ego = np.linalg.inv(ego.pose)

        # order: ego first, then collaborators within comm range
        agent_ids = [sample.ego_id]
        for aid, agent in sample.agents.items():
            if aid == sample.ego_id or agent.lidar is None:
                continue
            if agent.pose is not None:
                dist = float(np.linalg.norm(agent.pose[:2, 3] - ego.pose[:2, 3]))
                if dist > self.comm_range_m:
                    continue
            agent_ids.append(aid)
            if len(agent_ids) >= self.max_agents:
                break

        pillars = []
        for aid in agent_ids:
            pts = sample.lidar_in_ego_frame(aid) if aid != sample.ego_id \
                else sample.ego.lidar
            pillars.append(self.voxelizer(
                pts if pts is not None else np.zeros((0, 4), dtype=np.float32)))

        gt_boxes = _boxes_from_labels(ego.labels, T_world_to_ego,
                                      self.categories)
        targets = self.target_assigner(gt_boxes)
        return {
            "pillars": pillars,
            "agent_ids": [str(a) for a in agent_ids],
            "ego_index": 0,
            "gt_boxes": gt_boxes,
            "cls_target": targets["cls_target"],
            "reg_target": targets["reg_target"],
            "frame": int(k),
            "n_agents": len(agent_ids),
            "fault_records": self.bridge.drain_records(),
        }


def collate_cooperative(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate ego frames with varying agent counts into one flat batch.

    All agents of all frames are concatenated into a single "agent batch";
    `agent_frame` maps each agent back to its ego frame, `ego_mask` marks
    the ego rows. Pillar tensors are concatenated with a leading agent index
    column in `coords` so the scatter stage can address (agent, row, col).

    Output keys
    -----------
    features (ΣP, max_pts, 9) · coords (ΣP, 3)[agent, row, col] ·
    num_points (ΣP,) · agent_frame (Na,) · ego_mask (Na,) ·
    cls_target (B, H, W, A) · reg_target (B, H, W, A, 7) ·
    gt_boxes list[np.ndarray] · frames (B,) · agent_ids list[list[str]] ·
    fault_records list
    """
    features, coords, num_points = [], [], []
    agent_frame, ego_mask = [], []
    agent_counter = 0
    for b, item in enumerate(items):
        for i, pil in enumerate(item["pillars"]):
            p = pil["coords"].shape[0]
            features.append(pil["features"])
            num_points.append(pil["num_points"])
            coords.append(torch.cat([
                torch.full((p, 1), agent_counter, dtype=torch.int64),
                pil["coords"]], dim=1))
            agent_frame.append(b)
            ego_mask.append(i == item["ego_index"])
            agent_counter += 1

    return {
        "features": torch.cat(features) if features else torch.zeros(0, 1, 9),
        "coords": torch.cat(coords) if coords else torch.zeros(0, 3, dtype=torch.int64),
        "num_points": torch.cat(num_points) if num_points else torch.zeros(0, dtype=torch.int64),
        "agent_frame": torch.tensor(agent_frame, dtype=torch.int64),
        "ego_mask": torch.tensor(ego_mask, dtype=torch.bool),
        "cls_target": torch.stack([it["cls_target"] for it in items]),
        "reg_target": torch.stack([it["reg_target"] for it in items]),
        "gt_boxes": [it["gt_boxes"] for it in items],
        "frames": torch.tensor([it["frame"] for it in items]),
        "agent_ids": [it["agent_ids"] for it in items],
        "fault_records": [r for it in items for r in it["fault_records"]],
    }
