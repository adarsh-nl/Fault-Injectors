"""CooperativeSample -> CoRA training batch (spec §1.1, §4).

proj_first convention: each agent's points are projected into the EGO frame
before voxelisation (the OpenCOOD base the paper builds on), so all agents
share the ego BEV grid and pose error acts on exactly the tensor the paper's
robustness protocol perturbs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from cpbench.data import (AnchorGenerator, GridSpec, PillarVoxelizer,
                          TargetAssigner)
from cpbench.data.samples import agent_to_ego_matrix, labels_to_array, \
    ordered_agent_ids
from src.datasets.base import transform_points


class CoRADataset:
    """Wraps a `src.datasets` BaseDataset (opv2v / dair-v2x / synthetic).

    Optional `bridge` is a cpbench DataFaultBridge (corruption plane): it
    corrupts the CooperativeSample BEFORE any tensor exists.
    """

    def __init__(self, dataset, grid: GridSpec,
                 max_cav: int = 5, bridge=None, reg_dim: int = 8,
                 max_points_per_pillar: int = 32,
                 max_pillars: int = 12000) -> None:
        self.ds = dataset
        self.grid = grid
        self.max_cav = max_cav
        self.bridge = bridge
        self.voxelizer = PillarVoxelizer(grid, max_points_per_pillar,
                                         max_pillars)
        self.anchors = AnchorGenerator(grid)
        self.assigner = TargetAssigner(self.anchors, reg_dim=reg_dim)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, k: int) -> Dict:
        sample = self.ds.get_sample(k, load=("lidar", "labels"))
        if self.bridge is not None:
            sample = self.bridge.apply_to_sample(sample)

        ids = ordered_agent_ids(sample, max_cav=self.max_cav)
        ego = sample.ego
        agents = []
        for aid in ids:
            ag = sample.agents[aid]
            if ag.lidar is None:
                continue
            T = agent_to_ego_matrix(sample, aid)      # uses SHARED poses
            pts = transform_points(ag.lidar, T)
            agents.append(self.voxelizer(np.asarray(pts, dtype=np.float32)))

        T_w2e = np.linalg.inv(ego.pose) if ego.pose is not None else None
        gt = labels_to_array(ego.labels, T_world_to_ego=T_w2e)
        targets = self.assigner(gt)
        return {"agents": agents, "targets": targets, "index": k}

    @staticmethod
    def collate(items: Sequence[Dict]) -> Dict:
        """Flatten (sample, agent) into one pillar batch; the model splits it
        again with agent_counts (ego first per sample).

        PillarVoxelizer emits coords (P, 2) = (row, col); PointPillarScatter
        wants (P, 3) = [agent, row, col] with `agent` the flat
        (sample, agent) index -- prepended here.
        """
        feats, coords, nums, counts = [], [], [], []
        cls_t, reg_t = [], []
        idx = 0
        for it in items:
            counts.append(len(it["agents"]))
            for ag in it["agents"]:
                rc = ag["coords"]
                agent_col = torch.full((rc.shape[0], 1), idx,
                                       dtype=rc.dtype)
                coords.append(torch.cat([agent_col, rc], dim=1))
                idx += 1
                feats.append(ag["features"])
                nums.append(ag["num_points"])
            cls_t.append(it["targets"]["cls_target"])
            reg_t.append(it["targets"]["reg_target"])
        return {
            "voxel_features": torch.cat(feats, dim=0),
            "voxel_coords": torch.cat(coords, dim=0),
            "voxel_num": torch.cat(nums, dim=0),
            "agent_counts": counts,
            "cls_target": torch.stack(cls_t, dim=0),
            "reg_target": torch.stack(reg_t, dim=0),
            "indices": [it["index"] for it in items],
        }
