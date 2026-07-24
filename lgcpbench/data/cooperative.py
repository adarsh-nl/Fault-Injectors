"""
cooperative.py
--------------
Dataset producing LGCP frames, with physical fault injection built in.

This is the corruption plane (plane 1)
    Faults are applied here and ONLY here, by ``DataFaultBridge`` ->
    ``src.pipeline.FaultPipeline``, on the ``CooperativeSample`` BEFORE any
    tensor exists. Poses, LiDAR, images and the V2X link are corrupted at the
    point they would be corrupted in the real world. No model code, no
    scheduler, no metric ever corrupts a tensor.

    That single rule is what makes a measured robustness number attributable.
    If corruption could also happen mid-network, a drop in AP could be an
    artefact of where the injection was placed rather than of the fault.

Why LGCP needs one thing CoRA's dataset does not: positions
    ``corabench.data.cooperative.CoRADataset`` returns pillars, ids and
    targets. LGCP additionally needs each CAV's (x, y) in the ego frame,
    because path loss, interference range and the transmission schedule are
    all geometric.

    Crucially, those positions are read from the SHARED (and therefore
    possibly corrupted) poses, not from ground truth. A CAV that misreports
    its position makes the RSU schedule against a geometry that does not
    exist. Reading true poses here would silently immunise the scheduler
    against pose faults -- the exact failure mode this benchmark is for.

Why one frame per item, not a batch
    LGCP is defined per collaboration cycle: the RSU partitions, assigns,
    schedules and aggregates once per frame. Batching frames would mean
    batching independent RSU decisions, which has no meaning in the paper's
    model. Throughput comes from running frames in sequence, and the
    expensive part (encoding) is already batched across CAVs within a frame.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cpbench.data.preprocessing import GridSpec, PillarVoxelizer
from cpbench.data.samples import cooperative_gt_boxes
from cpbench.faults.bridge import DataFaultBridge, FaultRecord
from src.datasets.base import BaseDataset, Box3D

from ..orchestration.pipeline import FrameInput
from ..perception.protocol import AgentInputs
from .opencood_voxelizer import OpenCOODVoxelizer

logger = logging.getLogger(__name__)


# Ground truth is built by `cpbench.data.samples.cooperative_gt_boxes`: it
# merges the labels of EVERY agent from a freshly loaded CLEAN sample, so
# cooperation is rewarded (collaborator-revealed objects are in the answer
# key) and fault injection can never corrupt the ground truth.


class LGCPDataset:
    """Frames for the LGCP pipeline, corrupted upstream by the fault bridge.

    Purpose
        The single entry point from a ``src.datasets`` adapter (OPV2V,
        V2XSet, DAIR-V2X, synthetic) into ``LGCPPipeline``. Owns voxelisation
        and ego-frame transformation; owns no model and no metrics.

    Inputs
    ------
    adapter       any ``src.datasets.BaseDataset``.
    grid          GridSpec shared with the backbone, so the BEV geometry
                  cannot disagree.
    bridge        DataFaultBridge, or None for a clean run.
    max_agents    cap on CAVs per frame (ego always kept).
    comm_range_m  collaborators farther than this are excluded -- measured
                  with the CORRUPTED poses, since that is all the ego knows.
    categories    ground-truth category filter.

    Outputs
    -------
    ``__getitem__(k)`` -> (FrameInput, fault_records)

    Example
    -------
    >>> from cpbench.data.synthetic import SyntheticCooperativeDataset
    >>> from cpbench.data.preprocessing import GridSpec
    >>> spec = GridSpec((0.4, 0.4), (-38.4, -12.8, -3., 38.4, 12.8, 1.), 4)
    >>> ds = LGCPDataset(SyntheticCooperativeDataset(n_frames=2, n_agents=3), spec)
    >>> frame, faults = ds[0]
    >>> frame.agents.n_agents <= 3, faults
    (True, [])
    """

    def __init__(
        self,
        adapter: BaseDataset,
        grid: GridSpec,
        bridge: Optional[DataFaultBridge] = None,
        max_points_per_pillar: int = 32,
        max_pillars: int = 20000,
        max_agents: int = 5,
        comm_range_m: float = 70.0,
        categories: Optional[Sequence[str]] = None,
        feature_backend: str = "native",
        opencood_voxelizer: Optional["OpenCOODVoxelizer"] = None,
        gt_mode: str = "merge",
    ) -> None:
        self.adapter = adapter
        self.grid = grid
        self.gt_mode = gt_mode
        self.bridge = bridge or DataFaultBridge(
            None, fps=getattr(adapter, "fps", 10.0)
        )
        if feature_backend not in ("native", "opencood"):
            raise ValueError(
                f"feature_backend must be 'native' or 'opencood', got "
                f"{feature_backend!r}"
            )
        self.feature_backend = feature_backend
        # Voxelise ONCE, in the layout the chosen backend actually reads.
        # Producing both would double the per-frame preprocessing cost for no
        # benefit, since a backend never reads the other's tensors.
        self.voxelizer = (
            PillarVoxelizer(grid, max_points_per_pillar, max_pillars)
            if feature_backend == "native"
            else None
        )
        self.opencood_voxelizer = opencood_voxelizer
        if feature_backend == "opencood" and opencood_voxelizer is None:
            self.opencood_voxelizer = OpenCOODVoxelizer.from_grid_spec(
                grid, max_points_per_voxel=max_points_per_pillar
            )
        self.max_agents = int(max_agents)
        self.comm_range_m = float(comm_range_m)
        self.categories = list(categories) if categories else None

    def __len__(self) -> int:
        return len(self.adapter)

    @property
    def is_clean(self) -> bool:
        """True if no faults are configured -- the reference condition."""
        return self.bridge.is_clean

    def __getitem__(self, k: int) -> Tuple[FrameInput, List[FaultRecord]]:
        """Load, corrupt, voxelise and assemble one frame.

        Outputs
        -------
        (FrameInput, fault_records). The records are the audit trail drained
        from the bridge: what was corrupted, where, with which parameters.
        A clean run yields an empty list, which the benchmark asserts.
        """
        # ---- corruption plane: everything downstream sees only this ----
        sample = self.bridge.load(self.adapter, k, load=("lidar", "labels"))

        ego = sample.ego
        if ego.pose is None:
            raise ValueError(f"frame {k}: ego agent {sample.ego_id!r} has no pose")
        T_world_to_ego = np.linalg.inv(ego.pose)

        agent_ids = self._select_agents(sample, ego)
        clouds = [self._points_for(sample, aid) for aid in agent_ids]
        positions = self._positions(sample, agent_ids, T_world_to_ego)

        if self.feature_backend == "native":
            tensors = self._collate([self.voxelizer(p) for p in clouds])
            extra: Dict[str, Any] = {}
        else:
            tensors = {
                "features": torch.zeros(0, 1, 9),
                "coords": torch.zeros(0, 3, dtype=torch.long),
                "num_points": torch.zeros(0, dtype=torch.long),
            }
            extra = {
                "processed_lidar": self.opencood_voxelizer.collate(
                    [self.opencood_voxelizer.preprocess(p) for p in clouds]
                )
            }

        agents = AgentInputs(
            **tensors,
            n_agents=len(agent_ids),
            agent_ids=tuple(str(a) for a in agent_ids),
            positions=positions,
            ego_index=0,
            extra=extra,
        )
        frame = FrameInput(
            index=int(k),
            agents=agents,
            gt_boxes=cooperative_gt_boxes(
                self.adapter, k, categories=self.categories,
                point_range=self.grid.point_range, mode=self.gt_mode),
        )
        return frame, self.bridge.drain_records()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _select_agents(self, sample, ego) -> List[str]:
        """Ego first, then collaborators inside the communication range.

        Range is judged from the shared -- possibly corrupted -- poses, which
        is exactly the information the ego has. A large pose error can push a
        real neighbour out of range, or pull a distant one in, and both are
        genuine consequences worth measuring.
        """
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
        return agent_ids

    def _points_for(self, sample, agent_id: str) -> np.ndarray:
        """One CAV's points, already warped into the ego frame.

        The warp uses the SHARED pose, so pose corruption becomes feature
        misalignment here -- which is how a plane-1 fault reaches the model
        without any model code being fault-aware.
        """
        points = (
            sample.ego.lidar
            if agent_id == sample.ego_id
            else sample.lidar_in_ego_frame(agent_id)
        )
        if points is None:
            return np.zeros((0, 4), dtype=np.float32)
        return points

    def _positions(
        self, sample, agent_ids: Sequence[str], T_world_to_ego: np.ndarray
    ) -> np.ndarray:
        """(V, 2) CAV positions in the ego frame, from the shared poses."""
        out = np.zeros((len(agent_ids), 2), dtype=np.float64)
        for i, aid in enumerate(agent_ids):
            pose = sample.agents[aid].pose
            if pose is None:
                continue
            world = np.append(pose[:3, 3], 1.0)
            out[i] = (T_world_to_ego @ world)[:2]
        return out

    @staticmethod
    def _collate(pillars: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Flatten per-agent pillar dicts into one agent batch.

        ``PointPillarScatter`` expects coords as (P, 3) = [agent, row, col],
        so the agent index is prepended here -- the same convention corabench
        uses, so the shared encoder needs no LGCP-specific branch.
        """
        features, coords, num_points = [], [], []
        for agent_index, item in enumerate(pillars):
            n = int(item["features"].shape[0])
            if n == 0:
                continue
            features.append(item["features"])
            agent_col = torch.full((n, 1), agent_index, dtype=torch.long)
            coords.append(torch.cat([agent_col, item["coords"].long()], dim=1))
            num_points.append(item["num_points"])

        if not features:
            # Every CAV was dropped or has empty LiDAR -- a legitimate fault
            # outcome (agent_drop with a high rate), not an error.
            return {
                "features": torch.zeros(0, 1, 9),
                "coords": torch.zeros(0, 3, dtype=torch.long),
                "num_points": torch.zeros(0, dtype=torch.long),
            }
        return {
            "features": torch.cat(features, dim=0).float(),
            "coords": torch.cat(coords, dim=0),
            "num_points": torch.cat(num_points, dim=0),
        }
