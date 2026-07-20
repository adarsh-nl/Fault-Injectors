"""
synthetic.py
------------
In-memory synthetic cooperative scenes: a `src.datasets.BaseDataset` adapter
that needs no files. Used by unit tests, CI and smoke training runs, and as
a minimal reference for writing new adapters.

Each frame contains `n_objects` car-sized boxes on a ground plane; every
agent observes points sampled on the box surfaces plus ground clutter, from
its own pose (so cross-agent fusion and pose-error faults behave exactly as
with real data).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.datasets.base import AgentFrame, BaseDataset, Box3D


class SyntheticCooperativeDataset(BaseDataset):
    """Deterministic random cooperative scenes.

    Parameters
    ----------
    n_frames, n_agents, n_objects : scene size.
    area  : half-extent [m] of the square world the objects live in.
    seed  : scene generator seed (scenes are a pure function of it).

    Example
    -------
    >>> ds = SyntheticCooperativeDataset(n_frames=2, n_agents=2)
    >>> sample = ds.get_sample(0)
    >>> sorted(sample.agents)
    ['agent0', 'agent1']
    """

    name = "synthetic"
    fps = 10.0

    def __init__(self, n_frames: int = 12, n_agents: int = 3,
                 n_objects: int = 4, area: float = 15.0,
                 points_per_object: int = 220, clutter: int = 500,
                 seed: int = 0) -> None:
        self.n_frames = int(n_frames)
        self.n_agents = int(n_agents)
        self.n_objects = int(n_objects)
        self.area = float(area)
        self.points_per_object = int(points_per_object)
        self.clutter = int(clutter)
        self.seed = int(seed)

    # -- BaseDataset interface ----------------------------------------------

    def agent_ids(self):
        return [f"agent{i}" for i in range(self.n_agents)]

    def __len__(self) -> int:
        return self.n_frames

    def _rng(self, k: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, k]))

    def _scene(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame boxes (G, 7) and agent poses (N, 3)[x, y, yaw]."""
        rng = self._rng(k)
        boxes = np.zeros((self.n_objects, 7), dtype=np.float64)
        boxes[:, :2] = rng.uniform(-self.area, self.area, (self.n_objects, 2))
        boxes[:, 2] = -1.0
        boxes[:, 3:6] = (3.9, 1.6, 1.56)
        boxes[:, 6] = rng.uniform(-np.pi, np.pi, self.n_objects)
        poses = np.zeros((self.n_agents, 3))
        poses[:, :2] = rng.uniform(-self.area / 2, self.area / 2,
                                   (self.n_agents, 2))
        poses[:, 2] = rng.uniform(-np.pi, np.pi, self.n_agents)
        return boxes, poses

    @staticmethod
    def _pose_matrix(x: float, y: float, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:3, 3] = [x, y, 1.9]
        return T

    def _sample_points(self, boxes: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
        """World-frame points on box surfaces + ground clutter, (N, 4)."""
        pts = []
        for box in boxes:
            n = self.points_per_object
            local = rng.uniform(-0.5, 0.5, (n, 3)) * box[3:6]
            face = rng.integers(0, 3, n)
            sign = rng.choice([-0.5, 0.5], n)
            for axis in range(3):
                m = face == axis
                local[m, axis] = sign[m] * box[3 + axis]
            c, s = np.cos(box[6]), np.sin(box[6])
            world = np.empty((n, 4), dtype=np.float64)
            world[:, 0] = box[0] + local[:, 0] * c - local[:, 1] * s
            world[:, 1] = box[1] + local[:, 0] * s + local[:, 1] * c
            world[:, 2] = box[2] + local[:, 2]
            world[:, 3] = rng.uniform(0.1, 1.0, n)
            pts.append(world)
        ground = np.empty((self.clutter, 4))
        ground[:, :2] = rng.uniform(-self.area * 1.2, self.area * 1.2,
                                    (self.clutter, 2))
        ground[:, 2] = -1.8 + rng.normal(0, 0.02, self.clutter)
        ground[:, 3] = rng.uniform(0.0, 0.3, self.clutter)
        pts.append(ground)
        return np.concatenate(pts)

    def _load_agent(self, agent_id: str, k: int, load) -> Optional[AgentFrame]:
        idx = int(agent_id.replace("agent", ""))
        boxes, poses = self._scene(k)
        rng = self._rng(k * 1000 + idx + 1)
        T = self._pose_matrix(*poses[idx])
        frame = AgentFrame(agent_id=agent_id, timestamp=k / self.fps, pose=T)
        if "lidar" in load:
            world_pts = self._sample_points(boxes, rng)
            T_inv = np.linalg.inv(T)
            xyz1 = np.hstack([world_pts[:, :3],
                              np.ones((len(world_pts), 1))])
            local = (T_inv @ xyz1.T).T[:, :3]
            frame.lidar = np.hstack([local, world_pts[:, 3:4]]) \
                .astype(np.float32)
        if "labels" in load:
            frame.labels = [Box3D(center=b[:3].copy(), size=b[3:6].copy(),
                                  yaw=float(np.degrees(b[6])),
                                  category="car", frame="world")
                            for b in boxes]
        return frame
