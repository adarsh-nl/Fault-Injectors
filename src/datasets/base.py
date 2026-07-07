"""
base.py
-------
The common sample model every dataset adapter normalises into, and the
`BaseDataset` interface the fault pipeline consumes.

Canonical conventions (every adapter MUST produce these)
--------------------------------------------------------
* An AGENT is one cooperating platform: a connected vehicle, a roadside
  infrastructure unit, or a drone. Exactly one agent per sample is the ego.
* Each agent has an AGENT FRAME: a right-handed metric frame rigidly
  attached to the platform (x forward, y left, z up where the dataset
  allows). All of that agent's sensor data is expressed in its agent frame:
    - `lidar`  : (N, C>=3) float32, columns x, y, z [, intensity, ...]
    - `labels` : 3-D boxes (Box3D), frame given per box ('agent' | 'world')
* `pose` is the 4x4 rigid transform T_agent_to_world mapping agent-frame
  points into the dataset's world frame. Cross-agent fusion is
      T_j_to_ego = inv(T_ego_to_world) @ T_j_to_world
  which is exactly what pose-error injection corrupts.
* Images are (H, W, 3) uint8 RGB, one per named camera; each camera has an
  intrinsic K (3, 3) and extrinsic T_cam_to_agent (4, 4).
* Angles in degrees, distances in metres, timestamps in seconds.

Adapters may leave a field None when the dataset genuinely lacks it; the
fault pipeline skips faults whose inputs are missing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

AGENT_TYPES = ('vehicle', 'infrastructure', 'drone')


# ── Geometry helpers ────────────────────────────────────────────────────────

def transform_points(points, T):
    """
    Apply a 4x4 rigid transform to the xyz columns of an (N, C>=3) array.
    Extra columns (intensity, ...) pass through. Returns a new array.
    """
    pts = np.asarray(points)
    out = pts.copy()
    if len(pts):
        xyz1 = np.hstack([pts[:, :3], np.ones((len(pts), 1), dtype=pts.dtype)])
        out[:, :3] = (np.asarray(T, dtype=np.float64) @ xyz1.T).T[:, :3]
    return out


# ── Sample model ────────────────────────────────────────────────────────────

@dataclass
class CameraCalib:
    """Intrinsics + mounting of one camera on one agent."""
    K: np.ndarray                        # (3, 3)
    T_cam_to_agent: Optional[np.ndarray] = None   # (4, 4)
    name: str = ''


@dataclass
class Box3D:
    """One 3-D bounding box annotation."""
    center: np.ndarray                   # (3,) x, y, z of box centre
    size: np.ndarray                     # (3,) l, w, h in metres
    yaw: float = 0.0                     # degrees
    roll: float = 0.0
    pitch: float = 0.0
    category: str = ''
    track_id: str = ''
    frame: str = 'agent'                 # 'agent' or 'world'
    extra: dict = field(default_factory=dict)   # visibility, num points, ...


@dataclass
class AgentFrame:
    """Everything one agent contributes to one cooperative sample."""
    agent_id: str
    agent_type: str = 'vehicle'          # one of AGENT_TYPES
    is_ego: bool = False
    timestamp: Optional[float] = None
    pose: Optional[np.ndarray] = None    # (4, 4) T_agent_to_world
    lidar: Optional[np.ndarray] = None   # (N, C) in the AGENT frame
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    cameras: Dict[str, CameraCalib] = field(default_factory=dict)
    labels: List[Box3D] = field(default_factory=list)
    faults: dict = field(default_factory=dict)   # log of injected faults


@dataclass
class CooperativeSample:
    """One multi-agent frame: what the ego would fuse at time k."""
    frame_index: int
    ego_id: str
    agents: Dict[str, AgentFrame] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def ego(self):
        return self.agents[self.ego_id]

    def lidar_in_ego_frame(self, agent_id):
        """
        Agent `agent_id`'s point cloud expressed in the EGO agent frame,
        via the (possibly fault-corrupted) shared poses:
            T = inv(T_ego_to_world) @ T_agent_to_world
        This is the fusion-side view: pose error shows up here as
        misaligned geometry.
        """
        agent = self.agents[agent_id]
        if agent_id == self.ego_id:
            return agent.lidar
        if agent.lidar is None or agent.pose is None or self.ego.pose is None:
            raise ValueError(
                f'need lidar + poses to warp {agent_id!r} into the ego frame')
        T = np.linalg.inv(self.ego.pose) @ agent.pose
        return transform_points(agent.lidar, T)


# ── Dataset interface ───────────────────────────────────────────────────────

class BaseDataset:
    """
    Interface every dataset adapter implements.

    Subclasses set `name` and `fps`, and implement:
        agent_ids(self)                      -> list of agent id strings
        __len__(self)                        -> number of ego frames
        _load_agent(self, agent_id, k, load) -> AgentFrame or None
                                                (None = agent absent at k)
    `load` is a tuple naming the heavy payloads to actually read from disk,
    any of: 'lidar', 'images', 'labels'. Poses, calib and timestamps are
    always loaded (they are cheap and every fault needs them).
    """

    name = 'base'
    fps = 10.0

    # -- to implement -------------------------------------------------------

    def agent_ids(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _load_agent(self, agent_id, k, load):
        raise NotImplementedError

    # -- shared behaviour ---------------------------------------------------

    @property
    def ego_id(self):
        """Default ego: first agent id. Adapters override where needed."""
        return self.agent_ids()[0]

    def get_sample(self, k, agents=None, load=('lidar', 'images', 'labels')):
        """
        Build the cooperative sample at ego frame k.

        Parameters
        ----------
        k      : int  frame index in [0, len(self)).
        agents : optional list of agent ids to include (default: all).
        load   : which heavy payloads to read ('lidar', 'images', 'labels').

        Returns
        -------
        CooperativeSample. Agents absent at frame k are silently omitted.
        """
        if not (0 <= k < len(self)):
            raise IndexError(f'frame {k} out of range [0, {len(self)})')
        ids = list(agents) if agents is not None else self.agent_ids()
        sample = CooperativeSample(frame_index=k, ego_id=self.ego_id)
        for aid in ids:
            frame = self._load_agent(aid, k, tuple(load))
            if frame is not None:
                frame.is_ego = (aid == self.ego_id)
                sample.agents[aid] = frame
        if self.ego_id in sample.agents:
            sample.meta['fps'] = self.fps
        return sample

    def __repr__(self):
        return (f'{type(self).__name__}(name={self.name!r}, '
                f'agents={self.agent_ids()}, frames={len(self)})')
