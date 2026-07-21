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

`SyntheticCameraCooperativeDataset` extends this to the camera modality with
a minimal but geometrically *consistent* renderer -- see its docstring for
why consistency, rather than realism, is the property that matters here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.datasets.base import AgentFrame, BaseDataset, Box3D, CameraCalib


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


class SyntheticCameraCooperativeDataset(SyntheticCooperativeDataset):
    """Synthetic cooperative scenes with cameras as well as LiDAR.

    Purpose
        Keep the camera stack -- image backbone, camera-to-BEV lifting,
        multi-agent fusion, segmentation head -- fully exercisable with no
        dataset download, matching what ``SyntheticCooperativeDataset``
        already does for the LiDAR stack.

    Why a real projection and not random noise
        The point of this data is not realism, it is **geometric
        consistency**: image content, camera intrinsics, camera extrinsics,
        agent poses and 3-D labels all agree. Architectures that lift camera
        features to BEV by matching ray directions (CVT, CoBEVT's SinBEVT)
        have nothing to learn from noise, so a smoke test on random images
        cannot distinguish a correct implementation from one whose extrinsics
        are transposed. Here a box really does appear in the camera that can
        see it, at the pixels where it belongs -- so a wrong sign in the
        geometry shows up as a loss that will not descend.

    Inputs
    ------
    n_cameras     cameras per agent, spread evenly over 360 degrees (4 in
                  CoBEVT, giving full coverage with a 90-degree FOV).
    image_size    (height, width). Defaults to 64x64 to keep tests fast;
                  CoBEVT uses 512x512.
    fov_degrees   horizontal field of view per camera.
    cam_height    camera mount height above the agent origin, metres.

    Outputs
    -------
    Each ``AgentFrame`` additionally carries ``images`` (name -> (H, W, 3)
    uint8 RGB) and ``cameras`` (name -> CameraCalib with a real ``K`` and
    ``T_cam_to_agent``).

    Shapes
    ------
    images[name]                  (H, W, 3) uint8
    cameras[name].K               (3, 3)
    cameras[name].T_cam_to_agent  (4, 4)

    Example
    -------
    >>> ds = SyntheticCameraCooperativeDataset(n_frames=1, n_agents=2,
    ...                                        image_size=(32, 32))
    >>> sample = ds.get_sample(0, load=("images", "labels"))
    >>> agent = sample.agents["agent0"]
    >>> sorted(agent.images)
    ['camera0', 'camera1', 'camera2', 'camera3']
    >>> agent.images["camera0"].shape, agent.images["camera0"].dtype.name
    ((32, 32, 3), 'uint8')
    >>> agent.cameras["camera0"].K.shape
    (3, 3)
    """

    name = "synthetic_camera"

    def __init__(self, n_cameras: int = 4,
                 image_size: Tuple[int, int] = (64, 64),
                 fov_degrees: float = 90.0, cam_height: float = 1.6,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_cameras = int(n_cameras)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.fov_degrees = float(fov_degrees)
        self.cam_height = float(cam_height)
        if self.n_cameras < 1:
            raise ValueError(f"need at least 1 camera, got {self.n_cameras}")

    # -- camera rig ---------------------------------------------------------

    def camera_names(self) -> List[str]:
        return [f"camera{i}" for i in range(self.n_cameras)]

    def intrinsics(self) -> np.ndarray:
        """(3, 3) pinhole K shared by every camera in the rig."""
        height, width = self.image_size
        focal = (width / 2.0) / np.tan(np.radians(self.fov_degrees) / 2.0)
        return np.array([[focal, 0.0, width / 2.0],
                         [0.0, focal, height / 2.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def extrinsics(self, index: int) -> np.ndarray:
        """(4, 4) T_cam_to_agent for camera `index`.

        The agent frame is x-forward, y-left, z-up; the camera frame is the
        OpenCV convention x-right, y-down, z-forward. The rotation columns
        below are the camera axes expressed in agent coordinates, which is
        the direction that makes ``T_cam_to_agent`` (not its inverse).
        """
        theta = 2.0 * np.pi * index / self.n_cameras
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        T = np.eye(4)
        T[:3, 0] = (sin_t, -cos_t, 0.0)      # camera +x (right)
        T[:3, 1] = (0.0, 0.0, -1.0)          # camera +y (down)
        T[:3, 2] = (cos_t, sin_t, 0.0)       # camera +z (forward)
        T[:3, 3] = (0.0, 0.0, self.cam_height)
        return T

    # -- rendering ----------------------------------------------------------

    @staticmethod
    def _box_corners_3d(box: np.ndarray) -> np.ndarray:
        """(8, 3) world-frame corners of one (7,) box [x,y,z,l,w,h,yaw]."""
        length, width, height = box[3], box[4], box[5]
        signs = np.array(np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
                         ).T.reshape(-1, 3)
        local = signs * np.array([length, width, height])
        cos_y, sin_y = np.cos(box[6]), np.sin(box[6])
        rotated = np.empty_like(local)
        rotated[:, 0] = local[:, 0] * cos_y - local[:, 1] * sin_y
        rotated[:, 1] = local[:, 0] * sin_y + local[:, 1] * cos_y
        rotated[:, 2] = local[:, 2]
        return rotated + box[:3]

    def _render(self, boxes: np.ndarray, T_agent_to_world: np.ndarray,
                cam_index: int, rng: np.random.Generator) -> np.ndarray:
        """Render one camera view as (H, W, 3) uint8 RGB."""
        height, width = self.image_size
        # Background: a vertical gradient plus mild noise, so the image has
        # non-degenerate low-level statistics for a conv backbone.
        gradient = np.linspace(90, 150, height, dtype=np.float64)[:, None]
        image = np.repeat(gradient, width, axis=1)[:, :, None] * np.ones(3)
        image += rng.normal(0.0, 3.0, image.shape)

        K = self.intrinsics()
        T_world_to_cam = np.linalg.inv(
            T_agent_to_world @ self.extrinsics(cam_index))
        for box in boxes:
            corners = self._box_corners_3d(box)                    # (8, 3)
            homogeneous = np.hstack([corners, np.ones((8, 1))])
            in_cam = (T_world_to_cam @ homogeneous.T).T[:, :3]     # (8, 3)
            in_front = in_cam[:, 2] > 0.1
            if not in_front.any():
                continue                                    # behind the camera
            projected = (K @ in_cam[in_front].T).T
            pixels = projected[:, :2] / projected[:, 2:3]
            c0 = int(np.clip(np.floor(pixels[:, 0].min()), 0, width))
            c1 = int(np.clip(np.ceil(pixels[:, 0].max()), 0, width))
            r0 = int(np.clip(np.floor(pixels[:, 1].min()), 0, height))
            r1 = int(np.clip(np.ceil(pixels[:, 1].max()), 0, height))
            if c1 <= c0 or r1 <= r0:
                continue
            image[r0:r1, c0:c1] = (210.0, 70.0, 60.0)
        return np.clip(image, 0, 255).astype(np.uint8)

    # -- BaseDataset interface ----------------------------------------------

    def _load_agent(self, agent_id: str, k: int, load) -> Optional[AgentFrame]:
        frame = super()._load_agent(agent_id, k, load)
        if frame is None or "images" not in load:
            return frame
        idx = int(agent_id.replace("agent", ""))
        boxes, _ = self._scene(k)
        rng = self._rng(k * 1000 + idx + 7919)      # distinct from the lidar rng
        K = self.intrinsics()
        images: Dict[str, np.ndarray] = {}
        cameras: Dict[str, CameraCalib] = {}
        for cam_index, cam_name in enumerate(self.camera_names()):
            images[cam_name] = self._render(boxes, frame.pose, cam_index, rng)
            cameras[cam_name] = CameraCalib(
                K=K.copy(), T_cam_to_agent=self.extrinsics(cam_index),
                name=cam_name)
        frame.images = images
        frame.cameras = cameras
        return frame
