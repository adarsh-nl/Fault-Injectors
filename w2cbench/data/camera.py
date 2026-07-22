"""
camera.py
---------
Dataset for the camera track: cooperative frames -> multi-camera batches.

The same frame decision as the LiDAR track
------------------------------------------
Images stay in their own camera's frame and their calibration travels with
them, so the lift places features on **each agent's own** BEV grid. The
per-agent warp into the ego frame happens later, in fusion, exactly as it does
for LiDAR. That is what keeps this intermediate fusion, and it is what leaves a
pose error unable to touch selection: an agent decides what to transmit using
features it computed in a frame the pose error does not affect.

Detection, not segmentation
---------------------------
Where2comm is a detection model on both tracks, so this dataset produces the
same ``gt_boxes`` the LiDAR one does and the same ``DetectionTester`` scores
it. ``cobevtbench``'s camera dataset rasterises a segmentation target instead,
because that is what CoBEVT's camera experiments report -- which is why that
package needed a second tester, a second loss and a second metric family, and
this one does not.

Calibration is load-bearing, not metadata
-----------------------------------------
``K`` and ``T_cam_to_agent`` are carried per camera because the lift projects
through them. A missing ``CameraCalib`` is an error rather than a silent
identity: an identity intrinsic would place every pixel as though the camera
had unit focal length, producing a BEV map that is wrong in a way no loss
curve distinguishes from a hard scene.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cpbench.data import (GridSpec, agent_to_ego_matrix, labels_to_array,
                          ordered_agent_ids, world_to_ego_matrix)
from cpbench.faults import DataFaultBridge

logger = logging.getLogger(__name__)


class W2CCameraDataset(Dataset):
    """Cooperative camera frames plus 3-D detection targets.

    Purpose
        Turn a ``src.datasets`` adapter into the batches
        :class:`~w2cbench.models.where2comm.Where2comm` consumes on the camera
        track. Everything except the sensor keys matches
        :class:`~w2cbench.data.lidar.W2CLidarDataset`.

    Inputs
    ------
    adapter       any ``src.datasets.BaseDataset`` that loads ``images``
    grid          the shared ``GridSpec`` -- the same one the lift and the
                  LiDAR track use, so both tracks land on one BEV grid
    max_cav       agent cap; the ego is always kept
    bridge        DataFaultBridge; None means a provably clean run
    camera_names  fixed camera order, or None to sort each agent's own
    categories    label categories to keep as ground truth

    Outputs
    -------
    ``__getitem__`` returns one scene::

        images          (n_agents, M, H, W, 3) or (n_agents, M, 3, H, W)
        intrinsics      (n_agents, M, 3, 3)
        extrinsics      (n_agents, M, 4, 4) camera-to-agent
        T_agent_to_ego  (n_agents, 4, 4)
        gt_boxes        (G, 7) ego-frame ground truth
        n_agents, frame, fault_records

    Example
    -------
    >>> from cpbench.data import GridSpec, SyntheticCameraCooperativeDataset
    >>> spec = GridSpec(voxel_size=(1.6, 1.6),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> adapter = SyntheticCameraCooperativeDataset(
    ...     n_frames=2, n_agents=2, n_cameras=2, image_size=(32, 32))
    >>> ds = W2CCameraDataset(adapter, spec, max_cav=2)
    >>> item = ds[0]
    >>> item["images"].shape[:2], item["intrinsics"].shape
    (torch.Size([2, 2]), torch.Size([2, 2, 3, 3]))
    >>> ds.is_clean
    True
    """

    def __init__(self, adapter, grid: GridSpec, max_cav: int = 5,
                 bridge: Optional[DataFaultBridge] = None,
                 camera_names: Optional[Sequence[str]] = None,
                 categories: Optional[Sequence[str]] = None) -> None:
        self.adapter = adapter
        self.grid = grid
        self.max_cav = int(max_cav)
        self.camera_names = list(camera_names) if camera_names else None
        self.categories = tuple(categories) if categories else None
        self.bridge = bridge or DataFaultBridge(
            None, fps=getattr(adapter, "fps", 10.0))

    @property
    def is_clean(self) -> bool:
        return self.bridge.is_clean

    def __len__(self) -> int:
        return len(self.adapter)

    def _cameras_of(self, agent) -> List[str]:
        return self.camera_names if self.camera_names is not None \
            else sorted(agent.images)

    def _agent_arrays(self, agent) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        names = self._cameras_of(agent)
        if not names:
            raise ValueError(
                f"agent {agent.agent_id!r} has no camera images; the camera "
                "track needs 'images' in the dataset's load set")
        images, intrinsics, extrinsics = [], [], []
        for name in names:
            images.append(np.asarray(agent.images[name]))
            calib = agent.cameras.get(name)
            if calib is None:
                raise ValueError(
                    f"agent {agent.agent_id!r} camera {name!r} has an image "
                    "but no CameraCalib. The lift projects through K and "
                    "T_cam_to_agent, so substituting an identity would place "
                    "every pixel as though the camera had unit focal length -- "
                    "wrong in a way no loss curve distinguishes from a hard "
                    "scene.")
            intrinsics.append(np.asarray(calib.K, dtype=np.float32))
            extrinsics.append(np.asarray(
                calib.T_cam_to_agent if calib.T_cam_to_agent is not None
                else np.eye(4), dtype=np.float32))
        return np.stack(images), np.stack(intrinsics), np.stack(extrinsics)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.bridge.load(self.adapter, index,
                                  load=("images", "labels"))
        agent_ids = ordered_agent_ids(sample, self.max_cav)

        images, intrinsics, extrinsics, transforms = [], [], [], []
        for agent_id in agent_ids:
            agent = sample.agents[agent_id]
            image, k, e = self._agent_arrays(agent)
            images.append(image)
            intrinsics.append(k)
            extrinsics.append(e)
            transforms.append(agent_to_ego_matrix(sample, agent_id))

        boxes = labels_to_array(sample.ego.labels, world_to_ego_matrix(sample),
                                self.categories)
        return {
            "images": torch.from_numpy(np.stack(images)),
            "intrinsics": torch.from_numpy(np.stack(intrinsics)),
            "extrinsics": torch.from_numpy(np.stack(extrinsics)),
            "T_agent_to_ego": torch.from_numpy(
                np.stack(transforms).astype(np.float32)),
            "gt_boxes": boxes,
            "n_agents": len(agent_ids),
            "frame": int(index),
            "fault_records": self.bridge.drain_records(),
        }
