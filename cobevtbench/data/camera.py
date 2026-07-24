"""
camera.py
---------
Dataset for the camera track: cooperative samples -> CoBEVT camera batches.

Nothing here is fault-aware. Corruption has already happened, upstream, on
the ``CooperativeSample`` that ``DataFaultBridge`` returns -- so a dropped
camera arrives as a black image, a miscalibrated one as a perturbed ``K``,
and a mispositioned collaborator as a wrong pose. The dataset reads whatever
it is given and reports what was injected, which is what makes a measured
robustness number attributable to the fault rather than to this code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cpbench.data import BEVGrid, BEVRasterizer
from cpbench.faults import DataFaultBridge

from .transforms import (agent_to_ego_matrix, cooperative_gt_boxes,
                         ordered_agent_ids, world_to_ego_matrix)

logger = logging.getLogger(__name__)

# Segmentation class layouts, matching BevSegHead's targets.
DYNAMIC_CLASSES = ("background", "vehicle")
STATIC_CLASSES = ("background", "drivable_area", "lane")


class CoBEVTCameraDataset(Dataset):
    """Cooperative camera frames plus BEV segmentation targets.

    Purpose
        Turn a ``src.datasets`` adapter into the batches
        :class:`~cobevtbench.models.cobevt_camera.CoBEVTCamera` consumes.

    Inputs
    ------
    adapter     any ``src.datasets.BaseDataset``
    bev_grid    output BEVGrid the targets are rasterised on. Must match the
                model's output resolution -- CoBEVT: 256x256 over 100 m.
    max_cav     agent cap (CoBEVT: 5); ego is always kept
    bridge      DataFaultBridge; None means a provably clean run
    target      ``"dynamic"`` or ``"static"``
    camera_names  which cameras to read, in a fixed order. Order matters --
                the model has no camera-identity input beyond the extrinsics,
                so a rig that reorders between samples trains against
                inconsistent geometry.
    categories  label categories to rasterise as the foreground class

    Outputs
    -------
    ``__getitem__`` returns one *scene*, with a variable agent count:

    ``images``        (n_agents, M, H, W, 3) uint8
    ``intrinsics``    (n_agents, M, 3, 3)
    ``extrinsics``    (n_agents, M, 4, 4)  camera -> that agent's own frame
    ``T_agent_to_ego`` (n_agents, 4, 4)
    ``target``        (H_bev, W_bev) int64 label map, ego frame
    ``n_agents``      int
    ``frame``         source frame index
    ``fault_records`` whatever the bridge injected for this frame

    Batching is done by :func:`~cobevtbench.data.collate.collate_camera`.

    Example
    -------
    >>> from cpbench.data import SyntheticCameraCooperativeDataset, BEVGrid
    >>> adapter = SyntheticCameraCooperativeDataset(
    ...     n_frames=2, n_agents=2, image_size=(32, 32))
    >>> ds = CoBEVTCameraDataset(adapter, BEVGrid(16, 16, 40.0, 40.0), max_cav=2)
    >>> item = ds[0]
    >>> item["images"].shape, item["target"].shape
    (torch.Size([2, 4, 32, 32, 3]), torch.Size([16, 16]))
    >>> ds.is_clean
    True
    """

    def __init__(self, adapter, bev_grid: BEVGrid, max_cav: int = 5,
                 bridge: Optional[DataFaultBridge] = None,
                 target: str = "dynamic",
                 camera_names: Optional[Sequence[str]] = None,
                 categories: Optional[Sequence[str]] = None,
                 gt_mode: str = "merge") -> None:
        self.adapter = adapter
        self.bev_grid = bev_grid
        self.gt_mode = gt_mode
        self.max_cav = int(max_cav)
        self.target = target
        self.camera_names = list(camera_names) if camera_names else None
        self.categories = tuple(categories) if categories else None
        self.bridge = bridge or DataFaultBridge(
            None, fps=getattr(adapter, "fps", 10.0))
        self.class_names = (DYNAMIC_CLASSES if target == "dynamic"
                            else STATIC_CLASSES)
        self.rasterizer = BEVRasterizer(bev_grid,
                                        n_classes=len(self.class_names))

    @property
    def is_clean(self) -> bool:
        """True when no fault can be injected -- the reference condition."""
        return self.bridge.is_clean

    def __len__(self) -> int:
        return len(self.adapter)

    # -- per-agent extraction ----------------------------------------------

    def _cameras_of(self, agent) -> List[str]:
        if self.camera_names is not None:
            return self.camera_names
        return sorted(agent.images)

    def _agent_arrays(self, agent) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        names = self._cameras_of(agent)
        if not names:
            raise ValueError(
                f"agent {agent.agent_id!r} has no camera images; the camera "
                "track needs `images` in the dataset's load set")
        images, intrinsics, extrinsics = [], [], []
        for name in names:
            images.append(np.asarray(agent.images[name]))
            calib = agent.cameras.get(name)
            if calib is None:
                raise ValueError(
                    f"agent {agent.agent_id!r} camera {name!r} has an image but "
                    "no CameraCalib; SinBEVT lifts by ray matching and cannot "
                    "run without K and T_cam_to_agent")
            intrinsics.append(np.asarray(calib.K, dtype=np.float32))
            extrinsics.append(np.asarray(
                calib.T_cam_to_agent if calib.T_cam_to_agent is not None
                else np.eye(4), dtype=np.float32))
        return (np.stack(images), np.stack(intrinsics), np.stack(extrinsics))

    # -- Dataset interface --------------------------------------------------

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.bridge.load(self.adapter, index,
                                  load=("images", "labels"))
        agent_ids = ordered_agent_ids(sample, self.max_cav)

        images, intrinsics, extrinsics, transforms = [], [], [], []
        for agent_id in agent_ids:
            agent = sample.agents[agent_id]
            image, K, T_cam = self._agent_arrays(agent)
            images.append(image)
            intrinsics.append(K)
            extrinsics.append(T_cam)
            transforms.append(agent_to_ego_matrix(sample, agent_id))

        boxes = cooperative_gt_boxes(self.adapter, index,
                                     categories=self.categories,
                                     point_range=self.bev_grid.point_range,
                                     mode=self.gt_mode)
        target = self.rasterizer.rasterize(boxes)

        return {
            "images": torch.from_numpy(np.stack(images)),
            "intrinsics": torch.from_numpy(np.stack(intrinsics)),
            "extrinsics": torch.from_numpy(np.stack(extrinsics)),
            "T_agent_to_ego": torch.from_numpy(
                np.stack(transforms).astype(np.float32)),
            "target": torch.from_numpy(target),
            "gt_boxes": boxes,
            "n_agents": len(agent_ids),
            "frame": int(index),
            "fault_records": self.bridge.drain_records(),
        }
