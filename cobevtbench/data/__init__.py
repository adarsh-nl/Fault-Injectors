"""
Datasets, agent-axis collation and BEV target rasterization.

The corruption plane sits upstream of everything here: a `DataFaultBridge`
corrupts the CooperativeSample before any tensor is built, so nothing in this
sub-package is fault-aware.

Contents
--------
transforms  frame and unit conversions (Box3D degrees -> (N,7) radians, ego)
camera      CoBEVTCameraDataset: cooperative frames -> camera batches
lidar       CoBEVTLidarDataset: cooperative frames -> pillar batches
collate     batch scenes with different agent counts
"""

from .camera import (DYNAMIC_CLASSES, STATIC_CLASSES, CoBEVTCameraDataset)
from .collate import (camera_collator, collate_camera, collate_lidar,
                      lidar_collator)
from .lidar import CoBEVTLidarDataset
from .transforms import (agent_to_ego_matrix, labels_to_array,
                         ordered_agent_ids, world_to_ego_matrix)

__all__ = ["CoBEVTCameraDataset", "CoBEVTLidarDataset",
           "collate_camera", "collate_lidar", "camera_collator",
           "lidar_collator", "labels_to_array", "agent_to_ego_matrix",
           "world_to_ego_matrix", "ordered_agent_ids",
           "DYNAMIC_CLASSES", "STATIC_CLASSES"]
