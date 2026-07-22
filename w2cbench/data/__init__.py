"""Datasets and collation for w2cbench.

    lidar.py    W2CLidarDataset -- cooperative frames -> pillar batches
    collate.py  ragged agent counts -> one batch dict
    camera.py   W2CCameraDataset -- multi-camera frames + calibration

Sample-to-tensor conversions (labels, poses, agent ordering) live in
``cpbench.data.samples``: they are paper-agnostic, and a degrees-versus-radians
convention must have exactly one definition.
"""

from .collate import (camera_collator, collate_camera, collate_lidar,
                      lidar_collator)
from .lidar import W2CLidarDataset

__all__ = ["W2CLidarDataset", "W2CCameraDataset", "collate_lidar",
           "lidar_collator", "collate_camera", "camera_collator"]


def __getattr__(name: str):
    """The camera dataset is lazy: it reaches the torchvision-backed encoder
    only through config, but keeping the import symmetrical with
    ``w2cbench.models`` avoids a surprise for a LiDAR-only user."""
    if name == "W2CCameraDataset":
        from .camera import W2CCameraDataset
        return W2CCameraDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
