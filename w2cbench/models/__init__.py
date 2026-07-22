"""Where2comm's model stages.

    encoder.py         the Stage-1 contract every track implements
    encoder_lidar.py   PointPillars (LiDAR track)
    encoder_camera.py  ResNet + depth-splat lifting (camera track, step 15)
    confidence.py      the spatial confidence generator (step 4)
    where2comm.py      the K-round orchestrator (step 9)

Only the encoder is modality-specific; everything after it operates on the
BEV feature map and cannot tell how that map was produced.
"""

from .confidence import SpatialConfidenceGenerator
from .encoder import ObservationEncoder
from .encoder_lidar import LidarPillarEncoder
from .where2comm import Where2comm

__all__ = ["ObservationEncoder", "LidarPillarEncoder", "CameraEncoder",
           "SpatialConfidenceGenerator", "Where2comm", "BEVLifting",
           "DepthSplatLifting"]


def __getattr__(name: str):
    """Expose the camera track lazily; it needs torchvision, and a LiDAR-only
    run should not pay for that import."""
    if name == "CameraEncoder":
        from .encoder_camera import CameraEncoder
        return CameraEncoder
    if name in ("BEVLifting", "DepthSplatLifting"):
        from . import lifting
        return getattr(lifting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
