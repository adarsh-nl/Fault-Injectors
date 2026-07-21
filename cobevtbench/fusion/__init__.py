"""
Multi-agent fusion and the geometry that feeds it.

FuseBEVT is modality-agnostic by construction: it consumes (B, L, C, H, W)
plus an agent mask and returns (B, C, H, W). That is the only thing the camera
and LiDAR tracks share, and it is also exactly the contract that makes it a
drop-in OpenCOOD fusion module.

Contents
--------
fusebevt   FuseBEVT: depth x FAX self-attention, agent pooling, projection
geometry   regroup (flat agent stack -> padded batch) and SpatialTransform
           (CoBEVT's STTF warp into the ego frame)

camera_embedding  the ray-direction fields that perform camera-to-BEV
           lifting -- and the reason K and T are load-bearing tensors

compression       the bandwidth knob: a 1x1 bottleneck on the
           transmitted BEV map (the paper's 0x-64x ablation)
"""

from .compression import NaiveCompressor
from .fusebevt import POOL_MASKED_MEAN, POOL_MEAN, FuseBEVT
from .camera_embedding import CameraGeometryEmbedding, pixel_grid
from .geometry import SpatialTransform, regroup

__all__ = ["FuseBEVT", "POOL_MEAN", "POOL_MASKED_MEAN",
           "SpatialTransform", "regroup",
           "CameraGeometryEmbedding", "pixel_grid", "NaiveCompressor"]
