"""
fault_injectors
---------------
Fault injection for RGB-LiDAR multi-modal 3D object detection on Griffin.

Each failure mode lives in its own module and exposes a small, composable API.
A clean (perfectly synchronised) sample is the pair:

    X = (image, points)

where `image` is an (H, W, 3) RGB array and `points` is an (N, C) LiDAR array.
A fault injector takes a clean sample (or a stream of samples) and returns a
corrupted sample following the formal definitions in the project's fault spec.

Modules
-------
missing_modality      : sensor dropout (Bernoulli-gated zeroing / emptying)
temporal_misalignment : stale image pairing via index shifting
sensor_occlusion      : sensor-surface soiling / damage (dirt, scratch, crack)
brightness/darkness   : camera exposure corruptions (EXACT MultiCorrupt)
fog/snow              : camera adverse-weather corruptions (EXACT MultiCorrupt)
lidar_fog             : LiDAR fog (EXACT MultiCorrupt simulate_fog; runs on Griffin)
lidar_points_reduce   : LiDAR uniform point dropout (EXACT MultiCorrupt)
lidar_beams_reduce    : LiDAR beam reduction (EXACT needs 32-beam ring col; + Griffin-native)
lidar_snow            : LiDAR snow (MultiCorrupt/Hahner physics verbatim; two documented
                        Griffin-derived inputs: geometric ground mask, elevation channels)
"""

from .missing_modality import (
    MissingModalityInjector,
    drop_image,
    drop_points,
    bernoulli_mask,
)
from .temporal_misalignment import (
    TemporalMisalignmentInjector,
    sample_index_shift,
    physical_displacement,
)
from .sensor_occlusion import (
    SensorOcclusionInjector,
    OcclusionConfig,
)
# EXACT MultiCorrupt camera corruptions (wrap verbatim _mc_image.py)
from .brightness import BrightnessInjector
from .darkness import DarknessInjector
from .fog import FogInjector
from .snow import SnowInjector
# EXACT MultiCorrupt LiDAR corruptions (wrap verbatim _mc_lidar.py)
from .lidar_fog import LidarFogInjector
from .lidar_points_reduce import PointsReductionInjector
from .lidar_beams_reduce import BeamReductionInjector, BeamReductionInjectorGriffin
from .lidar_snow import LidarSnowInjector

__all__ = [
    'MissingModalityInjector',
    'drop_image',
    'drop_points',
    'bernoulli_mask',
    'TemporalMisalignmentInjector',
    'sample_index_shift',
    'physical_displacement',
    'SensorOcclusionInjector',
    'OcclusionConfig',
    'BrightnessInjector',
    'DarknessInjector',
    'FogInjector',
    'SnowInjector',
    'LidarFogInjector',
    'PointsReductionInjector',
    'BeamReductionInjector', 'BeamReductionInjectorGriffin',
    'LidarSnowInjector',
]