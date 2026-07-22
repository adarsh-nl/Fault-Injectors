"""
fault_injectors
---------------
Dataset-agnostic fault injection for multi-modal, multi-agent 3D perception.

Every injector operates on plain arrays and poses -- an (H, W, 3) RGB image,
an (N, C) LiDAR array, a (4, 4) pose matrix -- so they work with ANY autonomous
driving dataset (Griffin, OPV2V, V2XSet, DAIR-V2X, nuScenes, ...). Dataset
specifics (file formats, folder layouts, coordinate conventions) live in
`src.datasets`, which normalises everything into a common sample model that
`src.pipeline.FaultPipeline` feeds through these injectors.

Single-agent (sensor-level) failure modes
-----------------------------------------
missing_modality      : sensor dropout (Bernoulli-gated zeroing / emptying)
temporal_misalignment : stale image pairing via index shifting
sensor_occlusion      : sensor-surface soiling / damage (dirt, scratch, crack)
calibration           : camera intrinsic / extrinsic drift -- the geometry a
                        camera BEV model projects through, not metadata
lidar_fog             : LiDAR fog (EXACT MultiCorrupt simulate_fog)
lidar_snow            : LiDAR snow (MultiCorrupt/Hahner physics verbatim)

Cooperative / V2X (agent-level) failure modes
---------------------------------------------
pose_error            : localisation error on shared agent poses
                        (the V2X-ViT / CoBEVT robustness setting)
communication         : transmission latency, agent dropout (packet loss),
                        and bandwidth-limited point sharing
                        (the Where2comm / V2VNet robustness settings)

Optional MultiCorrupt-backed camera/LiDAR corruptions
-----------------------------------------------------
brightness/darkness/fog/snow (camera) and lidar_points_reduce /
lidar_beams_reduce wrap verbatim MultiCorrupt backends (_mc_image.py /
_mc_lidar.py). Those backend files are not present in every checkout; when
absent, the rest of the package still imports and any attempt to use one of
those injectors raises an informative ImportError instead of breaking the
whole package at import time.
"""

from .calibration import CalibrationErrorInjector, rotation_from_axis_angle
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
from .pose_error import PoseErrorInjector
from .communication import (
    CommLatencyInjector,
    AgentDropInjector,
    BandwidthLimitInjector,
)

__all__ = [
    'CalibrationErrorInjector',
    'rotation_from_axis_angle',
    'MissingModalityInjector',
    'drop_image',
    'drop_points',
    'bernoulli_mask',
    'TemporalMisalignmentInjector',
    'sample_index_shift',
    'physical_displacement',
    'SensorOcclusionInjector',
    'OcclusionConfig',
    'PoseErrorInjector',
    'CommLatencyInjector',
    'AgentDropInjector',
    'BandwidthLimitInjector',
]

# ── Optional injectors (require files not shipped in every checkout) ───────
#
# lidar_fog / lidar_snow need the verbatim MultiCorrupt backends (_mc_lidar.py,
# fog lookup tables, snowflake particle files); brightness/darkness/fog/snow
# and the points/beams reducers need _mc_image.py / _mc_lidar.py. If a backend
# is missing we register a stub that raises a clear error on USE, not import.

_OPTIONAL = {
    'BrightnessInjector':           ('.brightness',          'BrightnessInjector'),
    'DarknessInjector':             ('.darkness',            'DarknessInjector'),
    'FogInjector':                  ('.fog',                 'FogInjector'),
    'SnowInjector':                 ('.snow',                'SnowInjector'),
    'LidarFogInjector':             ('.lidar_fog',           'LidarFogInjector'),
    'PointsReductionInjector':      ('.lidar_points_reduce', 'PointsReductionInjector'),
    'BeamReductionInjector':        ('.lidar_beams_reduce',  'BeamReductionInjector'),
    'BeamReductionInjectorGriffin': ('.lidar_beams_reduce',  'BeamReductionInjectorGriffin'),
    'LidarSnowInjector':            ('.lidar_snow',          'LidarSnowInjector'),
}

MISSING_OPTIONAL = {}


def _make_stub(name, module, error):
    class _MissingInjector:  # noqa: D401 - stub
        _reason = (
            f"{name} is unavailable: importing '{module}' failed "
            f"({error}). This injector wraps a verbatim MultiCorrupt backend "
            f"(_mc_image.py / _mc_lidar.py) that is not present in this "
            f"checkout. Add the backend module to src/fault_injectors/ to "
            f"enable it; all other injectors work without it."
        )

        def __init__(self, *args, **kwargs):
            raise ImportError(self._reason)

    _MissingInjector.__name__ = name
    return _MissingInjector


import importlib as _importlib

for _name, (_mod, _attr) in _OPTIONAL.items():
    try:
        _m = _importlib.import_module(_mod, package=__name__)
        globals()[_name] = getattr(_m, _attr)
    except Exception as _e:  # ImportError or backend init failure
        MISSING_OPTIONAL[_name] = str(_e)
        globals()[_name] = _make_stub(_name, _mod, _e)
    __all__.append(_name)

del _importlib, _name, _mod, _attr
