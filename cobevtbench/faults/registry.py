"""
registry.py
-----------
Build a fault bridge from a config dict, including the stages
``DataFaultBridge`` cannot reach on its own.

Why this exists
---------------
``DataFaultBridge`` exposes ``pipeline`` (the four injectors
``FaultPipeline.from_config`` knows), ``image_stages`` and ``lidar_stages``.
It does **not** expose ``sample_stages``, and both of CoBEVT's own injectors
are sample stages -- they need to see the whole scene to select an agent and
a camera by name.

So this module composes: it asks ``DataFaultBridge`` for everything it
already handles, then attaches the CoBEVT-specific stages to the pipeline it
built. Nothing here re-implements corruption; it is wiring only, and
``src.pipeline.FaultPipeline`` remains the single place faults are applied.

Config shape
------------
::

    name: camera_dropout
    pipeline:                 # handled by DataFaultBridge / FaultPipeline
      pose_error: {sigma_xy: 0.4, sigma_heading: 0.4}
    camera_dropout:           # cobevtbench sample stage
      {agents: ego, n_drop: 4}
    calibration:              # cobevtbench sample stage
      {sigma_focal_px: 8.0}
    image_faults:             # src.fault_injectors image stages
      - {kind: fog, severity: 2}
    lidar_faults:
      - {kind: points_reduction, severity: 2}
    agent_scope: non-ego
    seed: 2022
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from cpbench.faults import DataFaultBridge

from .calibration import CalibrationErrorInjector
from .camera_dropout import CameraDropoutInjector

logger = logging.getLogger(__name__)

# Keys this module consumes itself, before handing the rest to the bridge.
_OWN_KEYS = ("camera_dropout", "calibration", "image_faults", "lidar_faults")


def _build_image_stage(spec: Dict[str, Any], seed: int) -> Callable:
    """One image-level corruption from ``src.fault_injectors``.

    Imported lazily and per-kind: the MultiCorrupt-backed weather injectors
    are optional dependencies that raise a clear ImportError on construction
    when absent. Importing them all eagerly would make a clean run fail on a
    machine that only ever needed clean runs.
    """
    spec = dict(spec)
    kind = spec.pop("kind", None)
    if kind is None:
        raise ValueError(f"image fault entry needs a 'kind': {spec}")
    spec.setdefault("seed", seed)

    import src.fault_injectors as injectors

    builders = {
        "fog": "FogInjector", "snow": "SnowInjector",
        "brightness": "BrightnessInjector", "darkness": "DarknessInjector",
    }
    if kind == "occlusion":
        # OcclusionConfig also has a field called `kind` (dirt / scratch /
        # crack), which collides with this registry's discriminator. Ours is
        # spelled `pattern` in config and mapped here, so a config never has
        # to carry two differently-scoped keys of the same name.
        if "pattern" in spec:
            spec["kind"] = spec.pop("pattern")
        config = injectors.OcclusionConfig(**spec)
        return injectors.SensorOcclusionInjector(config)
    if kind in builders:
        return getattr(injectors, builders[kind])(**spec)
    raise ValueError(
        f"unknown image fault kind {kind!r}; expected 'occlusion' or one of "
        f"{sorted(builders)}")


def _build_lidar_stage(spec: Dict[str, Any], seed: int) -> Callable:
    """One LiDAR-level corruption from ``src.fault_injectors``."""
    spec = dict(spec)
    kind = spec.pop("kind", None)
    if kind is None:
        raise ValueError(f"lidar fault entry needs a 'kind': {spec}")
    spec.setdefault("seed", seed)

    import src.fault_injectors as injectors

    builders = {
        "fog": "LidarFogInjector", "snow": "LidarSnowInjector",
        "points_reduction": "PointsReductionInjector",
        "beam_reduction": "BeamReductionInjector",
    }
    if kind not in builders:
        raise ValueError(
            f"unknown lidar fault kind {kind!r}; expected one of "
            f"{sorted(builders)}")
    return getattr(injectors, builders[kind])(**spec)


def build_bridge(config: Optional[Dict[str, Any]], fps: float = 10.0,
                 seed: int = 0) -> DataFaultBridge:
    """Assemble a :class:`DataFaultBridge` from a fault config.

    Passing ``config=None`` or ``{}`` yields a provably clean bridge --
    ``bridge.is_clean`` is True and no injector exists to fire. Every
    benchmark's reference condition must go through that path, because a
    "clean" run that quietly injected something makes every comparison
    against it meaningless.

    Example
    -------
    >>> clean = build_bridge(None)
    >>> clean.is_clean
    True
    >>> faulty = build_bridge({"camera_dropout": {"agents": "ego", "n_drop": 2}})
    >>> faulty.is_clean
    False
    >>> len(faulty.pipeline.sample_stages)
    1
    """
    config = dict(config or {})
    config.pop("name", None)
    config.pop("sweep", None)
    seed = int(config.get("seed", seed))

    own = {key: config.pop(key, None) for key in _OWN_KEYS}

    sample_stages: List[Any] = []
    if own["camera_dropout"]:
        sample_stages.append(CameraDropoutInjector(
            seed=seed, **dict(own["camera_dropout"])))
    if own["calibration"]:
        sample_stages.append(CalibrationErrorInjector(
            seed=seed, **dict(own["calibration"])))

    image_stages = [_build_image_stage(spec, seed)
                    for spec in (own["image_faults"] or [])]
    lidar_stages = [_build_lidar_stage(spec, seed)
                    for spec in (own["lidar_faults"] or [])]

    bridge_config = dict(config)
    if image_stages:
        bridge_config["image_stages"] = image_stages
    if lidar_stages:
        bridge_config["lidar_stages"] = lidar_stages

    bridge = DataFaultBridge(bridge_config or None, fps=fps, seed=seed)

    if sample_stages:
        if bridge.pipeline is None:
            # Nothing else was configured, so DataFaultBridge produced an
            # identity bridge. Build the pipeline it skipped, or the sample
            # stages would be attached to nothing and the condition would
            # silently inject nothing at all.
            from src.pipeline import FaultPipeline
            bridge.pipeline = FaultPipeline.from_config(
                {}, fps=fps, seed=seed, agent_scope=bridge.agent_scope)
        bridge.pipeline.sample_stages = (list(bridge.pipeline.sample_stages)
                                         + sample_stages)
    logger.info("fault bridge: clean=%s, sample_stages=%d, image_stages=%d, "
                "lidar_stages=%d", bridge.is_clean, len(sample_stages),
                len(image_stages), len(lidar_stages))
    return bridge
