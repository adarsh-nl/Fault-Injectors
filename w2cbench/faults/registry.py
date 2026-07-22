"""
registry.py
-----------
Build both fault planes from one config block.

Nothing here implements corruption. ``src.pipeline.FaultPipeline`` remains the
single place a physical fault is applied, and ``ProtocolFaultBridge`` the single
place a message is; this module is wiring, and its job is to make a config
either produce exactly the injectors it names or fail loudly.

Config shape
------------
::

    name: comm_stress
    pipeline:                 # plane 1, via cpbench.faults.DataFaultBridge
      pose_error: {sigma_xy: 0.4, sigma_heading: 0.4}
      agent_drop: {p_drop: 0.25}
      latency:    {mu_delay: 0.2}
    lidar_faults:             # plane 1, src.fault_injectors sensor stages
      - {kind: fog, severity: 2}
    protocol_pipeline:        # plane 2, this package's own
      request_loss:      {p_loss: 0.25}
      confidence_report: {mode: inflate, magnitude: 0.3}
    agent_scope: non-ego
    seed: 2022

Why lazily imported
-------------------
The weather injectors are optional dependencies that raise a clear ImportError
on construction when absent. Importing them eagerly would make a *clean* run
fail on a machine that only ever needed clean runs -- which on a shared cluster
is most of them.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from cpbench.faults import DataFaultBridge

from .protocol import ProtocolFaultBridge

logger = logging.getLogger(__name__)

# Keys this module consumes before handing the rest to DataFaultBridge.
_OWN_KEYS = ("image_faults", "lidar_faults", "calibration",
             "protocol_pipeline", "protocol_sweep")


def _build_image_stage(spec: Dict[str, Any], seed: int) -> Callable:
    """One image-level corruption from ``src.fault_injectors``."""
    spec = dict(spec)
    kind = spec.pop("kind", None)
    if kind is None:
        raise ValueError(f"image fault entry needs a 'kind': {spec}")
    spec.setdefault("seed", seed)

    import src.fault_injectors as injectors

    builders = {"fog": "FogInjector", "snow": "SnowInjector",
                "brightness": "BrightnessInjector",
                "darkness": "DarknessInjector"}
    if kind == "occlusion":
        # OcclusionConfig also has a field called `kind` (dirt / scratch /
        # crack), which collides with this registry's discriminator. Ours is
        # spelled `pattern` in config and mapped here, so a config never has
        # to carry two differently-scoped keys of the same name.
        if "pattern" in spec:
            spec["kind"] = spec.pop("pattern")
        return injectors.SensorOcclusionInjector(
            injectors.OcclusionConfig(**spec))
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

    builders = {"fog": "LidarFogInjector", "snow": "LidarSnowInjector",
                "points_reduction": "PointsReductionInjector",
                "beam_reduction": "BeamReductionInjector"}
    if kind not in builders:
        raise ValueError(
            f"unknown lidar fault kind {kind!r}; expected one of "
            f"{sorted(builders)}")
    return getattr(injectors, builders[kind])(**spec)


def build_bridge(config: Optional[Dict[str, Any]], fps: float = 10.0,
                 seed: int = 0) -> DataFaultBridge:
    """Assemble the physical-plane bridge from a fault config.

    ``config=None`` or ``{}`` yields a provably clean bridge --
    ``bridge.is_clean`` is True and no injector exists to fire. Every
    benchmark's reference condition takes that path, because a "clean" run
    that quietly injected something makes every comparison against it
    meaningless.

    Example
    -------
    >>> build_bridge(None).is_clean
    True
    >>> build_bridge({"pipeline": {"pose_error": {"sigma_xy": 0.4}}}).is_clean
    False
    """
    config = dict(config or {})
    for key in ("name", "sweep", "bandwidth_sweep"):
        config.pop(key, None)
    seed = int(config.get("seed", seed))

    own = {key: config.pop(key, None) for key in _OWN_KEYS}
    image_stages = [_build_image_stage(spec, seed)
                    for spec in (own["image_faults"] or [])]
    lidar_stages = [_build_lidar_stage(spec, seed)
                    for spec in (own["lidar_faults"] or [])]

    # Calibration is a SAMPLE stage: it needs the whole scene to pick agents
    # and cameras by name, which DataFaultBridge's image/lidar stage lists
    # cannot express. So the bridge is built first and the stage attached to
    # the pipeline it produced.
    sample_stages: List[Any] = []
    if own["calibration"]:
        from src.fault_injectors import CalibrationErrorInjector
        sample_stages.append(CalibrationErrorInjector(
            seed=seed, **dict(own["calibration"])))

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
            # stages would attach to nothing and the condition would silently
            # inject nothing at all.
            from src.pipeline import FaultPipeline
            bridge.pipeline = FaultPipeline.from_config(
                {}, fps=fps, seed=seed, agent_scope=bridge.agent_scope)
        bridge.pipeline.sample_stages = (list(bridge.pipeline.sample_stages)
                                         + sample_stages)

    logger.info("physical bridge: clean=%s, sample_stages=%d, image_stages=%d, "
                "lidar_stages=%d", bridge.is_clean, len(sample_stages),
                len(image_stages), len(lidar_stages))
    return bridge


def build_protocol_bridge(config: Optional[Dict[str, Any]],
                          seed: int = 0) -> ProtocolFaultBridge:
    """Assemble the protocol-plane bridge from the same fault config.

    Reads only ``protocol_pipeline``; a config with none produces a bridge
    with no injectors, matching the physical plane's guarantee.

    Example
    -------
    >>> build_protocol_bridge({"pipeline": {"pose_error": {}}}).is_clean
    True
    >>> build_protocol_bridge(
    ...     {"protocol_pipeline": {"request_loss": {"p_loss": 0.5}}}).is_clean
    False
    """
    config = dict(config or {})
    seed = int(config.get("seed", seed))
    return ProtocolFaultBridge.from_config(config.get("protocol_pipeline"),
                                           seed=seed)


def build_bridges(config: Optional[Dict[str, Any]], fps: float = 10.0,
                  seed: int = 0):
    """Both planes from one config, as the benchmark runner wants them.

    >>> physical, protocol = build_bridges(None)
    >>> physical.is_clean and protocol.is_clean
    True
    """
    return (build_bridge(config, fps=fps, seed=seed),
            build_protocol_bridge(config, seed=seed))
