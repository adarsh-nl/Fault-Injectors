"""
registry.py
-----------
Build both fault planes from one config block.

Nothing here implements corruption. ``src.pipeline.FaultPipeline`` remains
the single place a physical fault is applied, and ``MetadataFaultBridge``
the single place batch metadata is; this module is wiring, and its job is to
make a config either produce exactly the injectors it names or fail loudly.

Config shape
------------
::

    name: v2x_noise
    pipeline:                 # plane 1, via cpbench.faults.DataFaultBridge
      pose_error: {sigma_xy: 0.4, sigma_heading: 0.4}
      latency:    {mu_delay: 0.2}
      agent_drop: {p_drop: 0.25}
    lidar_faults:             # plane 1, src.fault_injectors sensor stages
      - {kind: fog, severity: 2}
    metadata_pipeline:        # plane 2, this package's own
      delay_encoding: {mode: zero}
      type_flip:      {p_flip: 0.5}
    agent_scope: non-ego
    seed: 2022

LiDAR-only: V2X-ViT has no camera track, so the image/calibration paths the
w2cbench registry carries have no meaning here and a config naming them
fails by name rather than being ignored.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from cpbench.faults import DataFaultBridge

from v2xvitbench.faults.metadata import MetadataFaultBridge

logger = logging.getLogger(__name__)

# Keys this module consumes before handing the rest to DataFaultBridge.
_OWN_KEYS = ("lidar_faults", "metadata_pipeline", "metadata_sweep")
_UNSUPPORTED = ("image_faults", "calibration")


def _build_lidar_stage(spec: Dict[str, Any], seed: int) -> Callable:
    """One LiDAR-level corruption from ``src.fault_injectors``.

    Lazily imported: the weather injectors are optional dependencies that
    raise a clear ImportError on construction when absent, and importing
    them eagerly would make a *clean* run fail on a machine that only ever
    needed clean runs.
    """
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
    for key in ("name", "sweep"):
        config.pop(key, None)
    seed = int(config.get("seed", seed))

    for key in _UNSUPPORTED:
        if key in config:
            raise ValueError(
                f"fault config names {key!r}, but V2X-ViT is LiDAR-only; "
                "remove the block or benchmark a camera model instead")

    own = {key: config.pop(key, None) for key in _OWN_KEYS}
    lidar_stages = [_build_lidar_stage(spec, seed)
                    for spec in (own["lidar_faults"] or [])]

    bridge_config = dict(config)
    if lidar_stages:
        bridge_config["lidar_stages"] = lidar_stages

    bridge = DataFaultBridge(bridge_config or None, fps=fps, seed=seed)
    logger.info("physical bridge: clean=%s, lidar_stages=%d",
                bridge.is_clean, len(lidar_stages))
    return bridge


def build_metadata_bridge(config: Optional[Dict[str, Any]],
                          seed: int = 0) -> MetadataFaultBridge:
    """Assemble the metadata-plane bridge from the same fault config.

    Reads only ``metadata_pipeline``; a config with none produces a bridge
    with no injectors, matching the physical plane's guarantee.

    Example
    -------
    >>> build_metadata_bridge({"pipeline": {"pose_error": {}}}).is_clean
    True
    >>> build_metadata_bridge(
    ...     {"metadata_pipeline": {"type_flip": {"p_flip": 0.5}}}).is_clean
    False
    """
    config = dict(config or {})
    seed = int(config.get("seed", seed))
    return MetadataFaultBridge.from_config(config.get("metadata_pipeline"),
                                           seed=seed)


def build_bridges(config: Optional[Dict[str, Any]], fps: float = 10.0,
                  seed: int = 0
                  ) -> Tuple[DataFaultBridge, MetadataFaultBridge]:
    """Both planes from one config, as the benchmark runner wants them.

    >>> physical, metadata = build_bridges(None)
    >>> physical.is_clean and metadata.is_clean
    True
    """
    return (build_bridge(config, fps=fps, seed=seed),
            build_metadata_bridge(config, seed=seed))
