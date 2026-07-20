"""
bridge.py
---------
The single corruption path of corabench.

Purpose
    Apply *physical* faults -- pose error, communication latency, agent
    dropout, bandwidth limits, sensor corruptions -- to raw cooperative
    samples exactly where they occur in the real world: on poses, LiDAR,
    images and the comm link, BEFORE the model forward. This wraps the
    existing `src.pipeline.FaultPipeline`; corabench adds configuration,
    bookkeeping and structured fault records, never new corruption sites.

Inputs
    A YAML-friendly config dict, e.g.::

        pipeline:
          latency:    {mu_delay: 0.2, sigma_jitter: 0.02}
          agent_drop: {p_drop: 0.25}
          pose_error: {sigma_xy: 0.4, sigma_heading: 0.4}
          bandwidth:  {keep_fraction: 0.5}
        agent_scope: non-ego
        seed: 2026

Outputs
    Corrupted `CooperativeSample`s plus a list of `FaultRecord`s harvested
    from the audit trail `FaultPipeline` leaves in `agent.faults` /
    `sample.meta`.

Example
    >>> bridge = DataFaultBridge({"pipeline": {"pose_error":
    ...     {"sigma_xy": 0.4, "sigma_heading": 0.4}}}, fps=10.0, seed=7)
    >>> sample = bridge.load(dataset, k=0)            # doctest: +SKIP
    >>> bridge.records[-1].fault_type                 # doctest: +SKIP
    'pose_error'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.pipeline import FaultPipeline

logger = logging.getLogger(__name__)


@dataclass
class FaultRecord:
    """One physically injected fault, for injection_summary.csv.

    Attributes
    ----------
    frame       ego frame index the fault was applied at.
    agent_id    affected agent ('*' for sample-wide faults such as latency).
    fault_type  e.g. 'pose_error', 'agent_drop', 'latency', 'bandwidth',
                'image', 'lidar'.
    target      what was corrupted: 'pose' | 'agent' | 'schedule' |
                'lidar' | 'image' | 'sample'.
    params      injector parameters / per-fault details as logged upstream.
    """

    frame: int
    agent_id: str
    fault_type: str
    target: str
    params: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        row = {"frame": self.frame, "agent_id": self.agent_id,
               "fault_type": self.fault_type, "target": self.target}
        for key, val in self.params.items():
            row[f"param_{key}"] = val if isinstance(val, (str, int, float, bool)) \
                else repr(val)
        return row


class DataFaultBridge:
    """Config-driven wrapper around `src.pipeline.FaultPipeline`.

    Parameters
    ----------
    config : dict with keys
        pipeline     -- `FaultPipeline.from_config` block (latency,
                        agent_drop, pose_error, bandwidth), or None/{} for a
                        clean bridge (identity).
        image_stages / lidar_stages -- optional lists of extra sensor-level
                        callables (constructed by the caller; e.g. fog, snow,
                        occlusion injectors from `src.fault_injectors`).
        agent_scope  -- 'all' | 'non-ego' | 'ego' | list of ids
                        (default 'non-ego': faults live on the V2X link).
    fps    : dataset frame rate (needed by the latency injector).
    seed   : master seed; forwarded so every injector is deterministic.

    Notes
    -----
    * ``load(dataset, k)`` must be used when latency is configured -- stale-
      frame scheduling has to re-read collaborator data from earlier frames.
    * ``apply_to_sample(sample)`` corrupts an already-loaded sample (no
      latency stage possible).
    * The bridge is stateless across frames except for `records`, which
      accumulates the audit trail; call ``drain_records()`` per frame/epoch.
    """

    def __init__(self, config: Optional[Dict[str, Any]], fps: float = 10.0,
                 seed: int = 0) -> None:
        config = dict(config or {})
        self.agent_scope = config.pop("agent_scope", "non-ego")
        self.seed = int(config.pop("seed", seed))
        self.name = config.pop("name", "faults")
        pipeline_cfg = config.pop("pipeline", None) or {}
        image_stages: Sequence = config.pop("image_stages", ())
        lidar_stages: Sequence = config.pop("lidar_stages", ())
        config.pop("sweep", None)  # sweep grids are expanded by evaluation/
        if config:
            raise ValueError(f"unknown fault-bridge config keys: {sorted(config)}")

        self.pipeline_cfg = dict(pipeline_cfg)
        if pipeline_cfg or image_stages or lidar_stages:
            self.pipeline: Optional[FaultPipeline] = FaultPipeline.from_config(
                dict(pipeline_cfg), fps=fps, seed=self.seed,
                agent_scope=self.agent_scope)
            self.pipeline.image_stages = list(image_stages)
            self.pipeline.lidar_stages = list(lidar_stages)
        else:
            self.pipeline = None  # clean bridge: identity
        self.records: List[FaultRecord] = []
        logger.info("DataFaultBridge(name=%s, scope=%s, seed=%d, pipeline=%s)",
                    self.name, self.agent_scope, self.seed,
                    sorted(self.pipeline_cfg) or "clean")

    @property
    def is_clean(self) -> bool:
        return self.pipeline is None

    # -- corruption entry points (delegate to src.pipeline) -----------------

    def load(self, dataset, k: int,
             load: Tuple[str, ...] = ("lidar", "labels")):
        """Load frame `k` from `dataset` and corrupt it (incl. latency)."""
        if self.pipeline is None:
            sample = dataset.get_sample(k, load=load)
        else:
            sample = self.pipeline.apply(dataset, k, load=load)
        self._harvest(sample)
        return sample

    def apply_to_sample(self, sample):
        """Corrupt an already-loaded CooperativeSample (no latency stage)."""
        if self.pipeline is not None:
            sample = self.pipeline.apply_to_sample(sample)
        self._harvest(sample)
        return sample

    # -- audit trail --------------------------------------------------------

    def drain_records(self) -> List[FaultRecord]:
        """Return accumulated FaultRecords and reset the buffer."""
        out, self.records = self.records, []
        return out

    def _harvest(self, sample) -> None:
        """Convert `agent.faults` / `sample.meta` audit info to FaultRecords."""
        frame = getattr(sample, "frame_index", -1)
        meta = getattr(sample, "meta", {}) or {}
        for key, val in meta.items():
            if key in ("fps",):
                continue
            self.records.append(FaultRecord(
                frame=frame, agent_id="*", fault_type=str(key),
                target="sample",
                params=val if isinstance(val, dict) else {"value": val}))
        for aid, agent in getattr(sample, "agents", {}).items():
            for fault_type, info in (getattr(agent, "faults", {}) or {}).items():
                target = {"pose_error": "pose", "comm_latency": "schedule",
                          "bandwidth": "lidar", "image": "image",
                          "lidar": "lidar"}.get(fault_type, "agent")
                self.records.append(FaultRecord(
                    frame=frame, agent_id=str(aid), fault_type=str(fault_type),
                    target=target,
                    params=info if isinstance(info, dict) else {"value": info}))

    def __repr__(self) -> str:
        return (f"DataFaultBridge(name={self.name!r}, "
                f"pipeline={sorted(self.pipeline_cfg) or 'clean'}, "
                f"scope={self.agent_scope!r}, seed={self.seed})")
