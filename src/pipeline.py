"""
pipeline.py
-----------
Compose fault injectors and apply them to any dataset's cooperative samples.

The pipeline is the only glue between `src.datasets` (which normalises every
dataset into `CooperativeSample`s) and `src.fault_injectors` (which corrupt
plain arrays and poses). It runs three kinds of stage, in this order:

1. SCHEDULE   communication latency: decides WHICH frame each non-ego
              agent's data is read from (must run first -- it re-loads data).
2. SAMPLE     agent-level faults on the assembled sample: agent dropout,
              pose error, bandwidth limits, missing modality -- anything with
              an `apply_to_sample(sample)` method or a sample->sample callable.
3. SENSOR     per-array faults: image->image and points->points callables
              (occlusion, fog, snow, brightness, ...) applied to the agents
              selected by `agent_scope`.

Usage
-----
    from src.datasets import load_dataset
    from src.fault_injectors import (PoseErrorInjector, AgentDropInjector,
                                     CommLatencyInjector)
    from src.pipeline import FaultPipeline

    ds = load_dataset('opv2v', scenario_dir='.../2021_08_18_09_02_56')

    pipe = FaultPipeline(
        latency=CommLatencyInjector(mu_delay=0.1, fps=ds.fps),
        sample_stages=[AgentDropInjector(p_drop=0.25),
                       PoseErrorInjector(sigma_xy=0.2, sigma_heading=0.2)],
        lidar_stages=[lambda pts: pts[::2]],       # any pts->pts callable
        agent_scope='non-ego',
    )
    corrupted = pipe(ds, k=0)

or declaratively (yaml-friendly, e.g. for severity sweeps):

    pipe = FaultPipeline.from_config({
        'latency':    {'mu_delay': 0.1, 'sigma_jitter': 0.02},
        'agent_drop': {'p_drop': 0.25},
        'pose_error': {'sigma_xy': 0.2, 'sigma_heading': 0.2},
        'bandwidth':  {'keep_fraction': 0.5},
    }, fps=10, seed=7)

Every injected fault is logged in `agent.faults` / `sample.meta`, so a run
is fully auditable and reproducible (all randomness is seeded).
"""

import numpy as np

from .fault_injectors.communication import (
    AgentDropInjector, BandwidthLimitInjector, CommLatencyInjector,
)
from .fault_injectors.pose_error import PoseErrorInjector


class FaultPipeline:
    """
    Parameters
    ----------
    latency       : CommLatencyInjector or None -- stale-frame scheduling.
    sample_stages : sequence of sample-level faults, each either an object
                    with `apply_to_sample(sample)` or a callable
                    sample -> sample.
    image_stages  : sequence of image -> image callables. Callables may
                    return either the image or an (image, info) tuple
                    (SensorOcclusionInjector does the latter); info dicts
                    are logged to `agent.faults`.
    lidar_stages  : sequence of (N, C) -> (M, C) callables.
    agent_scope   : which agents the SENSOR stages hit:
                    'all', 'non-ego', 'ego', or an explicit list of ids.
    """

    def __init__(self, latency=None, sample_stages=(), image_stages=(),
                 lidar_stages=(), agent_scope='all'):
        self.latency       = latency
        self.sample_stages = list(sample_stages)
        self.image_stages  = list(image_stages)
        self.lidar_stages  = list(lidar_stages)
        self.agent_scope   = agent_scope

    # ── main entry points ───────────────────────────────────────────────

    def apply(self, dataset, k, load=('lidar', 'images', 'labels')):
        """Load frame k from `dataset` and return the corrupted sample."""
        if self.latency is not None:
            sample = self.latency.apply(dataset, k, load=load)
        else:
            sample = dataset.get_sample(k, load=load)
        return self.apply_to_sample(sample)

    __call__ = apply

    def apply_to_sample(self, sample):
        """Corrupt an already-loaded CooperativeSample (in place, returned)."""
        for stage in self.sample_stages:
            fn = getattr(stage, 'apply_to_sample', stage)
            out = fn(sample)
            if out is not None:
                sample = out

        for agent_id in self._targets(sample):
            agent = sample.agents[agent_id]
            for cam, img in list(agent.images.items()):
                for stage in self.image_stages:
                    out = stage(img)
                    if isinstance(out, tuple):
                        out, info = out
                        agent.faults.setdefault('image', {}).setdefault(
                            cam, []).append(info)
                    img = out
                agent.images[cam] = img
            if agent.lidar is not None:
                pts = agent.lidar
                for stage in self.lidar_stages:
                    pts = stage(pts)
                agent.lidar = np.asarray(pts)
        return sample

    def _targets(self, sample):
        if self.agent_scope == 'all':
            return list(sample.agents)
        if self.agent_scope == 'non-ego':
            return [a for a, ag in sample.agents.items() if not ag.is_ego]
        if self.agent_scope == 'ego':
            return [a for a, ag in sample.agents.items() if ag.is_ego]
        return [a for a in self.agent_scope if a in sample.agents]

    # ── declarative construction ────────────────────────────────────────

    @classmethod
    def from_config(cls, config, fps=10.0, seed=0, agent_scope='non-ego'):
        """
        Build a pipeline from a plain dict (yaml/json-friendly).

        Recognised keys and their injectors (kwargs pass through):
            latency      -> CommLatencyInjector (fps filled in)
            agent_drop   -> AgentDropInjector
            pose_error   -> PoseErrorInjector
            bandwidth    -> BandwidthLimitInjector
        Order in the sample stage is fixed and sensible: drop, pose, bandwidth.
        Each injector gets an independent seed derived from `seed`.
        """
        config = dict(config)
        seeds = iter(np.random.SeedSequence(seed).generate_state(8).tolist())

        latency = None
        if 'latency' in config:
            kw = dict(config.pop('latency'))
            kw.setdefault('fps', fps)
            kw.setdefault('seed', next(seeds))
            latency = CommLatencyInjector(**kw)

        sample_stages = []
        for key, ctor in (('agent_drop', AgentDropInjector),
                          ('pose_error', PoseErrorInjector),
                          ('bandwidth', BandwidthLimitInjector)):
            if key in config:
                kw = dict(config.pop(key))
                kw.setdefault('seed', next(seeds))
                sample_stages.append(ctor(**kw))

        if config:
            raise ValueError(f'unknown fault config keys: {sorted(config)}; '
                             f'pass custom injectors via the constructor')
        return cls(latency=latency, sample_stages=sample_stages,
                   agent_scope=agent_scope)
