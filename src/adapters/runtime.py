"""
runtime.py
----------
Turn an OpenCOOD dataset class into a fault-injecting one.

``make_faulty_dataset(base_cls, spec)`` returns a subclass of ``base_cls`` with
two hooks. It takes the class **as an argument**, so this module -- like
:mod:`src.adapters.opencood` -- imports nothing from OpenCOOD and works
unchanged against ``opencood-official`` and ``v2xvit-official``.

Two hooks, not one
------------------
Every injector but one runs on the assembled ``CooperativeSample``. CommLatency
cannot: it decides *which file to read*, upstream of assembly. So:

* **hook 1** -- ``time_delay_calculation``: pre-assembly, frame selection.
  Overriding OpenCOOD's own seam keeps ``time_delay`` as a genuine model input
  (V2X-ViT's prior-encoding branch consumes it).
* **hook 2** -- ``retrieve_base_data``: post-assembly, everything else.

Unit note for hook 1: OpenCOOD's ``time_delay_calculation`` returns an integer
in units of **100 ms** (``time_delay = time_delay // 100``), then clamps against
``timestamp_index``. ``CommLatencyInjector.stale_index`` is **frame**-based. At
the 10 Hz of OPV2V/V2XSet, 1 frame == 1 unit, so the map is identity -- but the
conversion is named here so nobody later passes seconds.

Reproducibility under DataLoader workers
----------------------------------------
``opencood/tools/inference.py`` runs the loader with ``num_workers=16``, so
``retrieve_base_data`` executes in 16 forked processes. Any RNG state held on
the dataset object is therefore **per-worker**, and stateful injectors
(``AgentDropInjector``'s Gilbert-Elliott state, ``PoseErrorInjector``'s single
``default_rng``) would not reproduce across worker counts.

So every injector here is constructed **per call**, seeded from
``SeedSequence(base_seed, spawn_key=(idx, crc32(agent_id), stage))``. Nothing
accumulates. A run gives identical results at any ``num_workers`` including 0,
and every log row is re-derivable from ``(base_seed, idx, agent_id)``.

Known gap, stated rather than hidden: **bursty** (Gilbert-Elliott) AgentDrop is
inherently sequential and cannot be expressed this way. Only i.i.d. AgentDrop is
supported here; a bursty variant needs the chain advanced ``timestamp_index``
steps from a per-scenario seed, which is O(k) per sample. ``p_drop`` i.i.d. --
what the PoC and the standard protocol use -- has no such issue.
"""

import csv
import os
import zlib

import numpy as np

from ..fault_injectors.communication import AgentDropInjector
from ..fault_injectors.missing_modality import bernoulli_mask, drop_points
from ..fault_injectors.pose_error import PoseErrorInjector
from .opencood import OpenCOODAdapter

# Stage ordinals keep the sub-streams of one (idx, agent) independent.
_STAGE = {'pose_error': 0, 'agent_drop': 1, 'missing_lidar': 2,
          'points_reduce': 3, 'latency': 4, 'lidar_fog': 5, 'lidar_snow': 6}


def _seed(base_seed, idx, agent_id, stage):
    """A reproducible 32-bit seed for one (frame, agent, stage) draw.

    ``zlib.crc32`` rather than ``hash()``: Python's string hash is salted per
    process, so with forked DataLoader workers ``hash()`` would give a
    different seed in every worker.
    """
    key = (int(idx), int(zlib.crc32(str(agent_id).encode())), int(_STAGE[stage]))
    return int(np.random.SeedSequence(entropy=int(base_seed),
                                      spawn_key=key).generate_state(1)[0])


class FaultSpec:
    """
    Declarative description of one fault condition.

    Parameters
    ----------
    pose_error    : dict or None  kwargs for ``PoseErrorInjector`` (no ``seed``).
    agent_drop    : dict or None  ``{'p_drop': float}``, i.i.d. only.
    missing_lidar : dict or None  ``{'p_drop_lidar': float}``.
    points_reduce : dict or None  ``{'severity': 1|2|3}`` plus optional
                    ``'scope': 'all'|'non-ego'`` (default ``'all'``: sensor
                    degradation, like the weather stages).
    latency       : dict or None  kwargs for ``CommLatencyInjector``.
    lidar_fog     : dict or None  kwargs for ``LidarFogInjector`` (no ``seed``),
                    plus optional ``'scope': 'all'|'non-ego'``. Weather is
                    environmental, so the default scope is ``'all'`` -- fog
                    surrounds the ego too, unlike pose error, which is a
                    property of each *sender*.
    lidar_snow    : dict or None  kwargs for ``LidarSnowInjector`` (no
                    ``seed``), same optional ``'scope'``, default ``'all'``.
                    On OPV2V/V2XSet pass ``mount_height=1.9`` (CARLA LiDAR
                    mount; the injector default 1.10 is Griffin's) -- it seeds
                    the ground fit that calibrates the noise floor.
    seed          : int           base seed; every draw derives from it.
    log_dir       : str or None   where per-worker injection logs are written.
    """

    def __init__(self, pose_error=None, agent_drop=None, missing_lidar=None,
                 points_reduce=None, latency=None, lidar_fog=None,
                 lidar_snow=None, seed=0, log_dir=None):
        self.pose_error = pose_error
        self.agent_drop = agent_drop
        self.missing_lidar = missing_lidar
        self.points_reduce = points_reduce
        self.latency = latency
        self.lidar_fog = lidar_fog
        self.lidar_snow = lidar_snow
        self.seed = seed
        self.log_dir = log_dir

    @property
    def is_null(self):
        return not any((self.pose_error, self.agent_drop, self.missing_lidar,
                        self.points_reduce, self.latency, self.lidar_fog,
                        self.lidar_snow))

    def as_dict(self):
        return {'pose_error': self.pose_error, 'agent_drop': self.agent_drop,
                'missing_lidar': self.missing_lidar,
                'points_reduce': self.points_reduce, 'latency': self.latency,
                'lidar_fog': self.lidar_fog, 'lidar_snow': self.lidar_snow,
                'seed': self.seed}


class _InjectionLog:
    """One CSV per worker process; concatenate afterwards.

    16 workers appending to a single file would interleave rows mid-write.
    """

    FIELDS = ('idx', 'agent_id', 'is_ego', 'stage', 'seed', 'detail')

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self._fh = None
        self._writer = None
        self._pid = None

    def write(self, **row):
        if self.log_dir is None:
            return
        if self._fh is None or self._pid != os.getpid():
            self._pid = os.getpid()
            path = os.path.join(
                self.log_dir, 'injection_summary.%d.csv' % self._pid)
            new = not os.path.exists(path)
            self._fh = open(path, 'a')
            self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
            if new:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()


def make_faulty_dataset(base_cls, spec, adapter=None):
    """Return a subclass of ``base_cls`` that injects faults per ``spec``."""
    adapter = adapter or OpenCOODAdapter()

    class FaultyDataset(base_cls):

        fi_spec = spec
        fi_adapter = adapter

        def __init__(self, *args, **kwargs):
            super(FaultyDataset, self).__init__(*args, **kwargs)
            self._fi_log = _InjectionLog(spec.log_dir)
            self._fi_delays = None
            self._fi_checked_modality = False
            if spec.missing_lidar and spec.missing_lidar.get('p_drop_rgb'):
                raise ValueError(
                    'p_drop_rgb must be 0: OPV2V/V2XSet through OpenCOOD are '
                    'LiDAR-only, retrieve_base_data never reads the PNGs')
            if spec.lidar_snow:
                # Generate the per-severity snowflake particle files ONCE,
                # here in the parent process. Left to first use they would be
                # generated inside DataLoader workers -- 16 forked processes
                # racing to write the same 64 .npy files.
                from ..fault_injectors.snowflake_sampling import \
                    ensure_particle_files
                ensure_particle_files(spec.lidar_snow.get('severity', 2),
                                      verbose=False)

        # ── hook 1: pre-assembly frame selection ────────────────────────

        def time_delay_calculation(self, ego_flag):
            """
            UNEXERCISED in the PoseError PoC (``spec.latency`` is None, so this
            delegates straight to OpenCOOD). Wired but not yet validated; it
            gets its own PoC when CommLatency lands.

            ``time_delay_calculation(self, ego_flag)`` carries neither ``idx``
            nor ``cav_id``, so the delay map is built up front in
            ``retrieve_base_data`` (which we override anyway) and consumed here
            in the same ``scenario_database.items()`` order the caller loops in.
            No mutable per-call state, no ordering guesswork.
            """
            if spec.latency is None or self._fi_delays is None:
                return super(FaultyDataset, self).time_delay_calculation(ego_flag)
            if ego_flag:
                return 0
            if not self._fi_delays:
                raise AssertionError('delay map exhausted: the cav loop ran '
                                     'more times than the map was built for')
            return self._fi_delays.pop(0)

        def _fi_build_delay_map(self, idx):
            from ..fault_injectors.communication import CommLatencyInjector
            scenario_index = 0
            for i, ele in enumerate(self.len_record):
                if idx < ele:
                    scenario_index = i
                    break
            db = self.scenario_database[scenario_index]
            ts_index = idx if scenario_index == 0 \
                else idx - self.len_record[scenario_index - 1]

            delays = []
            for cav_id, content in db.items():
                if content['ego']:
                    delays.append(0)
                    continue
                inj = CommLatencyInjector(
                    seed=_seed(spec.seed, idx, cav_id, 'latency'),
                    **spec.latency)
                # frames -> OpenCOOD's 100 ms units; identity at 10 Hz.
                shift = ts_index - inj.stale_index(str(cav_id), ts_index)
                delays.append(int(max(0, shift)))
            self._fi_delays = delays

        # ── hook 2: post-assembly sample faults ─────────────────────────

        def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
            if spec.latency is not None:
                self._fi_build_delay_map(idx)
            base = super(FaultyDataset, self).retrieve_base_data(
                idx, cur_ego_pose_flag=cur_ego_pose_flag)
            if spec.is_null:
                # Still round-trip: the null path must exercise exactly the
                # same adapter code the faulty path does, or Gate 2 proves
                # nothing about the faulty run.
                sample = self.fi_adapter.to_canonical(base, idx)
                return self.fi_adapter.from_canonical(sample, base)

            sample = self.fi_adapter.to_canonical(base, idx)

            if not self._fi_checked_modality:
                self.fi_adapter.assert_modality(sample, 'lidar')
                self._fi_checked_modality = True

            ego_pose_before = sample.ego.pose.copy()
            self._fi_apply(sample, idx)

            if not np.array_equal(sample.ego.pose, ego_pose_before):
                raise AssertionError(
                    'ego pose was perturbed. GT is generated with the ego '
                    'pose as reference_lidar_pose, so this would move the '
                    'ground truth and the AP delta would not be attributable '
                    'to cooperative misalignment.')
            return self.fi_adapter.from_canonical(sample, base)

        # ── the sample-stage faults, all stateless per (idx, agent) ─────

        def _fi_apply(self, sample, idx):
            log = self._fi_log

            if spec.agent_drop:
                for aid in [a for a in list(sample.agents)
                            if not sample.agents[a].is_ego]:
                    s = _seed(spec.seed, idx, aid, 'agent_drop')
                    inj = AgentDropInjector(seed=s, **spec.agent_drop)
                    if inj._lost(aid):
                        del sample.agents[aid]
                        sample.meta.setdefault('dropped_agents', []).append(aid)
                        log.write(idx=idx, agent_id=aid, is_ego=0,
                                  stage='agent_drop', seed=s, detail='dropped')

            if spec.pose_error:
                for aid, agent in sample.agents.items():
                    if agent.is_ego:
                        continue
                    s = _seed(spec.seed, idx, aid, 'pose_error')
                    inj = PoseErrorInjector(seed=s, **spec.pose_error)
                    err = inj.sample_error()
                    agent.pose = inj.perturb_matrix(agent.pose, error=err)
                    agent.faults['pose_error'] = err
                    log.write(idx=idx, agent_id=aid, is_ego=0,
                              stage='pose_error', seed=s,
                              detail=_fmt(err))

            if spec.missing_lidar:
                p = spec.missing_lidar['p_drop_lidar']
                for aid, agent in sample.agents.items():
                    if agent.is_ego:
                        continue
                    s = _seed(spec.seed, idx, aid, 'missing_lidar')
                    keep = bernoulli_mask(p, np.random.default_rng(s))
                    agent.faults['missing_lidar'] = {'m_lidar': int(keep)}
                    if not keep:
                        agent.lidar = drop_points(agent.lidar)
                        log.write(idx=idx, agent_id=aid, is_ego=0,
                                  stage='missing_lidar', seed=s,
                                  detail='lidar_dropped')

            for stage_name in ('lidar_fog', 'lidar_snow'):
                cfg = getattr(spec, stage_name)
                if not cfg:
                    continue
                # Weather is environmental: default scope is EVERY agent,
                # ego included (contrast pose error, a per-sender fault).
                cfg = dict(cfg)
                scope = cfg.pop('scope', 'all')
                if stage_name == 'lidar_fog':
                    from ..fault_injectors.lidar_fog import LidarFogInjector \
                        as _Weather
                else:
                    from ..fault_injectors.lidar_snow import LidarSnowInjector \
                        as _Weather
                    cfg.setdefault('verbose', False)
                for aid, agent in sample.agents.items():
                    if scope == 'non-ego' and agent.is_ego:
                        continue
                    if agent.lidar is None or not len(agent.lidar):
                        continue          # e.g. already dropped by missing_lidar
                    s = _seed(spec.seed, idx, aid, stage_name)
                    before = agent.lidar
                    after = _Weather(seed=s, **cfg)(before)
                    agent.lidar = after
                    detail = ('pts=%d->%d;meanI=%.4f->%.4f'
                              % (len(before), len(after),
                                 float(np.asarray(before)[:, 3].mean()),
                                 float(np.asarray(after)[:, 3].mean())))
                    agent.faults[stage_name] = detail
                    log.write(idx=idx, agent_id=aid,
                              is_ego=int(agent.is_ego), stage=stage_name,
                              seed=s, detail=detail)

            if spec.points_reduce:
                # RNG defect fixed 2026-08-05: the injector now contains the
                # global numpy stream (save/seed/restore) and per-sample
                # independence comes from per-(idx, agent) construction here.
                from ..fault_injectors.lidar_points_reduce import \
                    PointsReductionInjector
                cfg = dict(spec.points_reduce)
                scope = cfg.pop('scope', 'all')
                for aid, agent in sample.agents.items():
                    if scope == 'non-ego' and agent.is_ego:
                        continue
                    if agent.lidar is None or not len(agent.lidar):
                        continue
                    s = _seed(spec.seed, idx, aid, 'points_reduce')
                    before = agent.lidar
                    after = np.ascontiguousarray(
                        PointsReductionInjector(seed=s, **cfg)(before),
                        dtype=np.float32)
                    agent.lidar = after
                    detail = 'pts=%d->%d' % (len(before), len(after))
                    agent.faults['points_reduce'] = detail
                    log.write(idx=idx, agent_id=aid, is_ego=int(agent.is_ego),
                              stage='points_reduce', seed=s, detail=detail)

    FaultyDataset.__name__ = 'Faulty' + base_cls.__name__
    return FaultyDataset


def _fmt(err):
    return ';'.join('%s=%.6f' % (k, v) for k, v in sorted(err.items()))
