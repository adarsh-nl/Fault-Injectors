"""
griffin.py
----------
Fault routing for Griffin (aerial-ground cooperative perception): per-agent,
by modality, on the canonical ``CooperativeSample`` form.

Griffin needs no dict-schema translation the way OpenCOOD did --
``GriffinDataset`` (`src/datasets/griffin.py`) already emits canonical
samples, and the ``griffin-release`` layout bakes no derived transforms, so a
null pass is a pure pass-through (the Phase 2 round-trip gate asserts it
bit-identical). What Griffin *does* need, and OpenCOOD did not, is per-agent
modality routing: every scene is one LiDAR+camera vehicle (the ego)
cooperating with one camera-only drone, so a fault's target set depends on
which agents carry the modality it corrupts.

Routing (approved spec, 2026-08-05):

    LiDAR faults (fog / snow / points-reduce)  -> LiDAR agents (the vehicle)
    camera faults (brightness / darkness /
                   fog / snow on images)       -> camera agents (drone + vehicle)
    MissingModality (Bernoulli, cameras)       -> the drone
    CommLatency (scene-clamped)                -> the drone (ego always current)
    AgentDrop                                  -> the drone (user-confirmed:
                                                  the vehicle is the ego of
                                                  every scene, so the drone is
                                                  the only droppable agent)
    PoseError                                  -> NOT WIRED, by spec

Explicit skip, never silent no-op
---------------------------------
The loophole this module exists to close: a LiDAR fault reaching the
LiDAR-less drone must not quietly do nothing -- and must not "run" on an
empty array and be counted as an injection. Every stage writes one routing
row per frame (``targets=[...];skipped=[drone:no-lidar]``), an empty target
set is logged as ``no_target_agents``, and a ``(0, C)`` cloud is logged as an
``empty-cloud`` skip instead of being fed to fog/snow.

Scene-boundary clamp for CommLatency
------------------------------------
``GriffinDataset`` indexes a subset side as one flat sequence (~50 scenes,
~140 frames each). An unclamped stale index near a scene start would pair the
vehicle with a *different scene's* drone frame. ``scene_infos.json`` carries
each scene's ``frames`` list; this module maps them to flat indices and
passes the scene's first index as ``k_min`` to
``CommLatencyInjector.stale_index`` -- the same role OpenCOOD's
``timestamp_index`` clamp plays there.

RNG hygiene
-----------
Injectors are constructed per ``(frame, agent[, camera], stage)``, seeded via
the same ``SeedSequence(entropy, spawn_key=(idx, crc32(agent), stage))``
derivation the OpenCOOD runtime uses (Griffin has its own stage table, so the
derivation is restated here with the identical shape rather than imported --
a Griffin log row is re-derivable exactly like an OpenCOOD one). The
MultiCorrupt image injectors reseed the *global* numpy RNG on every call
(upstream convention); ``_contained`` brackets every stage call with a
global-RNG save/restore, so routing composes cleanly no matter what a
verbatim backend does inside.

This module is imported by nothing on the OpenCOOD path; the OpenCOOD
baselines cannot reach it.
"""

import json
import os
import zlib
from contextlib import contextmanager

import numpy as np

from ..fault_injectors.communication import CommLatencyInjector
from ..fault_injectors.missing_modality import bernoulli_mask, drop_image
from .modality import agents_supporting
# Same log schema as the OpenCOOD runtime -- one implementation, imported.
from .runtime import _InjectionLog

_STAGE = {'agent_drop': 0, 'latency': 1, 'missing_modality': 2,
          'lidar_fog': 3, 'lidar_snow': 4, 'points_reduce': 5, 'camera': 6}

_CAMERA_KINDS = ('brightness', 'darkness', 'fog', 'snow')


def _seed(base, idx, agent_key, stage):
    """(base, frame, agent[, camera], stage) -> independent 32-bit seed.

    Identical derivation shape to ``runtime._seed`` (crc32, not ``hash()``,
    which is salted per process). ``agent_key`` may be ``'drone'`` or
    ``'drone/front'`` -- folding the camera name in gives per-camera streams.
    """
    key = (int(idx), int(zlib.crc32(str(agent_key).encode())),
           int(_STAGE[stage]))
    return int(np.random.SeedSequence(entropy=int(base),
                                      spawn_key=key).generate_state(1)[0])


@contextmanager
def _contained():
    """Bracket a stage call: the global numpy RNG is restored afterwards."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def _camera_injector(kind):
    if kind == 'brightness':
        from ..fault_injectors.brightness import BrightnessInjector as C
    elif kind == 'darkness':
        from ..fault_injectors.darkness import DarknessInjector as C
    elif kind == 'fog':
        from ..fault_injectors.fog import FogInjector as C
    else:
        from ..fault_injectors.snow import SnowInjector as C
    return C


class GriffinFaultSpec:
    """
    Declarative Griffin fault condition. All fields default off.

    agent_drop       : {'p_drop': float}          i.i.d., drone only.
    latency          : {'mu_delay': s, 'sigma_jitter': s}  drone only,
                       scene-clamped.
    missing_modality : {'p_drop_rgb': float}      drone cameras, Bernoulli.
    lidar_fog        : {'severity': 1|2|3}        LiDAR agents (vehicle).
    lidar_snow       : {'severity': 1|2|3, ...}   LiDAR agents (vehicle).
    points_reduce    : {'severity': 1|2|3}        LiDAR agents (vehicle);
                       RNG defect fixed 2026-08-05, sweep-ready.
    camera           : {'kind': 'brightness'|'darkness'|'fog'|'snow',
                        'severity': 1|2|3}        camera agents (drone+vehicle).
    seed             : base seed; every draw derives from
                       (seed, frame, agent[, camera], stage).
    log_dir          : per-worker routing/injection CSVs, or None.
    """

    def __init__(self, agent_drop=None, latency=None, missing_modality=None,
                 lidar_fog=None, lidar_snow=None, points_reduce=None,
                 camera=None, seed=0, log_dir=None):
        if camera and camera.get('kind') not in _CAMERA_KINDS:
            raise ValueError('camera.kind must be one of %s' % (_CAMERA_KINDS,))
        if missing_modality and missing_modality.get('p_drop_lidar'):
            raise ValueError(
                'p_drop_lidar must be 0 on Griffin: MissingModality routes to '
                'the camera-only drone; LiDAR loss is not expressible there')
        self.agent_drop = agent_drop
        self.latency = latency
        self.missing_modality = missing_modality
        self.lidar_fog = lidar_fog
        self.lidar_snow = lidar_snow
        self.points_reduce = points_reduce
        self.camera = camera
        self.seed = seed
        self.log_dir = log_dir

    @property
    def is_null(self):
        return not any((self.agent_drop, self.latency, self.missing_modality,
                        self.lidar_fog, self.lidar_snow, self.points_reduce,
                        self.camera))


class FaultedGriffinDataset:
    """
    Wrap a ``GriffinDataset``; ``get_sample(k)`` returns the corrupted
    canonical sample. The wrapped dataset is never mutated. With a null spec
    the sample is byte-for-byte what the wrapped dataset returns -- there is
    no translation layer to be inexact in.
    """

    def __init__(self, dataset, spec):
        self.ds = dataset
        self.spec = spec
        self._log = _InjectionLog(spec.log_dir)
        self._scene_start = self._build_scene_starts()
        self._lidar_mounts = self._load_lidar_mounts()
        if spec.lidar_snow:
            # One-time particle generation, here in the parent process (a
            # future DataLoader's forked workers must not race to write the
            # same 64 .npy files).
            from ..fault_injectors.snowflake_sampling import \
                ensure_particle_files
            ensure_particle_files(spec.lidar_snow.get('severity', 2),
                                  verbose=False)

    def __len__(self):
        return len(self.ds)

    # ── assembly (hook 1: CommLatency, scene-clamped) ───────────────────

    def get_sample(self, k, load=('lidar', 'images', 'labels')):
        spec = self.spec
        if spec.latency:
            sample = self._assemble_with_latency(k, load)
        else:
            sample = self.ds.get_sample(k, load=load)
        if spec.is_null:
            return sample
        self._apply(sample, k)
        if sample.ego_id not in sample.agents:
            raise AssertionError('the ego (vehicle) was removed -- forbidden')
        return sample

    def _assemble_with_latency(self, k, load):
        """Ego at k; each non-ego agent at its own scene-clamped stale frame."""
        spec = self.spec
        sample = self.ds.get_sample(k, load=load)
        k_min = self._scene_start[k]
        for aid in [a for a, ag in sample.agents.items() if not ag.is_ego]:
            inj = CommLatencyInjector(seed=_seed(spec.seed, k, aid, 'latency'),
                                      fps=self.ds.fps, **spec.latency)
            k_stale, delta = inj.stale_index(aid, k, k_min=k_min)
            assert k_stale >= k_min, 'stale index escaped the scene'
            if k_stale != k:
                stale = self.ds.get_sample(k_stale, agents=[aid], load=load)
                if aid in stale.agents:
                    sample.agents[aid] = stale.agents[aid]
            sample.agents[aid].faults['comm_latency'] = {
                'delta_frames': delta, 'frame_used': k_stale,
                'scene_start': k_min}
            self._log.write(idx=k, agent_id=aid, is_ego=0, stage='latency',
                            seed=_seed(spec.seed, k, aid, 'latency'),
                            detail='frame_used=%d;delta=%d;k_min=%d'
                                   % (k_stale, delta, k_min))
        return sample

    # ── routing (hook 2: sample faults) ─────────────────────────────────

    def _route(self, sample, k, stage, targets, skipped):
        """One routing row per stage per frame -- targets AND skips."""
        if targets:
            detail = 'targets=%s;skipped=%s' % (sorted(targets),
                                                sorted(skipped) or '[]')
        else:
            # Keep the per-agent reasons even when nobody is targeted --
            # "no_target_agents" alone would say the stage skipped but hide
            # WHY (no-lidar vs empty-cloud), which is the audit trail's job.
            detail = 'no_target_agents;skipped=%s' % (sorted(skipped) or '[]')
        self._log.write(idx=k, agent_id='*', is_ego=0, stage=stage, seed=0,
                        detail=detail)

    def _lidar_targets(self, sample, k, stage):
        """LiDAR-carrying agents with non-empty clouds; everything else is a
        logged skip (drone: no-lidar; degenerate: empty-cloud)."""
        supporting = set(agents_supporting(sample, 'lidar'))
        targets, skipped = [], []
        for aid, ag in sample.agents.items():
            if aid not in supporting:
                skipped.append('%s:no-lidar' % aid)
            elif not len(ag.lidar):
                skipped.append('%s:empty-cloud' % aid)
            else:
                targets.append(aid)
        self._route(sample, k, stage, targets, skipped)
        return targets

    def _apply(self, sample, k):
        spec, log = self.spec, self._log

        if spec.agent_drop:
            # Drone only: the vehicle is the ego of every Griffin scene, so
            # the drone is the only droppable agent (user-confirmed).
            targets = [a for a, ag in sample.agents.items()
                       if not ag.is_ego and ag.agent_type == 'drone']
            self._route(sample, k, 'agent_drop', targets,
                        ['%s:ego-protected' % a for a, ag in
                         sample.agents.items() if ag.is_ego])
            for aid in targets:
                s = _seed(spec.seed, k, aid, 'agent_drop')
                rng = np.random.default_rng(s)
                if rng.random() < spec.agent_drop['p_drop']:
                    del sample.agents[aid]
                    sample.meta.setdefault('dropped_agents', []).append(aid)
                    log.write(idx=k, agent_id=aid, is_ego=0,
                              stage='agent_drop', seed=s, detail='dropped')

        if spec.missing_modality:
            targets = [a for a, ag in sample.agents.items()
                       if ag.agent_type == 'drone' and ag.images]
            self._route(sample, k, 'missing_modality', targets,
                        ['%s:not-drone' % a for a, ag in sample.agents.items()
                         if ag.agent_type != 'drone'])
            p = spec.missing_modality['p_drop_rgb']
            for aid in targets:
                agent = sample.agents[aid]
                s = _seed(spec.seed, k, aid, 'missing_modality')
                keep = bernoulli_mask(p, np.random.default_rng(s))
                agent.faults['missing_modality'] = {'m_rgb': int(keep)}
                if not keep:
                    for cam in list(agent.images):
                        agent.images[cam] = drop_image(agent.images[cam])
                    log.write(idx=k, agent_id=aid, is_ego=0,
                              stage='missing_modality', seed=s,
                              detail='cameras_dropped=%d' % len(agent.images))

        for stage in ('lidar_fog', 'lidar_snow', 'points_reduce'):
            cfg = getattr(self.spec, stage)
            if not cfg:
                continue
            targets = self._lidar_targets(sample, k, stage)
            for aid in targets:
                agent = sample.agents[aid]
                s = _seed(spec.seed, k, aid, stage)
                cfg_call = dict(cfg)
                mount = self._lidar_mounts.get(aid)
                if stage == 'lidar_fog':
                    from ..fault_injectors.lidar_fog import LidarFogInjector
                    inj = LidarFogInjector(seed=s, T_lidar_to_ego=mount,
                                           **cfg_call)
                elif stage == 'lidar_snow':
                    from ..fault_injectors.lidar_snow import LidarSnowInjector
                    cfg_call.setdefault('verbose', False)
                    inj = LidarSnowInjector(seed=s, T_lidar_to_ego=mount,
                                            **cfg_call)
                else:
                    # frame-agnostic subsampling: no mount transform needed
                    from ..fault_injectors.lidar_points_reduce import \
                        PointsReductionInjector
                    inj = PointsReductionInjector(seed=s, **cfg_call)
                before = agent.lidar
                with _contained():
                    after = inj(before)
                agent.lidar = after
                detail = ('pts=%d->%d;meanI=%.4f->%.4f'
                          % (len(before), len(after),
                             float(np.asarray(before)[:, 3].mean()),
                             float(np.asarray(after)[:, 3].mean())))
                agent.faults[stage] = detail
                log.write(idx=k, agent_id=aid, is_ego=int(agent.is_ego),
                          stage=stage, seed=s, detail=detail)

        if spec.camera:
            supporting = set(agents_supporting(sample, 'images'))
            targets = sorted(supporting)
            self._route(sample, k, 'camera', targets,
                        ['%s:no-cameras' % a for a in sample.agents
                         if a not in supporting])
            ctor = _camera_injector(spec.camera['kind'])
            kwargs = {kk: vv for kk, vv in spec.camera.items() if kk != 'kind'}
            for aid in targets:
                agent = sample.agents[aid]
                for cam in sorted(agent.images):
                    s = _seed(spec.seed, k, '%s/%s' % (aid, cam), 'camera')
                    with _contained():
                        agent.images[cam] = ctor(seed=s, **kwargs)(
                            agent.images[cam])
                s = _seed(spec.seed, k, aid, 'camera')
                log.write(idx=k, agent_id=aid, is_ego=int(agent.is_ego),
                          stage='camera', seed=s,
                          detail='%s;cams=%d' % (spec.camera['kind'],
                                                 len(agent.images)))

    # ── setup helpers ───────────────────────────────────────────────────

    def _build_scene_starts(self):
        """
        Flat frame index -> first flat index of its scene.

        Uses the vehicle side's ``scene_infos.json`` ``frames`` lists, mapped
        to flat indices through the sorted pose-file basenames (the ordering
        ``GriffinDataset`` indexes by). Any frame not covered by scene_infos
        (none observed) conservatively clamps to itself, i.e. zero staleness
        rather than a cross-scene pairing.
        """
        pose_files = self.ds._files['vehicle']['pose']
        ts_to_idx = {os.path.splitext(os.path.basename(p))[0]: i
                     for i, p in enumerate(pose_files)}
        starts = list(range(len(pose_files)))          # default: clamp to self
        info_path = os.path.join(self.ds.roots['vehicle'], 'scene_infos.json')
        if os.path.exists(info_path):
            with open(info_path) as fh:
                scenes = json.load(fh)
            for scene in scenes:
                frames = scene.get('info', {}).get('frames', [])
                idxs = sorted(ts_to_idx[f] for f in frames if f in ts_to_idx)
                if idxs:
                    first = idxs[0]
                    for i in idxs:
                        starts[i] = first
        return starts

    def _load_lidar_mounts(self):
        """T_lidar_to_ego per agent from calib/lidar_top.json (vehicle only:
        the drone has calib stubs but no lidar data, and no entry is made for
        agents whose calib lacks the file)."""
        from ..data_loaders import load_sensor_extrinsic
        mounts = {}
        for aid, root in self.ds.roots.items():
            try:
                mounts[aid] = load_sensor_extrinsic(
                    os.path.join(root, 'calib'), 'lidar_top')
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                pass
        return mounts
