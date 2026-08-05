"""
Griffin adapter login-node gates. No GPU, no model, no sweep.

    A  null round trip        -- null spec returns fields bit-identical
    B  per-agent fires        -- each injector hits exactly its agent type
    C  skips-not-noops        -- no-target / no-lidar / empty-cloud are LOGGED
    D  ego protection         -- vehicle undroppable, ego frame never stale
    E  latency scene clamp    -- stale frames stay inside their own scene
    F  snow parameterisation  -- per-severity removal fractions (REVIEW gate:
                                 numbers for the user, not a pass/fail)

    .venv-hpc/bin/python tools/test_griffin_gates.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters.griffin import (FaultedGriffinDataset,          # noqa: E402
                                  GriffinFaultSpec, _seed)
from src.datasets import load_dataset                             # noqa: E402

ROOT = ('/datasets/eemcs/ps/cv/huggingface/griffin/datasets/'
        'griffin_50scenes_25m/griffin_50scenes_25m/griffin-release')

FAILURES, NCHECKS = [], [0]


def check(name, cond, extra=''):
    NCHECKS[0] += 1
    print('%-4s %s%s' % ('ok' if cond else 'FAIL', name,
                         ('  -- ' + extra) if extra and not cond else ''))
    if not cond:
        FAILURES.append(name)


def read_log(log_dir):
    import csv
    import glob
    rows = []
    for f in glob.glob(os.path.join(log_dir, 'injection_summary.*.csv')):
        rows += list(csv.DictReader(open(f)))
    return rows


def fields_equal(a, b):
    """Bit-identical comparison of two AgentFrames' payloads."""
    if (a.lidar is None) != (b.lidar is None):
        return False
    if a.lidar is not None and not np.array_equal(a.lidar, b.lidar):
        return False
    if sorted(a.images) != sorted(b.images):
        return False
    for cam in a.images:
        if not np.array_equal(a.images[cam], b.images[cam]):
            return False
    if (a.pose is None) != (b.pose is None):
        return False
    if a.pose is not None and not np.array_equal(a.pose, b.pose):
        return False
    return len(a.labels) == len(b.labels)


def main():
    tmp = tempfile.mkdtemp(prefix='griffin_gates_')
    ds = load_dataset('griffin', veh_root=ROOT + '/vehicle-side',
                      drone_root=ROOT + '/drone-side')
    K = 5                                    # mid-scene frame used throughout

    # ── A. null round trip ──────────────────────────────────────────────
    ref = ds.get_sample(K)
    null = FaultedGriffinDataset(ds, GriffinFaultSpec()).get_sample(K)
    check('A: agent sets equal', sorted(null.agents) == sorted(ref.agents))
    for aid in ref.agents:
        check('A: %s bit-identical' % aid,
              fields_equal(null.agents[aid], ref.agents[aid]))
    check('A: ego id preserved', null.ego_id == ref.ego_id == 'vehicle')

    # ── B. per-agent injection-fires ────────────────────────────────────
    def run(spec_kw, name):
        d = os.path.join(tmp, name)
        os.makedirs(d, exist_ok=True)
        f = FaultedGriffinDataset(ds, GriffinFaultSpec(seed=1234, log_dir=d,
                                                       **spec_kw))
        return f.get_sample(K), read_log(d)

    # LiDAR fault: vehicle cloud changes, drone untouched
    s, rows = run({'lidar_fog': {'severity': 2}}, 'fog')
    check('B: fog changed vehicle cloud',
          not np.array_equal(s.agents['vehicle'].lidar,
                             ref.agents['vehicle'].lidar))
    check('B: fog left drone lidar-less', s.agents['drone'].lidar is None)
    check('B: fog left all images untouched',
          all(np.array_equal(s.agents[a].images[c], ref.agents[a].images[c])
              for a in s.agents for c in s.agents[a].images))
    fired = [r for r in rows if r['stage'] == 'lidar_fog'
             and r['agent_id'] == 'vehicle']
    routed = [r for r in rows if r['agent_id'] == '*']
    check('B: fog fired on vehicle only', len(fired) == 1
          and not any(r['agent_id'] == 'drone' for r in rows
                      if r['stage'] == 'lidar_fog'))
    check('C: fog routing row logs drone skip',
          any('drone:no-lidar' in r['detail'] for r in routed), str(routed))

    # camera fault: both agents' images change, vehicle cloud untouched
    s, rows = run({'camera': {'kind': 'brightness', 'severity': 2}}, 'cam')
    for aid in ('vehicle', 'drone'):
        check('B: brightness changed %s images' % aid,
              all(not np.array_equal(s.agents[aid].images[c],
                                     ref.agents[aid].images[c])
                  for c in s.agents[aid].images))
    check('B: brightness left vehicle cloud untouched',
          np.array_equal(s.agents['vehicle'].lidar,
                         ref.agents['vehicle'].lidar))
    check('B: per-camera draws differ (front vs back seeds)',
          _seed(1234, K, 'drone/front', 'camera')
          != _seed(1234, K, 'drone/back', 'camera'))

    # MissingModality p=1: drone cameras zeroed, vehicle untouched
    s, rows = run({'missing_modality': {'p_drop_rgb': 1.0}}, 'mm')
    check('B: missing_modality zeroed drone cameras',
          all(not s.agents['drone'].images[c].any()
              for c in s.agents['drone'].images))
    check('B: missing_modality left vehicle images untouched',
          all(np.array_equal(s.agents['vehicle'].images[c],
                             ref.agents['vehicle'].images[c])
              for c in s.agents['vehicle'].images))
    check('B: missing_modality logged the drop',
          any(r['stage'] == 'missing_modality' and 'cameras_dropped'
              in r['detail'] for r in rows))

    # AgentDrop p=1: drone gone, vehicle (ego) present
    s, rows = run({'agent_drop': {'p_drop': 1.0}}, 'drop')
    check('B: agent_drop removed the drone', 'drone' not in s.agents)
    check('D: agent_drop kept the ego vehicle', 'vehicle' in s.agents)
    check('D: routing row records ego protection',
          any('vehicle:ego-protected' in r['detail'] for r in rows))

    # ── C. skips-not-noops on degenerate samples ────────────────────────
    # no LiDAR agents at all -> no_target_agents, logged, no crash
    d = os.path.join(tmp, 'no_target')
    os.makedirs(d)
    f = FaultedGriffinDataset(ds, GriffinFaultSpec(
        seed=1, log_dir=d, lidar_fog={'severity': 2}))
    sample = ds.get_sample(K)
    sample.agents['vehicle'].lidar = None            # simulate lidar-less scene
    f._apply(sample, K)
    rows = read_log(d)
    check('C: all-agents-lidar-less -> no_target_agents logged with reasons',
          any(r['detail'].startswith('no_target_agents')
              and 'no-lidar' in r['detail'] for r in rows), str(rows))

    # empty cloud -> logged skip, injector NOT run
    d = os.path.join(tmp, 'empty')
    os.makedirs(d)
    f = FaultedGriffinDataset(ds, GriffinFaultSpec(
        seed=1, log_dir=d, lidar_fog={'severity': 2}))
    sample = ds.get_sample(K)
    sample.agents['vehicle'].lidar = np.zeros((0, 4), np.float32)
    f._apply(sample, K)
    rows = read_log(d)
    check('C: empty cloud -> logged skip, not fed to fog',
          any('vehicle:empty-cloud' in r['detail'] for r in rows)
          and not any(r['stage'] == 'lidar_fog' and r['agent_id'] == 'vehicle'
                      for r in rows))
    check('C: empty cloud unchanged',
          sample.agents['vehicle'].lidar.shape == (0, 4))

    # ── D/E. latency: ego current, drone stale, scene-clamped ───────────
    d = os.path.join(tmp, 'lat')
    os.makedirs(d)
    f = FaultedGriffinDataset(ds, GriffinFaultSpec(
        seed=1234, log_dir=d, latency={'mu_delay': 0.5, 'sigma_jitter': 0.1}))
    starts = f._scene_start
    scene2 = sorted(set(starts))[1]              # first frame of scene 2
    hits_boundary = clamped = 0
    for k in [scene2, scene2 + 1, scene2 + 2, scene2 + 60, scene2 + 61]:
        s = f.get_sample(k, load=('lidar',))
        fault = s.agents['drone'].faults.get('comm_latency', {})
        used, k_min = fault.get('frame_used'), fault.get('scene_start')
        check('E: k=%d stale frame %s within scene (start %s)'
              % (k, used, k_min), used is not None and used >= k_min == starts[k])
        if used == k_min and fault['delta_frames'] > k - k_min:
            clamped += 1
        if k - starts[k] < 5:
            hits_boundary += 1
        check('D: k=%d ego has no latency fault' % k,
              'comm_latency' not in s.agents['vehicle'].faults)
    check('E: boundary frames exercised the clamp (%d cases, %d clamped)'
          % (hits_boundary, clamped), hits_boundary >= 3 and clamped >= 1)
    # determinism: same k -> same stale frame
    a = f.get_sample(scene2 + 60, load=('lidar',)).agents['drone'] \
        .faults['comm_latency']['frame_used']
    b = f.get_sample(scene2 + 60, load=('lidar',)).agents['drone'] \
        .faults['comm_latency']['frame_used']
    check('E: latency deterministic per (seed, k)', a == b)

    # ── F. snow parameterisation (REVIEW numbers, not pass/fail) ────────
    print('\n--- F: snow per-severity removal on Griffin vehicle clouds ---')
    from src.fault_injectors.lidar_snow import LidarSnowInjector
    from src.fault_injectors.snowflake_sampling import ensure_particle_files
    v = ref.agents['vehicle'].lidar.astype(np.float64)
    mount = FaultedGriffinDataset(ds, GriffinFaultSpec())._lidar_mounts['vehicle']
    for sev in (1, 2, 3):
        try:
            ensure_particle_files(sev, verbose=False)
            inj = LidarSnowInjector(severity=sev, seed=1000,
                                    T_lidar_to_ego=mount, verbose=False)
            out = inj(v.copy())
            fl = inj.last_flags
            print('  sev%d: pts %d->%d  REMOVED %.1f%%  attenuated=%d '
                  'scatter=%d  meanI %.4f->%.4f'
                  % (sev, len(v), len(out), 100.0 * (len(v) - len(out)) / len(v),
                     int((fl == 1).sum()), int((fl == 2).sum()),
                     v[:, 3].mean(), out[:, 3].mean()))
        except Exception as e:                      # noqa: BLE001
            print('  sev%d: NOT RUN (%s: %s)' % (sev, type(e).__name__, e))

    shutil.rmtree(tmp, ignore_errors=True)
    print('\n%d checks, %d failures' % (NCHECKS[0], len(FAILURES)))
    if FAILURES:
        print('FAILED:', FAILURES)
        return 1
    print('GRIFFIN GATES PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
