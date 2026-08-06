"""
Gate 1 -- null-pipeline field-level round trip of the OpenCOOD adapter.

Runs on the login node in seconds, no GPU, no OpenCOOD import. Builds a
synthetic ``retrieve_base_data`` dict with OpenCOOD's exact schema and asserts
that ``from_canonical(to_canonical(d))`` reproduces every field the model
consumes -- and that the faulty path moves exactly one thing.

    ~/.conda/envs/opencood-official/bin/python tools/test_gate1.py
"""

import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters.opencood import ModalityError, OpenCOODAdapter   # noqa: E402
from src.adapters.runtime import _seed                             # noqa: E402
from src.datasets.opv2v import x_to_world                          # noqa: E402

FAILURES = []
NCHECKS = [0]


def check(name, cond, extra=''):
    NCHECKS[0] += 1
    print('%-4s %s%s' % ('ok' if cond else 'FAIL', name,
                         ('  -- ' + extra) if extra and not cond else ''))
    if not cond:
        FAILURES.append(name)


def x1_to_x2(x1, x2):
    """OpenCOOD's composition, reproduced for the reference values."""
    return np.linalg.inv(x_to_world(x2)) @ x_to_world(x1)


def make_base(n_pts=(500, 300, 120)):
    rng = np.random.default_rng(7)
    poses = {'641': [10.0, -5.0, 1.9, 0.1, 37.0, -0.2],
             '650': [48.0, 12.0, 1.9, -0.1, -110.0, 0.3],
             '-1':  [-20.0, 30.0, 5.0, 0.0, 90.0, 0.0]}
    ego = '641'
    base = OrderedDict()
    for (cav, pose), n in zip(poses.items(), n_pts):
        params = {
            'lidar_pose': list(pose),
            'ego_speed': 7.5,
            'vehicles': {12: {'location': [1.0, 2.0, 0.5],
                              'center': [0.0, 0.0, 0.7],
                              'extent': [2.3, 1.0, 0.75],
                              'angle': [0.0, 45.0, 0.0]}},
            'transformation_matrix': x1_to_x2(pose, poses[ego]),
            'gt_transformation_matrix': x1_to_x2(pose, poses[ego]),
            'spatial_correction_matrix': np.eye(4),
        }
        base[cav] = {'ego': cav == ego, 'time_delay': 0, 'params': params,
                     'lidar_np': rng.normal(0, 20, (n, 4)).astype(np.float32)}
    return base, ego


def main():
    adapter = OpenCOODAdapter()

    # ── null round trip ─────────────────────────────────────────────────
    base, ego = make_base()
    ref = {k: {'lidar': v['lidar_np'].copy(),
               'T': v['params']['transformation_matrix'].copy(),
               'gt': v['params']['gt_transformation_matrix'].copy(),
               'sc': v['params']['spatial_correction_matrix'].copy(),
               'pose': list(v['params']['lidar_pose']),
               'speed': v['params']['ego_speed'],
               'veh': v['params']['vehicles']}
           for k, v in base.items()}

    sample = adapter.to_canonical(base, frame_index=0)
    out = adapter.from_canonical(sample, base)

    check('key order preserved', list(out) == list(ref))
    check('ego is first key', out[list(out)[0]]['ego'] is True)
    check('agent count', len(out) == 3)

    worst_T = 0.0
    for k, v in out.items():
        p = v['params']
        worst_T = max(worst_T,
                      np.abs(p['transformation_matrix'] - ref[k]['T']).max())
        check('lidar values %s' % k,
              np.array_equal(v['lidar_np'], ref[k]['lidar']))
        check('lidar dtype %s' % k, v['lidar_np'].dtype == np.float32)
        check('lidar shape %s' % k, v['lidar_np'].shape == ref[k]['lidar'].shape)
        check('lidar columns=4 %s' % k, v['lidar_np'].shape[1] == 4)
        check('lidar_pose clean %s' % k, list(p['lidar_pose']) == ref[k]['pose'])
        check('ego_speed passthrough %s' % k, p['ego_speed'] == ref[k]['speed'])
        check('vehicles identical object %s' % k, p['vehicles'] is ref[k]['veh'])
        check('gt_transformation_matrix untouched %s' % k,
              np.array_equal(p['gt_transformation_matrix'], ref[k]['gt']))
        check('spatial_correction_matrix untouched %s' % k,
              np.array_equal(p['spatial_correction_matrix'], ref[k]['sc']))
        check('time_delay passthrough %s' % k, v['time_delay'] == 0)

    check('transformation_matrix EXACT (max|d|=%.3e)' % worst_T,
          worst_T == 0.0, 'expected bit-identical, got %.3e' % worst_T)

    # ── infrastructure / ego typing ─────────────────────────────────────
    check('RSU typed infrastructure',
          sample.agents['-1'].agent_type == 'infrastructure')
    check('ego flagged once',
          sum(a.is_ego for a in sample.agents.values()) == 1)
    check('ego_id correct', sample.ego_id == ego)

    # ── modality gate ───────────────────────────────────────────────────
    try:
        adapter.assert_modality(sample, 'images')
        check('image injector blocked on LiDAR-only', False, 'no raise')
    except ModalityError:
        check('image injector blocked on LiDAR-only', True)
    try:
        adapter.assert_modality(sample, 'lidar')
        check('lidar gate passes', True)
    except ModalityError:
        check('lidar gate passes', False)

    # ── faulty path moves exactly one thing ─────────────────────────────
    from src.fault_injectors.pose_error import PoseErrorInjector
    base2, _ = make_base()
    s2 = adapter.to_canonical(base2, frame_index=0)
    ego_pose_before = s2.ego.pose.copy()
    for aid, ag in s2.agents.items():
        if ag.is_ego:
            continue
        inj = PoseErrorInjector(sigma_xy=0.2, sigma_heading=0.2,
                                seed=_seed(1234, 0, aid, 'pose_error'))
        ag.pose = inj.perturb_matrix(ag.pose, error=inj.sample_error())
    out2 = adapter.from_canonical(s2, base2)

    check('ego pose untouched by fault',
          np.array_equal(s2.ego.pose, ego_pose_before))
    check('ego transformation_matrix unchanged',
          np.array_equal(out2[ego]['params']['transformation_matrix'],
                         ref[ego]['T']))
    moved = [k for k in out2
             if not np.array_equal(out2[k]['params']['transformation_matrix'],
                                   ref[k]['T'])]
    check('exactly the 2 non-ego matrices moved', sorted(moved) == ['-1', '650'],
          'moved=%s' % moved)
    check('GT untouched under fault',
          all(np.array_equal(out2[k]['params']['gt_transformation_matrix'],
                             ref[k]['gt']) for k in out2))
    check('lidar untouched by PoseError',
          all(np.array_equal(out2[k]['lidar_np'], ref[k]['lidar'])
              for k in out2))

    # ── seeding is process-stable and per-(idx, agent) ───────────────────
    check('seed stable across calls',
          _seed(1234, 5, '650', 'pose_error') == _seed(1234, 5, '650', 'pose_error'))
    check('seed varies with idx',
          _seed(1234, 5, '650', 'pose_error') != _seed(1234, 6, '650', 'pose_error'))
    check('seed varies with agent',
          _seed(1234, 5, '650', 'pose_error') != _seed(1234, 5, '-1', 'pose_error'))
    check('seed varies with stage',
          _seed(1234, 5, '650', 'pose_error') != _seed(1234, 5, '650', 'agent_drop'))

    # ── drop path keeps ego first ───────────────────────────────────────
    base3, _ = make_base()
    s3 = adapter.to_canonical(base3, frame_index=0)
    del s3.agents['650']
    out3 = adapter.from_canonical(s3, base3)
    check('drop: agent removed', list(out3) == ['641', '-1'])
    check('drop: ego still first', out3[list(out3)[0]]['ego'] is True)

    s4 = adapter.to_canonical(make_base()[0], frame_index=0)
    del s4.agents['641']
    try:
        adapter.from_canonical(s4, base3)
        check('dropping ego refused', False, 'no raise')
    except ValueError:
        check('dropping ego refused', True)

    print('\n%d checks, %d failures' % (NCHECKS[0], len(FAILURES)))
    if FAILURES:
        print('FAILED: %s' % FAILURES)
        return 1
    print('GATE 1 PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
