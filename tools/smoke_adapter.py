"""
CPU-only smoke test of the adapter against REAL V2XSet data.

Builds the real ``IntermediateFusionDataset`` over one scenario, pulls a few
samples through (a) the official path, (b) the null wrapper, (c) the sev-2
wrapper, and compares. No GPU, no model -- login-node safe, single core.

Checks the things a synthetic dict cannot: real ``reform_param`` output, real
``pcd_to_np`` clouds, real yaml, and whether ``__getitem__`` survives the
rebuilt dict all the way to preprocessed features.

    ~/.conda/envs/opencood-official/bin/python tools/smoke_adapter.py
"""

import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CFG = os.path.expanduser('~/opencood-eval/cobevt/config.yaml')
ROOT = '/datasets/eemcs/ps/cv/opencood/v2xset/test'
N = 4


def build(spec_kw, root):
    import opencood.data_utils.datasets as ocds
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from src.adapters import FaultSpec, make_faulty_dataset

    hypes = yaml_utils.load_yaml(CFG)
    hypes['root_dir'] = root
    hypes['validate_dir'] = root
    ws = hypes.get('wild_setting')
    if ws:
        ws['loc_err'] = False
        ws['async'] = False

    cls = ocds.IntermediateFusionDataset
    if spec_kw is not None:
        cls = make_faulty_dataset(cls, FaultSpec(seed=1234, **spec_kw))
    return cls(params=hypes, visualize=False, train=False)


def main():
    # One scenario only: enough to exercise every path, seconds to load.
    scen = sorted(os.listdir(ROOT))[0]
    root = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'fi_smoke_root')
    if not os.path.exists(root):
        os.makedirs(root)
    link = os.path.join(root, scen)
    if not os.path.exists(link):
        os.symlink(os.path.join(ROOT, scen), link)
    print('scenario: %s' % scen)

    ds_off = build(None, root)
    ds_null = build({}, root)
    ds_pose = build({'pose_error': {'sigma_xy': 0.2, 'sigma_heading': 0.2}},
                    root)
    print('%d samples' % len(ds_off))

    fails = []
    dxs = []
    for idx in range(N):
        b_off = ds_off.retrieve_base_data(idx)
        b_null = ds_null.retrieve_base_data(idx)
        b_pose = ds_pose.retrieve_base_data(idx)

        assert list(b_off) == list(b_null), 'null changed key order'
        ego = [k for k in b_off if b_off[k]['ego']][0]

        for k in b_off:
            a, b = b_off[k]['params'], b_null[k]['params']
            d = np.abs(a['transformation_matrix'] -
                       b['transformation_matrix']).max()
            if d != 0.0:
                fails.append('idx %d cav %s: null T differs by %.3e' % (idx, k, d))
            if not np.array_equal(b_off[k]['lidar_np'], b_null[k]['lidar_np']):
                fails.append('idx %d cav %s: null lidar differs' % (idx, k))
            if b_off[k]['lidar_np'].dtype != b_null[k]['lidar_np'].dtype:
                fails.append('idx %d cav %s: null lidar dtype differs' % (idx, k))

            # faulty: ego must be untouched, non-ego must move
            dp = np.abs(a['transformation_matrix'] -
                        b_pose[k]['params']['transformation_matrix']).max()
            if k == ego and dp != 0.0:
                fails.append('idx %d EGO T moved by %.3e' % (idx, dp))
            if k != ego:
                if dp == 0.0:
                    fails.append('idx %d cav %s: pose fault had NO effect' % (idx, k))
                dxs.append(abs(b_pose[k]['params']['transformation_matrix'][0, 3]
                               - a['transformation_matrix'][0, 3]))
            if not np.array_equal(a['gt_transformation_matrix'],
                                  b_pose[k]['params']['gt_transformation_matrix']):
                fails.append('idx %d cav %s: GT matrix moved' % (idx, k))

        print('idx %d: %d cavs (ego=%s)  ok' % (idx, len(b_off), ego))

    # full __getitem__ must survive the rebuilt dict
    for name, ds in (('null', ds_null), ('pose_sev2', ds_pose)):
        item = ds[0]['ego']
        vf = item['processed_lidar']['voxel_features']
        print('%s: __getitem__ ok, voxel_features %s, objects %d'
              % (name, np.asarray(vf).shape, int(item['object_bbx_mask'].sum())))

    # GT identical between clean and faulty -- the delta must be attributable
    g0 = ds_null[0]['ego']['object_bbx_center']
    g1 = ds_pose[0]['ego']['object_bbx_center']
    if not np.array_equal(g0, g1):
        fails.append('GT boxes differ between clean and faulty')
    else:
        print('GT boxes identical clean vs faulty: ok')

    if dxs:
        print('mean |dTx| over %d non-ego agents = %.4f m '
              '(expect ~0.160 for sigma=0.2)' % (len(dxs), float(np.mean(dxs))))

    print('\n%d failures' % len(fails))
    for f in fails:
        print('  FAIL %s' % f)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
