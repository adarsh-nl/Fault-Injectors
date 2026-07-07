"""
opencood_integration.py
-----------------------
Inject this repository's cooperative faults into an OpenCOOD-style
dataloader -- the framework behind the reference implementations of
V2VNet (arXiv:2008.07519, OpenCOOD re-impl), CoBEVT (arXiv:2207.02202),
Where2comm (arXiv:2209.12836) and V2X-ViT (arXiv:2203.10638).

Every OpenCOOD dataset (early/intermediate/late fusion) funnels through
one method:

    base_data_dict = self.retrieve_base_data(idx)
    # OrderedDict {cav_id: {'ego': bool,
    #                       'params': <the frame's yaml dict>,
    #                       'lidar_np': (N, 4) float array, ...}}

so corrupting that dict corrupts what EVERY fusion model sees, with no
model-specific code. This module provides `corrupt_base_data`, plus a
mixin that patches it into any OpenCOOD dataset class.

Usage (inside a V2X-ViT / CoBEVT / Where2comm eval script):

    from opencood.data_utils.datasets import build_dataset
    from examples.opencood_integration import add_faults_to_dataset

    dataset = build_dataset(hypes, visualize=False, train=False)
    add_faults_to_dataset(
        dataset,
        pose_error={'sigma_xy': 0.4, 'sigma_heading': 0.4},
        agent_drop={'p_drop': 0.25},
        bandwidth={'keep_fraction': 0.5},
        seed=2026,
    )
    # ... evaluate as usual; every retrieved sample is now corrupted.

Latency note: OpenCOOD's V2X-ViT fork already implements asynchronous
data retrieval natively (`wild_setting: async_mode / time_delay` in the
yaml hypes); prefer that for latency there, and use this repository's
CommLatencyInjector when working outside OpenCOOD.
"""

import sys
from pathlib import Path

# make `src` importable when this file is run from the repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fault_injectors import (  # noqa: E402
    AgentDropInjector, BandwidthLimitInjector, PoseErrorInjector,
)


def corrupt_base_data(base_data_dict, pose_injector=None, drop_injector=None,
                      bandwidth_injector=None):
    """
    Corrupt one OpenCOOD `retrieve_base_data` output in place.

    The ego cav (entry with ['ego'] == True) is never corrupted: pose error,
    packet loss and bandwidth limits are properties of the V2X link, and the
    ego's own sensors do not cross it.
    """
    ego_ids = [cid for cid, c in base_data_dict.items() if c.get('ego')]
    protect = set(ego_ids)

    if drop_injector is not None:
        keep = drop_injector.keep_mask(list(base_data_dict), protect=protect)
        for cav_id in [c for c, k in keep.items() if not k]:
            del base_data_dict[cav_id]

    for cav_id, cav in base_data_dict.items():
        if cav_id in protect:
            continue
        if pose_injector is not None and 'params' in cav:
            params = cav['params']
            if 'lidar_pose' in params:
                params['lidar_pose'] = pose_injector.perturb_pose6(
                    params['lidar_pose'])
        if bandwidth_injector is not None and cav.get('lidar_np') is not None:
            cav['lidar_np'] = bandwidth_injector(cav['lidar_np'])
    return base_data_dict


def add_faults_to_dataset(dataset, pose_error=None, agent_drop=None,
                          bandwidth=None, seed=0):
    """
    Monkey-patch an instantiated OpenCOOD dataset so that every
    `retrieve_base_data` call is corrupted. Returns the dataset.

    Parameters mirror `FaultPipeline.from_config`:
        pose_error : kwargs for PoseErrorInjector  (e.g. sigma_xy in metres,
                     sigma_heading in degrees -- the V2X-ViT noise setting)
        agent_drop : kwargs for AgentDropInjector  (p_drop, burst)
        bandwidth  : kwargs for BandwidthLimitInjector (keep_fraction, ...)
    """
    pose_inj = PoseErrorInjector(seed=seed, **pose_error) if pose_error else None
    drop_inj = AgentDropInjector(seed=seed + 1, **agent_drop) if agent_drop else None
    bw_inj = BandwidthLimitInjector(seed=seed + 2, **bandwidth) if bandwidth else None

    original = dataset.retrieve_base_data

    def retrieve_base_data(idx, *args, **kwargs):
        base = original(idx, *args, **kwargs)
        return corrupt_base_data(base, pose_inj, drop_inj, bw_inj)

    dataset.retrieve_base_data = retrieve_base_data
    return dataset


if __name__ == '__main__':
    # Self-contained demo on a fake base_data_dict (no OpenCOOD install).
    from collections import OrderedDict

    import numpy as np

    base = OrderedDict({
        '641': {'ego': True, 'params': {'lidar_pose': [0, 0, 1.9, 0, 0, 0]},
                'lidar_np': np.random.rand(100, 4)},
        '650': {'ego': False, 'params': {'lidar_pose': [12, 5, 1.9, 0, 30, 0]},
                'lidar_np': np.random.rand(100, 4)},
    })
    corrupt_base_data(
        base,
        pose_injector=PoseErrorInjector(sigma_xy=0.4, sigma_heading=0.4),
        bandwidth_injector=BandwidthLimitInjector(keep_fraction=0.5),
    )
    print('ego pose (untouched):', base['641']['params']['lidar_pose'])
    print('cav pose (perturbed):',
          np.round(base['650']['params']['lidar_pose'], 3).tolist())
    print('cav points after bandwidth limit:', base['650']['lidar_np'].shape)
