"""LidarFogInjector - EXACT MultiCorrupt simulate_fog (arXiv:2402.11677).

LiDAR fog: attenuates each return's intensity (Beer-Lambert, P_R_fog_hard) and
converts some returns into nearer fog-scatter points (P_R_fog_soft), using the
precomputed integral lookup tables. Runs on Griffin (N,4) clouds directly - it
uses only x,y,z and intensity (column 3), no ring column needed.
Severity 1/2/3 -> (alpha,beta) = (0.02,0.008)/(0.03,0.008)/(0.06,0.05), i.e. the
300/150/50 m visibility levels of Table I. All physics is verbatim (_mc_lidar.py).

Frames: ranges are measured from the sensor origin. If `T_lidar_to_ego` is given,
input/output are treated as EGO-frame (Griffin `load_lidar` default) and converted
to/from the sensor frame around the verbatim simulation. If None, points are
assumed already sensor-centred (the ~1.1 m mount offset is a small range error).
"""
from __future__ import annotations
import numpy as np
from . import _mc_lidar as _mcl


class LidarFogInjector:
    def __init__(self, severity=2, noise=10, T_lidar_to_ego=None, seed=1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.noise, self.seed = severity, noise, seed
        self.T_lidar_to_ego = None if T_lidar_to_ego is None else np.asarray(T_lidar_to_ego, float)

    def __call__(self, points):
        _mcl.RNG = np.random.default_rng(self.seed)     # reproducible (P_R_fog_soft uses _mcl.RNG)
        pc = np.asarray(points, dtype=np.float64).copy()
        if self.T_lidar_to_ego is not None:
            T_inv = np.linalg.inv(self.T_lidar_to_ego)
            pc[:, :3] = (T_inv @ np.c_[pc[:, :3], np.ones(len(pc))].T).T[:, :3]
        augmented_pc, _fog_pc, _num_fog, _info = _mcl.simulate_fog(self.severity, pc, self.noise)
        if self.T_lidar_to_ego is not None:
            augmented_pc[:, :3] = (self.T_lidar_to_ego @ np.c_[augmented_pc[:, :3],
                                   np.ones(len(augmented_pc))].T).T[:, :3]
        return augmented_pc.astype(np.float32)
