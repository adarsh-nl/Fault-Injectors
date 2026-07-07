"""PointsReductionInjector - EXACT MultiCorrupt pointsreducing (arXiv:2402.11677).

Uniform random point dropout; severity 1/2/3 drops p = 70/80/90 % (Table I),
keeping (100 - p) %. Verbatim logic (_mc_lidar.py); wrapper only seeds numpy.
"""
from __future__ import annotations
import numpy as np
from . import _mc_lidar as _mcl


class PointsReductionInjector:
    def __init__(self, severity: int = 2, seed: int = 1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.seed = severity, seed

    def __call__(self, points: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)                      # pointsreducing uses np.random.permutation
        return _mcl.pointsreducing(np.asarray(points).copy(), self.severity)
