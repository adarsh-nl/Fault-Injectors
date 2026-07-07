"""LiDAR beam reduction (MultiCorrupt, arXiv:2402.11677).

Two classes:

* BeamReductionInjector - EXACT MultiCorrupt reduce_LiDAR_beamsV2, called verbatim
  (_mc_lidar.py). It selects whole scan lines by an INTEGER RING INDEX at column 4
  and its allowed-beam lists are specific to a 32-beam sensor. Griffin clouds are
  (N,4) with no ring column, so on Griffin this raises a clear error rather than
  silently misreading intensity as a ring. Use it only with (N,5) 32-beam clouds
  (e.g. nuScenes) or after adding a genuine 32-beam ring column.

* BeamReductionInjectorGriffin - a SEPARATE Griffin-native reducer (NOT a
  modification of MultiCorrupt's function). Griffin's 80-beam clouds have no ring
  column, so the beam is recovered from each point's elevation angle and binned
  into `num_beams` rings; severity 1/2/3 keeps 1/2, 1/4, 1/8 of the beams (the same
  fractions as MultiCorrupt's 16/8/4 of 32), i.e. 40/20/10 of 80.
"""
from __future__ import annotations
import numpy as np
from . import _mc_lidar as _mcl

_KEEP_FRAC = {1: 0.5, 2: 0.25, 3: 0.125}


class BeamReductionInjector:
    """EXACT reduce_LiDAR_beamsV2 - requires a 32-beam ring index at column 4."""

    def __init__(self, severity: int = 2, seed: int = 1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.seed = severity, seed

    def __call__(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] < 5:
            raise ValueError(
                "reduce_LiDAR_beamsV2 needs a ring-index column at index 4 (nuScenes-style "
                f"(N,5), 32-beam); got shape {pts.shape}. Griffin clouds are (N,4) with no ring "
                "column - use BeamReductionInjectorGriffin instead, or add a 32-beam ring column.")
        return _mcl.reduce_LiDAR_beamsV2(pts.copy(), self.severity)


class BeamReductionInjectorGriffin:
    """Griffin-native beam reduction via elevation binning (80-beam, no ring column)."""

    def __init__(self, severity: int = 2, num_beams: int = 80, ring_col=None, seed: int = 1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.num_beams, self.ring_col, self.seed = severity, num_beams, ring_col, seed

    def __call__(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points)
        if self.ring_col is not None and pts.shape[1] > self.ring_col:
            bidx = pts[:, self.ring_col].astype(int)
        else:
            r_xy = np.hypot(pts[:, 0], pts[:, 1])
            phi = np.arctan2(pts[:, 2], r_xy + 1e-12)
            lo, hi = float(phi.min()), float(phi.max())
            bidx = np.clip(np.floor((phi - lo) / (hi - lo + 1e-12) * self.num_beams).astype(int),
                           0, self.num_beams - 1)
        keep = max(1, int(round(self.num_beams * _KEEP_FRAC[self.severity])))
        kept = np.unique(np.linspace(0, self.num_beams - 1, keep).round().astype(int))
        return pts[np.isin(bidx, kept)]
