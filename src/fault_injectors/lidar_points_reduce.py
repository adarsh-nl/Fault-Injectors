"""PointsReductionInjector - EXACT MultiCorrupt pointsreducing (arXiv:2402.11677).

Uniform random point dropout; severity 1/2/3 drops p = 70/80/90 % (Table I),
keeping (100 - p) %. Verbatim logic (_mc_lidar.py, one
``np.random.permutation`` on the global stream).

RNG contract (fixed 2026-08-05; the earlier wrapper called
``np.random.seed(self.seed)`` bare -- global, fixed, every call -- which gave
every agent and every frame the identical permutation prefix AND rewound the
global stream OpenCOOD's own ``shuffle_points`` draws from):

* The global numpy RNG is seeded for the verbatim backend and RESTORED
  afterwards, so a call is invisible to the caller's global stream (same
  containment as ``LidarSnowInjector``).
* Per-sample independence is the caller's contract, identical to every other
  injector here: construct per ``(frame, agent)`` with a seed derived via
  ``SeedSequence(entropy=base, spawn_key=(idx, crc32(agent), stage))`` --
  see ``src/adapters/runtime.py`` / ``src/adapters/griffin.py``. Same seed in
  -> identical reduction out; different derived seeds -> independent
  permutations.
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
        state = np.random.get_state()
        try:
            np.random.seed(self.seed)          # pointsreducing uses np.random.permutation
            return _mcl.pointsreducing(np.asarray(points).copy(), self.severity)
        finally:
            np.random.set_state(state)
