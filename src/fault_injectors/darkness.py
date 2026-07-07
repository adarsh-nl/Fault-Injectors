"""DarknessInjector - EXACT MultiCorrupt darkness (arXiv:2402.11677).

Low-light: range compression (imadjust, gamma=2) + Poisson-Gaussian noise (s=25/12/5). Camera only.
The corruption function is MultiCorrupt's converter/img.py 'darkness', called verbatim
(see _mc_image.py). The wrapper only seeds numpy for reproducibility and returns
uint8; it does not alter the algorithm.
"""
from __future__ import annotations
import numpy as np
from ._mc_image import darkness as _mc_fn


class DarknessInjector:
    def __init__(self, severity: int = 2, seed: int = 1000):
        if severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {severity}")
        self.severity, self.seed = severity, seed

    def __call__(self, image: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)                      # MultiCorrupt seeds numpy globally
        out = np.asarray(_mc_fn(np.asarray(image), self.severity))
        return np.clip(out, 0, 255).astype(np.uint8)
