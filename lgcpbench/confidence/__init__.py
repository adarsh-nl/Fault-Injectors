"""
lgcpbench.confidence
====================
Paper contributions C2 -- Eq. 1, 2 and 3.

    Eq. 1   F_i({v_j}) = f_gen(f_i,j)          per-CAV area confidence
    Eq. 2   F_i(V^_i) = 1 - PROD (1 - F_i({v_k}))   noisy-OR over a group
    Eq. 3   accuracy proxy = mean area confidence

The boundary between planes: above here everything is discrete (reports,
groups, schedules); below here everything is tensors.

Example
-------
>>> import torch
>>> from lgcpbench.roi import AreaGrid
>>> from lgcpbench.confidence import AreaConfidenceEstimator, NoisyOrCombiner
>>> grid = AreaGrid((-20.0, -12.0, -3.0, 20.0, 12.0, 1.0))
>>> est = AreaConfidenceEstimator(grid, feature_hw=(8, 16), pooling="max")
>>> conf_map = torch.full((3, 1, 8, 16), 0.5)
>>> matrix = est(conf_map, area_ids=[0, 1, 2])
>>> combiner = NoisyOrCombiner()
>>> round(combiner.combine(matrix.for_area(0)), 4)   # three CAVs at 0.5
0.875
"""

from .combiner import NoisyOrCombiner, global_accuracy_proxy
from .estimator import AreaConfidenceEstimator, AreaConfidenceMatrix
from .pooling import (
    AreaPooling,
    MaxPooling,
    MeanPooling,
    TopKMeanPooling,
    available_poolings,
    make_pooling,
)

__all__ = [
    "AreaConfidenceEstimator",
    "AreaConfidenceMatrix",
    "NoisyOrCombiner",
    "global_accuracy_proxy",
    "AreaPooling",
    "MaxPooling",
    "MeanPooling",
    "TopKMeanPooling",
    "make_pooling",
    "available_poolings",
]
