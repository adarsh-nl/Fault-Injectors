"""
lgcpbench.roi
=============
Paper contribution C1: partition of the road of interest into non-overlapping
areas, restricted to the areas currently occupied by vehicles.

Pure geometry -- numpy only, no torch, no dataset, no OpenCOOD. Everything
here unit-tests standalone on CPU.

Example
-------
>>> import numpy as np
>>> from lgcpbench.roi import AreaGrid, BoxOccupancy
>>> grid = AreaGrid(point_range=(-140.8, -38.4, -3.0, 140.8, 38.4, 1.0))
>>> len(grid)
377
>>> occ = BoxOccupancy()(grid, boxes=np.array([[0.0, 0.0], [25.0, 3.0]]))
>>> int(occ.sum())
2
>>> int(grid.cell_counts((48, 176)).sum())   # strict partition of the BEV map
8448
"""

from .grid import DEFAULT_AREA_SIZE_M, Area, AreaGrid
from .occupancy import (
    AllAreasOccupancy,
    BoxOccupancy,
    OccupancyEstimator,
    available_occupancy_sources,
    make_occupancy_estimator,
)

__all__ = [
    "Area",
    "AreaGrid",
    "DEFAULT_AREA_SIZE_M",
    "OccupancyEstimator",
    "AllAreasOccupancy",
    "BoxOccupancy",
    "make_occupancy_estimator",
    "available_occupancy_sources",
]
