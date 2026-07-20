"""
lgcpbench.data
==============
The corruption plane (plane 1): datasets that apply physical faults upstream
of every tensor.

Faults are applied by ``corabench.faults.DataFaultBridge`` over
``src.pipeline.FaultPipeline`` -- the user's existing injectors -- on the
``CooperativeSample`` before voxelisation. Nothing downstream corrupts data.

Example
-------
>>> from cpbench.data.preprocessing import GridSpec
>>> from cpbench.data.synthetic import SyntheticCooperativeDataset
>>> from cpbench.faults.bridge import DataFaultBridge
>>> from lgcpbench.data import LGCPDataset
>>> spec = GridSpec((0.4, 0.4), (-38.4, -12.8, -3., 38.4, 12.8, 1.), 4)
>>> bridge = DataFaultBridge({"pipeline": {"pose_error": {"sigma_xy": 0.5}}},
...                          seed=0)
>>> ds = LGCPDataset(SyntheticCooperativeDataset(n_frames=2, n_agents=3),
...                  spec, bridge=bridge)
>>> frame, faults = ds[0]
>>> bool(faults)                      # the audit trail is populated
True
"""

from .cooperative import LGCPDataset
from .opencood_voxelizer import OpenCOODVoxelizer

__all__ = ["LGCPDataset", "OpenCOODVoxelizer"]
