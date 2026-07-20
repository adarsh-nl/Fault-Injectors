"""The corruption plane: physical faults applied upstream of every tensor.

``DataFaultBridge`` wraps ``src.pipeline.FaultPipeline`` and corrupts the
``CooperativeSample`` BEFORE voxelisation. No model, scheduler or metric code
ever corrupts a tensor.
"""

from .bridge import DataFaultBridge, FaultRecord

__all__ = ["DataFaultBridge", "FaultRecord"]
