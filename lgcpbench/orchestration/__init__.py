"""
lgcpbench.orchestration
=======================
The RSU controller and the full LGCP cycle (paper Algorithm 3, section III).

This is the only package that knows about all the others. Everything below it
-- roi, confidence, selection, network, perception -- is independently
importable and independently testable, which is what keeps the control plane
exercisable without a backbone, a dataset or a GPU.

Example
-------
>>> from lgcpbench.orchestration import FrameInput, LGCPPipeline, RSUController
"""

from .global_view import AGGREGATION_MODES, GlobalViewAggregator
from .pipeline import CommAccounting, FrameInput, FrameResult, LGCPPipeline
from .rsu import RSUController

__all__ = [
    "RSUController",
    "LGCPPipeline",
    "FrameInput",
    "FrameResult",
    "CommAccounting",
    "GlobalViewAggregator",
    "AGGREGATION_MODES",
]
