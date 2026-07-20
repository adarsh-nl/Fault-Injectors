"""
lgcpbench.perception.opencood
=============================
Run pretrained OpenCOOD models (Where2comm, CoBEVT, CoAlign) inside LGCP.

This subpackage is the ONLY place OpenCOOD is referenced. The core of
lgcpbench depends on the ``CollabPerceptionModel`` protocol and never on
OpenCOOD, which is what lets the rest of the project run on Python 3.9 CPU
while OpenCOOD stays pinned to Python 3.7 + spconv + CUDA.

Importing ``adapter`` does not import OpenCOOD; only
``OpenCOODBackbone.from_config`` does, and it fails with actionable guidance
when the environment is absent.

Verification status: written against OpenCOOD sources read at
github.com/DerrickXuNu/OpenCOOD@main and unit-tested against structural
stubs. NOT yet executed against real weights -- see adapter.py.
"""

from .fusion import (
    CoAlignFusion,
    CoBEVTFusion,
    FusionStrategy,
    Where2commFusion,
    available_core_methods,
    build_fusion_strategy,
)

__all__ = [
    "FusionStrategy",
    "Where2commFusion",
    "CoBEVTFusion",
    "CoAlignFusion",
    "build_fusion_strategy",
    "available_core_methods",
]
