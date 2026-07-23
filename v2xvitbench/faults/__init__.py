"""Fault planes for v2xvitbench.

Plane 1 (physical): src injectors on raw data, via ``build_bridge``.
Plane 2 (metadata): this package's injectors on the collated batch's V2X
metadata, via ``build_metadata_bridge``.
"""

from v2xvitbench.faults.injectors import (AgentTypeFlipInjector,
                                          CorrectionMatrixInjector,
                                          DelayEncodingInjector,
                                          MetadataInjector,
                                          PriorNoiseInjector,
                                          make_metadata_injector)
from v2xvitbench.faults.metadata import MetadataFaultBridge
from v2xvitbench.faults.registry import (build_bridge, build_bridges,
                                         build_metadata_bridge)

__all__ = [
    "AgentTypeFlipInjector", "CorrectionMatrixInjector",
    "DelayEncodingInjector", "MetadataFaultBridge", "MetadataInjector",
    "PriorNoiseInjector", "build_bridge", "build_bridges",
    "build_metadata_bridge", "make_metadata_injector",
]
