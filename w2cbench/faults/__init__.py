"""Both fault planes, and the wiring that builds them from one config.

    registry.py   config -> physical bridge (cpbench.faults) + protocol bridge
    protocol.py   ProtocolFaultBridge -- the message boundary
    injectors.py  RequestLoss, ConfidenceReport, BandwidthCap

The physical plane is not reimplemented here: ``src.pipeline.FaultPipeline``
remains the single place raw data is corrupted. The protocol plane exists
because Where2comm's messages carry control payloads -- request maps, reported
confidence -- that no sensor-level injector can reach.
"""

from .injectors import (BandwidthCapInjector, ConfidenceReportInjector,
                        ProtocolInjector, RequestLossInjector,
                        available_protocol_injectors, make_protocol_injector)
from .protocol import STAGES, ProtocolFaultBridge
from .registry import build_bridge, build_bridges, build_protocol_bridge

__all__ = ["build_bridge", "build_protocol_bridge", "build_bridges",
           "ProtocolFaultBridge", "STAGES", "ProtocolInjector",
           "RequestLossInjector", "ConfidenceReportInjector",
           "BandwidthCapInjector", "make_protocol_injector",
           "available_protocol_injectors"]
