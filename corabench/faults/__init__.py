"""Physical, upstream fault application (the ONLY corruption path)."""

from .bridge import DataFaultBridge, FaultRecord

__all__ = ["DataFaultBridge", "FaultRecord"]
