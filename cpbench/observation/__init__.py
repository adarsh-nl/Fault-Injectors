"""Read-only observation: the measurement plane.

Taps never modify the forward pass. ``emit`` hands observers a DETACHED
tensor, and with ``taps=None`` the hook costs one ``is None`` check.

Each paper package owns its own location registry (``<paper>/observation/
locations.py``); only the mechanism lives here.
"""

from .recorders import DriftTap, StatsTap, TensorDumpTap
from .taps import NullTap, TapProtocol, TapRecord, TapSet, emit

__all__ = ["TapRecord", "TapProtocol", "NullTap", "TapSet", "emit",
           "StatsTap", "TensorDumpTap", "DriftTap"]
