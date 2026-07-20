"""CoRA's model.

The generic PointPillars encoder and detection heads are paper-agnostic and
live in ``cpbench.models``.
"""

from .cora import CoRAModel

__all__ = ["CoRAModel"]
