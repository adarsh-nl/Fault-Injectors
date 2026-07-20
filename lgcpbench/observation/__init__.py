"""
lgcpbench.observation
=====================
Measurement plane for LGCP.

Tensor observation is inherited unchanged from ``corabench.observation``
(``emit``, ``TapSet``, ``StatsTap``, ``TensorDumpTap``). This package adds
only what LGCP needs beyond it: recording of CONTROL-plane decisions, which
are not tensors and which corabench's recorders ignore by contract.

The canonical location registry (``locations.py``) arrives with step 7 of the
implementation plan.

Example
-------
>>> from cpbench.observation.taps import TapSet, emit
>>> from lgcpbench.observation import ControlPlaneTap
>>> tap = ControlPlaneTap(retain=True)
>>> emit(TapSet([tap]), {"a": 1.0}, module="M", location="lgcp/selection/loads")
>>> tap.locations()
['lgcp/selection/loads']
"""

from .recorders import ControlPlaneTap, ControlRecord

__all__ = ["ControlPlaneTap", "ControlRecord"]
