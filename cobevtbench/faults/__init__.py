"""
CoBEVT-specific fault surface.

Two injectors live here, both **sample stages** -- they mutate the whole
CooperativeSample before any tensor exists, so no model code is fault-aware
and a measured degradation stays attributable to the fault.

They are here rather than in ``src.fault_injectors`` because each has exactly
one consumer today. A second camera paper is the trigger to promote them:
premature generality is how a toolkit accumulates abstractions nobody needs.

Contents
--------
calibration     CalibrationErrorInjector -- perturb K and T_cam_to_agent, the
                matrices SinBEVT lifts by. The fault surface specific to how
                CoBEVT works, with no equivalent in src/.
camera_dropout  CameraDropoutInjector -- blind chosen cameras on chosen
                agents; reproduces the paper's own section 7.4 experiment.
registry        build_bridge(config) -- compose a DataFaultBridge that also
                carries the sample stages the bridge cannot configure itself.
"""

from .calibration import CalibrationErrorInjector, rotation_from_axis_angle
from .camera_dropout import CameraDropoutInjector
from .registry import build_bridge

__all__ = ["CalibrationErrorInjector", "CameraDropoutInjector", "build_bridge",
           "rotation_from_axis_angle"]
