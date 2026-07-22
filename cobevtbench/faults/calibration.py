"""
calibration.py
--------------
Camera miscalibration: perturb intrinsics and extrinsics.

**This now lives in** :mod:`src.fault_injectors.calibration`. It moved when
``w2cbench`` became the second camera paper to need it -- which is the trigger
this module's own docstring named while it still lived here. Camera
calibration error is a physical sensor fault, and every other physical sensor
fault in this repository lives in ``src.fault_injectors``.

This module re-exports, so every existing import path keeps working.
"""

from __future__ import annotations

from src.fault_injectors.calibration import (CalibrationErrorInjector,
                                             rotation_from_axis_angle)

__all__ = ["CalibrationErrorInjector", "rotation_from_axis_angle"]
