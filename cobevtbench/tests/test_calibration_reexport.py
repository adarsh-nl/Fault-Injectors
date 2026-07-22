"""
``CalibrationErrorInjector`` moved to ``src.fault_injectors``; this pins the
re-export.

The move happened when ``w2cbench`` became the second camera paper to need it,
which is the trigger the injector's own docstring named while it still lived
here. Camera calibration drift is a physical sensor fault, and every other
physical sensor fault in this repository lives in ``src.fault_injectors``.

The assertion lives in this package because a paper package must not import a
sibling -- ``w2cbench`` cannot check ``cobevtbench``'s import paths, and the
layering suite enforces that.
"""

from __future__ import annotations

from src.fault_injectors import CalibrationErrorInjector as Canonical
from src.fault_injectors.calibration import rotation_from_axis_angle as canonical_rodrigues


def test_the_old_paths_resolve_to_the_canonical_class() -> None:
    from cobevtbench.faults.calibration import (CalibrationErrorInjector,
                                                rotation_from_axis_angle)
    assert CalibrationErrorInjector is Canonical
    assert rotation_from_axis_angle is canonical_rodrigues


def test_the_registry_still_builds_it() -> None:
    """cobevtbench's own fault registry names this injector; the move must not
    have quietly turned its camera-calibration condition into a no-op."""
    from cobevtbench.faults.registry import build_bridge
    bridge = build_bridge({"calibration": {"sigma_focal_px": 8.0}})
    assert not bridge.is_clean
    assert isinstance(bridge.pipeline.sample_stages[0], Canonical)
