"""
Observation plane -- the canonical registry of CoBEVT's named tensors.

The tap *mechanism* (emit, TapSet, StatsTap, TensorDumpTap, DriftTap) lives in
`cpbench.observation` and is imported, never re-implemented. What is
paper-specific is the *set* of named observation points, which is why every
paper package owns its own registry.

    from cpbench.observation import emit, StatsTap, TapSet   # mechanism
    from cobevtbench.observation import validate_location    # vocabulary
"""

from .locations import LOCATIONS, Location, all_locations, validate_location

__all__ = ["LOCATIONS", "Location", "all_locations", "validate_location"]
