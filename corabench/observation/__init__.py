"""CoRA's observation-location registry.

The tap mechanism itself (``emit``, ``TapSet``, ``StatsTap``, ...) is
paper-agnostic and lives in ``cpbench.observation``. Only the registry of
CoRA's 52 named locations is here, because each paper names its own.
"""

from .locations import LOCATIONS, Location, all_locations, validate_location

__all__ = ["Location", "LOCATIONS", "all_locations", "validate_location"]
