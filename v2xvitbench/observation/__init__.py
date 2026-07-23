"""Tap-location registry for v2xvitbench.

The tap *mechanism* (``emit``, ``TapSet``, recorders) lives in
``cpbench.observation``; what is paper-specific is the set of names.
"""

from v2xvitbench.observation.locations import (LOCATIONS, Location,
                                               all_locations,
                                               validate_location)

__all__ = ["LOCATIONS", "Location", "all_locations", "validate_location"]
