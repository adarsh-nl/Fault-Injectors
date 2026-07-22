"""Where2comm's observation-point registry.

The tap mechanism itself -- ``emit``, ``TapSet``, ``StatsTap``,
``TensorDumpTap``, ``DriftTap`` -- lives in ``cpbench.observation`` and is
imported from there. Only the *names* are paper-specific, and they live in
:mod:`w2cbench.observation.locations`.
"""

from .locations import LOCATIONS, Location, all_locations, validate_location

__all__ = ["LOCATIONS", "Location", "all_locations", "validate_location"]
