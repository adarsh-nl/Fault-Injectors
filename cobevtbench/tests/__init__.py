"""Test package marker.

Present so this package's ``conftest.py`` and test modules get a qualified
module name. Without it, pytest imports them as bare ``conftest`` /
``test_<name>`` and collides with any other test directory in the repository
using the same basename -- which ``lgcpbench/tests`` does. ``corabench/tests``
carries the same marker for the same reason.
"""
