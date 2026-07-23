"""
Documentation is executable here: every module's doctests run in CI.

``pytest --doctest-modules`` covers this when invoked with the flag; this
module makes the sweep unconditional, so a doctest that drifts from the code
fails the plain ``pytest v2xvitbench`` invocation the README advertises.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import v2xvitbench


def test_every_module_doctest_passes() -> None:
    failures = []
    for info in pkgutil.walk_packages(v2xvitbench.__path__,
                                      prefix="v2xvitbench."):
        if ".tests" in info.name or info.name.endswith(".tests"):
            continue
        module = importlib.import_module(info.name)
        result = doctest.testmod(module, verbose=False)
        if result.failed:
            failures.append(f"{info.name}: {result.failed} failed")
    assert not failures, "doctest failures:\n  " + "\n  ".join(failures)
