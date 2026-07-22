"""
Enforce the repository's dependency direction.

    src/  <-  cpbench/  <-  {corabench/, lgcpbench/, cobevtbench/, w2cbench/}

This is the rule the extraction exists to make true, and it is exactly the
kind of thing that decays silently: one convenient import from a paper package
into the core, or from one paper into another, and the core stops being
reusable without anyone noticing until the next paper is added.

A static check over the import graph is cheap and catches it at the commit
that introduces it.
"""

from __future__ import annotations

import ast
import itertools
import pathlib
from typing import Iterator, Set, Tuple

REPO = pathlib.Path(__file__).resolve().parents[2]
PAPER_PACKAGES = ("corabench", "lgcpbench", "cobevtbench", "w2cbench")


def _imports(path: pathlib.Path) -> Iterator[str]:
    """Top-level package of every absolute import in a file."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]


def _python_files(package: str, include_tests: bool = False) -> Iterator[pathlib.Path]:
    root = REPO / package
    if not root.exists():  # pragma: no cover
        return
    for path in sorted(root.rglob("*.py")):
        if not include_tests and "tests" in path.parts:
            continue
        yield path


def _offenders(package: str, forbidden: Tuple[str, ...],
               include_tests: bool = False) -> Set[str]:
    found = set()
    for path in _python_files(package, include_tests):
        for module in _imports(path):
            if module in forbidden:
                found.add(f"{path.relative_to(REPO)} imports {module}")
    return found


def test_cpbench_does_not_import_any_paper_package() -> None:
    """The core must not depend on a paper. Otherwise adding a third paper
    means untangling the second one out of the shared code."""
    offenders = _offenders("cpbench", PAPER_PACKAGES, include_tests=True)
    assert not offenders, "cpbench must not import a paper package:\n  " + "\n  ".join(
        sorted(offenders)
    )


def test_paper_packages_do_not_import_each_other() -> None:
    """The paper packages are siblings. Anything two of them need belongs in
    cpbench instead.

    Written over every ordered pair rather than as a hand-written list, so
    adding a fourth paper to PAPER_PACKAGES is one edit and cannot leave a
    direction unchecked. The pairwise form is what caught nothing yet and is
    meant to keep catching nothing.
    """
    offenders: Set[str] = set()
    for package, sibling in itertools.permutations(PAPER_PACKAGES, 2):
        offenders |= _offenders(package, (sibling,), include_tests=True)
    assert not offenders, "paper packages must not import each other:\n  " + "\n  ".join(
        sorted(offenders)
    )


def test_src_does_not_import_upward() -> None:
    """The fault-injection toolkit is the base layer and stays standalone --
    it is usable on its own, independent of any benchmark here."""
    offenders = _offenders("src", ("cpbench",) + PAPER_PACKAGES, include_tests=True)
    assert not offenders, "src must not import upward:\n  " + "\n  ".join(sorted(offenders))


def test_no_stale_references_to_moved_modules() -> None:
    """Modules extracted into cpbench must not still be addressed through
    their old corabench paths -- those packages no longer exist, so a stale
    reference is a latent ImportError."""
    moved = {
        "corabench.observation.taps", "corabench.observation.recorders",
        "corabench.utils", "corabench.faults", "corabench.logbook",
        "corabench.metrics", "corabench.comms",
        "corabench.data.preprocessing", "corabench.data.postprocessing",
        "corabench.data.synthetic",
        "corabench.models.encoder", "corabench.models.heads",
    }
    offenders = []
    for package in ("cpbench",) + PAPER_PACKAGES:
        for path in _python_files(package, include_tests=True):
            text = path.read_text()
            for name in moved:
                # match imports, not prose mentions in docstrings
                if f"from {name} import" in text or f"import {name}\n" in text:
                    offenders.append(f"{path.relative_to(REPO)} -> {name}")
    assert not offenders, "stale imports of moved modules:\n  " + "\n  ".join(offenders)


# Names that were consolidated into cpbench because the copies were
# effectively identical -- and, in one case, because they were NOT, in a way
# nothing detected. A paper package growing its own again is the regression
# this guards.
CONSOLIDATED = ("DetectionLoss", "sigmoid_focal_loss", "ResnetEncoder",
                "labels_to_array", "ordered_agent_ids", "agent_to_ego_matrix",
                "world_to_ego_matrix")


def test_no_paper_package_redefines_a_consolidated_name() -> None:
    """``DetectionLoss`` lived in two packages and they had DIVERGED: one
    collapsed the class axis, writing every positive anchor into channel 0
    regardless of its label. At ``num_classes: 1`` -- every shipped default --
    the two agree exactly, so nothing failed and the loss still fell, on the
    wrong objective.

    That is the failure mode this test exists for. Two copies of a thing are
    tolerable; two copies that agree on the default path and disagree off it
    are not, because the disagreement surfaces as a model that trains and
    performs slightly worse for reasons nobody can find.

    Re-exports are fine and are what makes each move additive -- only a fresh
    ``class``/``def`` counts.
    """
    offenders = []
    for package in PAPER_PACKAGES:
        for path in _python_files(package):
            text = path.read_text()
            for name in CONSOLIDATED:
                for keyword in ("class", "def"):
                    if f"\n{keyword} {name}(" in text:
                        offenders.append(
                            f"{path.relative_to(REPO)} redefines {name}")
    assert not offenders, (
        "these were consolidated into cpbench; import them instead of "
        "redefining:\n  " + "\n  ".join(sorted(offenders)))


def test_every_package_is_importable() -> None:
    """Catches a package left without an __init__ or with a broken re-export."""
    import importlib

    for name in (
        "cpbench", "cpbench.observation", "cpbench.data", "cpbench.models",
        "cpbench.metrics", "cpbench.logbook", "cpbench.faults", "cpbench.utils",
        "cpbench.comms",
        "corabench", "corabench.models", "corabench.fusion", "corabench.data",
        "corabench.observation", "corabench.evaluation", "corabench.training",
        "lgcpbench", "lgcpbench.roi", "lgcpbench.confidence",
        "lgcpbench.selection", "lgcpbench.network", "lgcpbench.perception",
        "lgcpbench.orchestration", "lgcpbench.metrics", "lgcpbench.data",
        "lgcpbench.observation",
        "cobevtbench", "cobevtbench.observation", "cobevtbench.attention",
        "cobevtbench.fusion", "cobevtbench.models", "cobevtbench.data",
        "cobevtbench.faults", "cobevtbench.training", "cobevtbench.evaluation",
        "cobevtbench.scripts",
        # w2cbench is being built subpackage by subpackage (design doc s13);
        # each entry lands with the step that creates it.
        "w2cbench", "w2cbench.observation", "w2cbench.models",
        "w2cbench.comm", "w2cbench.fusion", "w2cbench.data",
        "w2cbench.training", "w2cbench.evaluation", "w2cbench.faults", "w2cbench.scripts", "cpbench.training",
    ):
        importlib.import_module(name)
