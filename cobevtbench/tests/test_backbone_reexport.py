"""
The image backbone moved to ``cpbench.models.image``; this pins the re-export.

A move like that is additive or it is a breaking change, with no middle ground
for a module other code imports. What makes it additive is that every existing
path resolves to the *same class object* -- not an equivalent copy, which could
drift, but the same one.

This assertion lives here rather than beside the moved module because
``cpbench`` must not import a paper package, even in a test, and the layering
suite enforces that.
"""

from __future__ import annotations

from cpbench.models.image import ResnetEncoder as Canonical


def test_every_cobevtbench_path_resolves_to_the_canonical_class() -> None:
    from cobevtbench.models import ResnetEncoder as ViaPackage
    from cobevtbench.models.backbone import ResnetEncoder as ViaModule
    assert ViaModule is Canonical
    assert ViaPackage is Canonical


def test_the_normalisation_constants_came_along() -> None:
    """cobevt_camera reads these; leaving them behind would be an ImportError
    at model construction rather than at import."""
    from cobevtbench.models.backbone import IMAGENET_MEAN, IMAGENET_STD
    assert len(IMAGENET_MEAN) == 3 and len(IMAGENET_STD) == 3
