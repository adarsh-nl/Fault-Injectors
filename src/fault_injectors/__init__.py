"""
fault_injectors
---------------
Fault injection for RGB-LiDAR multi-modal 3D object detection on Griffin.

Each failure mode lives in its own module and exposes a small, composable API.
A clean (perfectly synchronised) sample is the pair:

    X = (image, points)

where `image` is an (H, W, 3) RGB array and `points` is an (N, C) LiDAR array.
A fault injector takes a clean sample (or a stream of samples) and returns a
corrupted sample following the formal definitions in the project's fault spec.

Modules
-------
missing_modality : sensor dropout (Bernoulli-gated zeroing / emptying)
"""

from .missing_modality import (
    MissingModalityInjector,
    drop_image,
    drop_points,
    bernoulli_mask,
)

__all__ = [
    'MissingModalityInjector',
    'drop_image',
    'drop_points',
    'bernoulli_mask',
]
