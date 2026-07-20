"""CoRA's cooperative dataset.

BEV geometry, voxelisation, anchors and box decoding are paper-agnostic and
live in ``cpbench.data``.
"""

from .cooperative import CoRADataset, collate_cooperative

__all__ = ["CoRADataset", "collate_cooperative"]
