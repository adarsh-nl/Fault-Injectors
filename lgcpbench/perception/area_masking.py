"""
area_masking.py
---------------
Restrict a full BEV feature map to the cells of one area, and account for the
bits that restriction saves.

Paper mapping
    Section III, step 3: "CAVs partition their perception data into different
    areas according to the partitioning rule of the RSU, and then transmit the
    area-specific perception data to the corresponding group leader."
    Section VI-C: "Each complete shared feature is compressed to 2.16Mb."

Design doc derivation D2 -- why the payload formula is exact, not fitted
    The OPV2V Where2comm BEV feature is (C=256, H=48, W=176):

        256 * 48 * 176 = 2,162,688  ~=  2.16 x 10^6 bits

    Exactly one bit per feature element. So the paper's compression is
    1 bit/element and the area-restricted payload is

        bits(area, cav) = C * |cells(area)| * bits_per_element

    This is the mechanism behind the reported 44x reduction, and it means our
    communication accounting is derived from the paper's own number rather
    than tuned to reproduce it.

Why slicing, not boolean indexing
    ``AreaGrid`` assigns feature cells by centre, so an area's cells always
    form a contiguous rectangle (asserted in test_roi). Extraction is
    therefore a VIEW. On a 7-CAV frame with a few dozen occupied areas, a
    boolean-index copy would allocate once per (area, CAV) pair every frame;
    a view allocates nothing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from ..roi.grid import AreaGrid

# Design doc D2: the paper's "2.16Mb per complete shared feature" implies
# exactly one bit per feature-map element.
DEFAULT_BITS_PER_ELEMENT: float = 1.0


class AreaFeatureMasker:
    """Extract and account for area-restricted BEV features.

    Purpose
        The single place that knows how an area maps onto feature cells. Both
        the perception path (what the leader fuses) and the communication
        path (how many bits that costs) go through here, so they can never
        disagree -- a divergence that would silently invalidate the paper's
        headline reduction claim.

    Inputs
    ------
    grid            the AreaGrid partition.
    feature_hw      (H, W) of the BEV feature map.
    bits_per_element compression, in bits per feature element (default 1.0
                    per derivation D2).

    Example
    -------
    >>> from lgcpbench.roi import AreaGrid
    >>> g = AreaGrid((-140.8, -38.4, -3.0, 140.8, 38.4, 1.0))
    >>> m = AreaFeatureMasker(g, feature_hw=(48, 176))
    >>> feat = torch.zeros(256, 48, 176)
    >>> tuple(m.extract(feat, area_id=200).shape)
    (256, 4, 7)
    >>> m.payload_bits(area_id=200, channels=256)     # C * 28 cells
    7168
    >>> m.full_map_bits(channels=256)                 # the paper's 2.16 Mb
    2162688
    >>> round(m.reduction_ratio([200], channels=256))
    302
    """

    def __init__(
        self,
        grid: AreaGrid,
        feature_hw: Tuple[int, int],
        bits_per_element: float = DEFAULT_BITS_PER_ELEMENT,
    ) -> None:
        if bits_per_element <= 0:
            raise ValueError(f"bits_per_element must be > 0, got {bits_per_element}")
        self.grid = grid
        self.feature_hw = (int(feature_hw[0]), int(feature_hw[1]))
        self.bits_per_element = float(bits_per_element)

        # Per-area slice bounds and cell counts, computed once. Shared with
        # the confidence estimator through the grid's cache so the two
        # planes cannot disagree about where an area lives.
        self._bounds = grid.all_cell_bounds(self.feature_hw)
        self._counts: np.ndarray = grid.cell_counts(self.feature_hw)

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    def bounds(self, area_id: int) -> Tuple[int, int, int, int]:
        """(row0, row1, col0, col1) feature-cell slice of an area."""
        return self._bounds[area_id]

    def cell_count(self, area_id: int) -> int:
        """Number of feature cells owned by an area."""
        return int(self._counts[area_id])

    def area_shape(self, area_id: int) -> Tuple[int, int]:
        """(h_a, w_a) of the extracted sub-map."""
        r0, r1, c0, c1 = self._bounds[area_id]
        return (r1 - r0, c1 - c0)

    def is_empty(self, area_id: int) -> bool:
        """True if the area owns no feature cells (area smaller than a cell)."""
        return self.cell_count(area_id) == 0

    # ------------------------------------------------------------------ #
    # extraction
    # ------------------------------------------------------------------ #

    def extract(self, features: torch.Tensor, area_id: int) -> torch.Tensor:
        """Restrict a BEV feature map to one area.

        Inputs  features (..., H, W); area_id.
        Outputs (..., h_a, w_a) -- a VIEW into ``features``, not a copy.

        Raises if ``features`` does not match the configured feature_hw,
        which catches a backbone/grid mismatch at the first frame rather than
        as silently wrong boxes later.
        """
        if tuple(features.shape[-2:]) != self.feature_hw:
            raise ValueError(
                f"feature map is {tuple(features.shape[-2:])} but masker was built "
                f"for {self.feature_hw}; grid and backbone disagree"
            )
        r0, r1, c0, c1 = self._bounds[area_id]
        return features[..., r0:r1, c0:c1]

    def scatter_into(
        self, canvas: torch.Tensor, area_feature: torch.Tensor, area_id: int
    ) -> torch.Tensor:
        """Write an area sub-map back into a full-size canvas, in place.

        Used when assembling a global feature view for visualisation or for
        baselines that fuse on the full map.

        Inputs  canvas (..., H, W); area_feature (..., h_a, w_a).
        Outputs the same ``canvas`` object.
        """
        r0, r1, c0, c1 = self._bounds[area_id]
        expected = (r1 - r0, c1 - c0)
        if tuple(area_feature.shape[-2:]) != expected:
            raise ValueError(
                f"area {area_id} expects {expected}, got {tuple(area_feature.shape[-2:])}"
            )
        canvas[..., r0:r1, c0:c1] = area_feature
        return canvas

    # ------------------------------------------------------------------ #
    # communication accounting (design doc D2)
    # ------------------------------------------------------------------ #

    def payload_bits(self, area_id: int, channels: int) -> int:
        """Bits one CAV transmits to share one area's features.

        ``bits = channels * cells(area) * bits_per_element``  (D2).
        """
        return int(round(channels * self.cell_count(area_id) * self.bits_per_element))

    def full_map_bits(self, channels: int) -> int:
        """Bits to share a COMPLETE feature map -- the paper's 2.16 Mb.

        This is the denominator of the reduction ratio, and what the
        vehicle-based and edge-assisted baselines transmit per link.
        """
        h, w = self.feature_hw
        return int(round(channels * h * w * self.bits_per_element))

    def reduction_ratio(self, area_ids, channels: int) -> float:
        """Full-map bits divided by the bits actually sent for these areas.

        Inputs  area_ids : iterable of area ids transmitted by one CAV.
        Outputs ratio > 1 means LGCP sent less than a full map.

        Returns ``inf`` when nothing is transmitted (a CAV in no group),
        which is a real and meaningful outcome, not an error.
        """
        sent = sum(self.payload_bits(a, channels) for a in area_ids)
        if sent == 0:
            return float("inf")
        return self.full_map_bits(channels) / sent
