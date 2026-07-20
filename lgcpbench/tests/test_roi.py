"""
Tests for lgcpbench.roi -- area partitioning (paper contribution C1).

The properties asserted here are the ones the rest of LGCP relies on:
    * the partition covers the RoI exactly and does not overlap (paper: "non-
      overlapping areas");
    * the induced feature-cell partition is strict, so payload accounting
      (design doc D2) cannot double-count boundary cells;
    * occupancy restricts N, which drives Algorithm 2's O(N^2) cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from lgcpbench.roi import (
    AllAreasOccupancy,
    Area,
    AreaGrid,
    BoxOccupancy,
    available_occupancy_sources,
    make_occupancy_estimator,
)

# OPV2V / V2XSet lidar range used throughout the paper's experiments.
OPV2V_RANGE = (-140.8, -38.4, -3.0, 140.8, 38.4, 1.0)
OPV2V_FEATURE_HW = (48, 176)


@pytest.fixture
def grid() -> AreaGrid:
    return AreaGrid(point_range=OPV2V_RANGE)


@pytest.fixture
def small_grid() -> AreaGrid:
    """4 x 4 areas of exactly 10 x 6 m, no remainder."""
    return AreaGrid(point_range=(-20.0, -12.0, -3.0, 20.0, 12.0, 1.0))


# --------------------------------------------------------------------- #
# partition geometry
# --------------------------------------------------------------------- #


def test_grid_dimensions_match_paper_setting(grid: AreaGrid) -> None:
    """280 m x 80 m RoI, 10 m x 6 m areas (paper section VI-C)."""
    # 281.6 / 10 -> 29 columns (28 full + remainder), 76.8 / 6 -> 13 rows.
    assert (grid.n_rows, grid.n_cols) == (13, 29)
    assert len(grid) == 13 * 29


def test_partition_covers_roi_exactly(grid: AreaGrid) -> None:
    """Areas tile the RoI with no gap and no overhang."""
    x_min, y_min, _, x_max, y_max, _ = grid.point_range
    roi_area = (x_max - x_min) * (y_max - y_min)
    assert sum(a.area_m2 for a in grid) == pytest.approx(roi_area)


def test_partition_is_non_overlapping(small_grid: AreaGrid) -> None:
    """No two areas share interior area (the paper's core structural claim)."""
    areas = list(small_grid)
    for i, a in enumerate(areas):
        for b in areas[i + 1 :]:
            overlap_x = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
            overlap_y = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
            assert max(0.0, overlap_x) * max(0.0, overlap_y) == 0.0


def test_nominal_area_size_is_10x6(small_grid: AreaGrid) -> None:
    for a in small_grid:
        assert a.size_m == pytest.approx((10.0, 6.0))


def test_remainder_row_and_column_are_narrower_not_dropped(grid: AreaGrid) -> None:
    """ceil-based partition keeps the remainder strip instead of discarding it."""
    last_col = grid[grid.n_cols - 1]
    last_row = grid[(grid.n_rows - 1) * grid.n_cols]
    assert 0 < last_col.size_m[0] < 10.0
    assert 0 < last_row.size_m[1] < 6.0


def test_area_ids_are_row_major_and_dense(grid: AreaGrid) -> None:
    for i, a in enumerate(grid):
        assert a.id == i
        assert a.id == a.row * grid.n_cols + a.col


# --------------------------------------------------------------------- #
# point -> area lookup
# --------------------------------------------------------------------- #


def test_area_of_point_agrees_with_contains(small_grid: AreaGrid) -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform([-20.0, -12.0], [20.0, 12.0], size=(200, 2))
    for x, y in pts:
        aid = small_grid.area_of_point(float(x), float(y))
        assert aid is not None
        assert small_grid[aid].contains(float(x), float(y))


def test_area_of_point_outside_roi_is_none(small_grid: AreaGrid) -> None:
    assert small_grid.area_of_point(1e3, 0.0) is None
    assert small_grid.area_of_point(0.0, -1e3) is None


def test_far_edge_is_clamped_into_final_area(small_grid: AreaGrid) -> None:
    """The RoI boundary itself must belong to some area, not fall through."""
    aid = small_grid.area_of_point(20.0, 12.0)
    assert aid == len(small_grid) - 1


def test_area_of_points_matches_scalar_lookup(small_grid: AreaGrid) -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform([-25.0, -15.0], [25.0, 15.0], size=(300, 2))
    vec = small_grid.area_of_points(pts)
    for (x, y), got in zip(pts, vec):
        expected = small_grid.area_of_point(float(x), float(y))
        assert got == (-1 if expected is None else expected)


def test_area_of_points_empty_input(small_grid: AreaGrid) -> None:
    assert small_grid.area_of_points(np.empty((0, 2))).shape == (0,)


# --------------------------------------------------------------------- #
# BEV feature-cell partition (design doc D2 depends on this being strict)
# --------------------------------------------------------------------- #


def test_cell_partition_is_strict(grid: AreaGrid) -> None:
    """Every feature cell belongs to exactly one area; counts sum to H*W."""
    h, w = OPV2V_FEATURE_HW
    counts = grid.cell_counts(OPV2V_FEATURE_HW)
    assert counts.shape == (len(grid),)
    assert int(counts.sum()) == h * w


def test_cell_area_ids_in_range(grid: AreaGrid) -> None:
    ids = grid.cell_area_ids(OPV2V_FEATURE_HW)
    assert ids.shape == OPV2V_FEATURE_HW
    assert ids.min() >= 0 and ids.max() < len(grid)


def test_cell_area_ids_cached_and_read_only(grid: AreaGrid) -> None:
    a = grid.cell_area_ids(OPV2V_FEATURE_HW)
    b = grid.cell_area_ids(OPV2V_FEATURE_HW)
    assert a is b, "repeated lookups must hit the cache"
    with pytest.raises(ValueError):
        a[0, 0] = 999


def test_cell_mask_matches_cell_area_ids(grid: AreaGrid) -> None:
    ids = grid.cell_area_ids(OPV2V_FEATURE_HW)
    for area_id in (0, 100, len(grid) - 1):
        assert np.array_equal(grid.cell_mask(area_id, OPV2V_FEATURE_HW), ids == area_id)


def test_cell_bounds_is_a_contiguous_rectangle(grid: AreaGrid) -> None:
    """Areas are axis-aligned, so their cells slice as a view, not a copy.

    This is what lets area-restricted transmission avoid a boolean-index copy
    on the hot path.
    """
    for area_id in (0, 57, 200, len(grid) - 1):
        mask = grid.cell_mask(area_id, OPV2V_FEATURE_HW)
        r0, r1, c0, c1 = grid.cell_bounds(area_id, OPV2V_FEATURE_HW)
        rebuilt = np.zeros_like(mask)
        rebuilt[r0:r1, c0:c1] = True
        assert np.array_equal(mask, rebuilt)


def test_area_cell_count_matches_paper_payload_scale(grid: AreaGrid) -> None:
    """Design doc D2: area payload is C * (cells in area), and one area is a
    ~1% slice of the full map -- the mechanism behind the paper's 44x claim.

    A 10 x 6 m area spans 10/1.6 = 6.25 by 6/1.6 = 3.75 feature cells. Since
    cells are assigned whole, interior areas take 6 or 7 columns and 3 or 4
    rows, and the mix tiles the map exactly. The bound that matters is the
    WORST case (7 x 4 = 28 cells), because that sets peak payload.
    """
    counts = grid.cell_counts(OPV2V_FEATURE_HW)
    interior = counts[counts > 0]

    stride_x = (140.8 - -140.8) / OPV2V_FEATURE_HW[1]
    stride_y = (38.4 - -38.4) / OPV2V_FEATURE_HW[0]
    assert (stride_x, stride_y) == pytest.approx((1.6, 1.6))

    # cells per area are whole-cell roundings of 6.25 x 3.75
    assert set(np.unique(interior).tolist()) <= {c * r for c in (6, 7) for r in (3, 4)} | {
        c * r for c in range(1, 8) for r in range(1, 5)
    }
    assert int(interior.max()) == 7 * 4
    # and the average lands on the exact continuous value
    assert float(interior.mean()) == pytest.approx(6.25 * 3.75, rel=0.15)

    channels = 256
    full_map_bits = channels * OPV2V_FEATURE_HW[0] * OPV2V_FEATURE_HW[1]
    assert full_map_bits == 2_162_688  # paper: "compressed to 2.16Mb"
    # peak area payload is well under 1% of a full feature map
    assert channels * int(interior.max()) < full_map_bits / 100


def test_cell_partition_strict_for_other_shapes(small_grid: AreaGrid) -> None:
    for hw in [(8, 16), (12, 20), (1, 1)]:
        assert int(small_grid.cell_counts(hw).sum()) == hw[0] * hw[1]


# --------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"point_range": (0.0, 0.0, 0.0, 1.0, 1.0)},          # wrong length
        {"point_range": (0.0, 0.0, 0.0, 0.0, 1.0, 1.0)},     # empty in x
        {"point_range": OPV2V_RANGE, "area_size_m": (0.0, 6.0)},   # zero size
        {"point_range": OPV2V_RANGE, "area_size_m": (10.0,)},      # wrong length
    ],
)
def test_invalid_construction_raises(kwargs) -> None:
    with pytest.raises(ValueError):
        AreaGrid(**kwargs)


def test_cell_mask_rejects_bad_area_id(small_grid: AreaGrid) -> None:
    with pytest.raises(IndexError):
        small_grid.cell_mask(len(small_grid), (8, 16))


# --------------------------------------------------------------------- #
# occupancy (assumption B8)
# --------------------------------------------------------------------- #


def test_all_areas_occupancy(small_grid: AreaGrid) -> None:
    occ = AllAreasOccupancy()(small_grid)
    assert occ.shape == (len(small_grid),) and occ.all()


def test_box_occupancy_marks_only_occupied_areas(small_grid: AreaGrid) -> None:
    boxes = np.array([[0.0, 0.0], [-15.0, -9.0]])
    occ = BoxOccupancy(include_cavs=False)(small_grid, boxes=boxes)
    expected = {small_grid.area_of_point(0.0, 0.0), small_grid.area_of_point(-15.0, -9.0)}
    assert set(np.flatnonzero(occ).tolist()) == expected


def test_box_occupancy_counts_cavs_as_vehicles(small_grid: AreaGrid) -> None:
    cavs = np.array([[5.0, 3.0]])
    with_cavs = BoxOccupancy(include_cavs=True)(small_grid, cav_positions=cavs)
    without = BoxOccupancy(include_cavs=False)(small_grid, cav_positions=cavs)
    assert with_cavs.sum() == 1
    assert without.sum() == 0


def test_box_occupancy_ignores_points_outside_roi(small_grid: AreaGrid) -> None:
    occ = BoxOccupancy(include_cavs=False)(small_grid, boxes=np.array([[1e4, 1e4]]))
    assert not occ.any()


def test_box_occupancy_empty_and_none_inputs(small_grid: AreaGrid) -> None:
    assert not BoxOccupancy()(small_grid).any()
    assert not BoxOccupancy()(small_grid, boxes=np.empty((0, 2))).any()


def test_dilation_expands_the_occupied_set(small_grid: AreaGrid) -> None:
    boxes = np.array([[0.0, 0.0]])
    tight = BoxOccupancy(include_cavs=False, dilate_rings=0)(small_grid, boxes=boxes)
    loose = BoxOccupancy(include_cavs=False, dilate_rings=1)(small_grid, boxes=boxes)
    assert tight.sum() == 1
    # interior cell -> 3x3 Chebyshev neighbourhood
    assert loose.sum() == 9
    assert np.all(loose[tight])


def test_occupancy_reduces_n_on_a_realistic_frame(grid: AreaGrid) -> None:
    """The point of B8: N drops from 377 to a few dozen, which is what keeps
    Algorithm 2's O(N^2) tractable."""
    rng = np.random.default_rng(7)
    boxes = rng.uniform([-140.0, -38.0], [140.0, 38.0], size=(30, 2))
    occ = BoxOccupancy(include_cavs=False)(grid, boxes=boxes)
    assert 0 < occ.sum() <= 30
    assert occ.sum() < len(grid) / 4


def test_occupancy_registry(small_grid: AreaGrid) -> None:
    assert set(available_occupancy_sources()) == {"all", "gt", "prev_global_view"}
    assert isinstance(make_occupancy_estimator("gt", dilate_rings=1), BoxOccupancy)
    assert isinstance(make_occupancy_estimator("all"), AllAreasOccupancy)
    with pytest.raises(KeyError):
        make_occupancy_estimator("nope")


def test_box_occupancy_rejects_malformed_input(small_grid: AreaGrid) -> None:
    with pytest.raises(ValueError):
        BoxOccupancy()(small_grid, boxes=np.zeros((4,)))


def test_estimators_satisfy_the_protocol(small_grid: AreaGrid) -> None:
    from lgcpbench.roi import OccupancyEstimator

    for est in (AllAreasOccupancy(), BoxOccupancy()):
        assert isinstance(est, OccupancyEstimator)
