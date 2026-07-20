"""
Tests for lgcpbench.confidence -- paper Eq. 1, 2, 3.

The two properties that matter most downstream:
    * Eq. 2's closed-form gain equals the naive difference form, because
      Algorithm 1 uses the closed form on its hot path;
    * gains are non-increasing when candidates are visited in descending
      confidence, because Algorithm 1's early termination depends on it.
If either broke, grouping would silently admit or reject the wrong CAVs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpbench.observation.recorders import StatsTap
from cpbench.observation.taps import TapSet
from lgcpbench.confidence import (
    AreaConfidenceEstimator,
    AreaConfidenceMatrix,
    MaxPooling,
    MeanPooling,
    NoisyOrCombiner,
    TopKMeanPooling,
    available_poolings,
    global_accuracy_proxy,
    make_pooling,
)
from lgcpbench.roi import AreaGrid

SMALL_RANGE = (-20.0, -12.0, -3.0, 20.0, 12.0, 1.0)
SMALL_HW = (8, 16)


@pytest.fixture
def grid() -> AreaGrid:
    return AreaGrid(point_range=SMALL_RANGE)


@pytest.fixture
def estimator(grid: AreaGrid) -> AreaConfidenceEstimator:
    return AreaConfidenceEstimator(grid, feature_hw=SMALL_HW)


# --------------------------------------------------------------------- #
# Eq. 2 -- noisy-OR
# --------------------------------------------------------------------- #


def test_combine_matches_the_paper_formula() -> None:
    c = NoisyOrCombiner()
    vals = [0.2, 0.5, 0.7]
    expected = 1.0 - np.prod([1 - v for v in vals])
    assert c.combine(vals) == pytest.approx(expected)


def test_combine_is_order_invariant() -> None:
    """The product is commutative, so group confidence cannot depend on the
    order members were admitted -- only WHICH members were admitted."""
    c = NoisyOrCombiner()
    vals = [0.1, 0.44, 0.9, 0.3]
    rng = np.random.default_rng(0)
    base = c.combine(vals)
    for _ in range(10):
        assert c.combine(rng.permutation(vals)) == pytest.approx(base)


def test_combine_is_monotone_non_decreasing_in_group_size() -> None:
    """Adding a CAV can never reduce area confidence (Eq. 2)."""
    c = NoisyOrCombiner()
    rng = np.random.default_rng(1)
    vals = rng.uniform(0, 1, size=8)
    running = [c.combine(vals[:k]) for k in range(len(vals) + 1)]
    assert all(b >= a - 1e-12 for a, b in zip(running, running[1:]))


def test_combine_stays_in_unit_interval() -> None:
    c = NoisyOrCombiner()
    rng = np.random.default_rng(2)
    for _ in range(50):
        v = rng.uniform(0, 1, size=rng.integers(1, 10))
        assert 0.0 <= c.combine(v) <= 1.0


def test_empty_group_has_zero_confidence() -> None:
    """An area with no assigned group has no expected perception quality --
    which is what makes an orphaned area visible in the metrics."""
    assert NoisyOrCombiner().combine([]) == 0.0


def test_certain_member_saturates_the_group() -> None:
    assert NoisyOrCombiner().combine([1.0, 0.3]) == pytest.approx(1.0)


def test_zero_confidence_member_contributes_nothing() -> None:
    c = NoisyOrCombiner()
    assert c.combine([0.6, 0.0]) == pytest.approx(c.combine([0.6]))


# --------------------------------------------------------------------- #
# Eq. 8 -- the closed-form gain that Algorithm 1 relies on
# --------------------------------------------------------------------- #


def test_closed_form_gain_equals_the_difference_form() -> None:
    """gain(S, v) == combine(S + [v]) - combine(S), derived as (1-F(S))*f_v.

    Algorithm 1 evaluates Eq. 8 with the closed form on every candidate, so
    if this drifted, grouping would admit the wrong CAVs everywhere.
    """
    c = NoisyOrCombiner()
    rng = np.random.default_rng(3)
    for _ in range(200):
        group = rng.uniform(0, 1, size=rng.integers(0, 6))
        candidate = float(rng.uniform(0, 1))
        current = c.combine(group)
        naive = c.combine(list(group) + [candidate]) - current
        assert c.gain(current, candidate) == pytest.approx(naive, abs=1e-12)


def test_update_equals_recombining_from_scratch() -> None:
    c = NoisyOrCombiner()
    rng = np.random.default_rng(4)
    vals = rng.uniform(0, 1, size=6)
    incremental = 0.0
    for v in vals:
        incremental = c.update(incremental, v)
    assert incremental == pytest.approx(c.combine(vals))


def test_gains_are_non_increasing_in_descending_order() -> None:
    """Diminishing returns, exactly (not just informally as in the paper).

    F(S) is non-decreasing so (1-F(S)) is non-increasing; visiting candidates
    in descending f makes f non-increasing too. Hence the gain sequence is
    non-increasing, and Algorithm 1 may STOP at the first candidate failing
    Eq. 8 instead of scanning the rest.
    """
    c = NoisyOrCombiner()
    rng = np.random.default_rng(5)
    for _ in range(50):
        vals = np.sort(rng.uniform(0, 1, size=8))[::-1]
        current, gains = 0.0, []
        for v in vals:
            gains.append(c.gain(current, v))
            current = c.update(current, v)
        assert all(b <= a + 1e-12 for a, b in zip(gains, gains[1:]))


def test_gain_batch_matches_scalar_gain() -> None:
    c = NoisyOrCombiner()
    cands = np.array([0.1, 0.5, 0.9])
    got = c.gain_batch(0.4, cands)
    assert got == pytest.approx([c.gain(0.4, v) for v in cands])


def test_combine_batch_matches_scalar_combine() -> None:
    c = NoisyOrCombiner()
    vals = np.array([[0.2, 0.5], [0.9, 0.1], [0.0, 0.0]])
    got = c.combine_batch(vals, axis=-1)
    assert got == pytest.approx([c.combine(row) for row in vals])


# --------------------------------------------------------------------- #
# clipping -- the control-plane fault surface reaches this code
# --------------------------------------------------------------------- #


def test_out_of_range_confidence_is_clamped_by_default() -> None:
    """A control-plane injector deliberately falsifies reported confidence.
    An injected 1.7 must not produce a NEGATIVE group confidence, which would
    corrupt the objective rather than merely bias it."""
    c = NoisyOrCombiner(clip=True)
    assert c.combine([1.7, 0.5]) == pytest.approx(1.0)
    assert 0.0 <= c.combine([-3.0, 0.5]) <= 1.0
    assert c.combine([-3.0, 0.5]) == pytest.approx(0.5)


def test_strict_mode_rejects_out_of_range() -> None:
    c = NoisyOrCombiner(clip=False)
    with pytest.raises(ValueError):
        c.combine([1.7, 0.5])
    with pytest.raises(ValueError):
        c.gain(0.5, -0.2)


# --------------------------------------------------------------------- #
# Eq. 3
# --------------------------------------------------------------------- #


def test_global_accuracy_proxy_is_the_mean() -> None:
    assert global_accuracy_proxy([0.8, 0.6, 1.0]) == pytest.approx(0.8)
    assert global_accuracy_proxy([]) == 0.0


# --------------------------------------------------------------------- #
# pooling (assumption B1)
# --------------------------------------------------------------------- #


def test_pooling_shapes() -> None:
    patch = torch.rand(4, 3, 5)
    for pool in (MaxPooling(), MeanPooling(), TopKMeanPooling(k=3)):
        assert tuple(pool(patch).shape) == (4,)


def test_max_pooling_picks_the_strongest_cell() -> None:
    patch = torch.tensor([[[0.1, 0.8], [0.2, 0.3]]])
    assert MaxPooling()(patch).item() == pytest.approx(0.8)


def test_mean_pooling_is_diluted_by_empty_road() -> None:
    """The reason max is the default: one clearly-seen object in an otherwise
    empty area is exactly the CAV a group wants, and mean under-rates it."""
    patch = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    assert MaxPooling()(patch).item() == pytest.approx(1.0)
    assert MeanPooling()(patch).item() == pytest.approx(0.25)


def test_topk_degenerates_to_max_and_mean() -> None:
    patch = torch.rand(3, 4, 4)
    assert torch.allclose(TopKMeanPooling(k=1)(patch), MaxPooling()(patch))
    assert torch.allclose(TopKMeanPooling(k=999)(patch), MeanPooling()(patch))


def test_pooling_outputs_stay_in_unit_interval() -> None:
    """Eq. 2 is only valid on [0, 1]."""
    patch = torch.rand(5, 3, 3)
    for pool in (MaxPooling(), MeanPooling(), TopKMeanPooling(k=2)):
        out = pool(patch)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_pooling_handles_an_area_with_no_cells() -> None:
    """An area smaller than one feature cell owns no evidence; zero is the
    honest answer and contributes nothing to Eq. 2."""
    empty = torch.zeros(3, 0, 4)
    for pool in (MaxPooling(), MeanPooling(), TopKMeanPooling(k=2)):
        out = pool(empty)
        assert tuple(out.shape) == (3,) and float(out.abs().sum()) == 0.0


def test_pooling_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        MaxPooling()(torch.rand(4, 3))


def test_pooling_registry() -> None:
    assert set(available_poolings()) == {"max", "mean", "topk_mean"}
    assert isinstance(make_pooling("topk_mean", k=2), TopKMeanPooling)
    with pytest.raises(KeyError):
        make_pooling("nope")
    with pytest.raises(ValueError):
        TopKMeanPooling(k=0)


# --------------------------------------------------------------------- #
# estimator (Eq. 1)
# --------------------------------------------------------------------- #


def test_estimator_shape_and_ids(estimator: AreaConfidenceEstimator) -> None:
    m = estimator(torch.rand(3, 1, *SMALL_HW), area_ids=[0, 2, 4], agent_ids=("a", "b", "c"))
    assert m.values.shape == (3, 3)
    assert m.area_ids.tolist() == [0, 2, 4]
    assert m.agent_ids == ("a", "b", "c")
    assert m.n_agents == 3 and m.n_areas == 3


def test_estimator_accepts_both_layouts(estimator: AreaConfidenceEstimator) -> None:
    conf = torch.rand(2, *SMALL_HW)
    a = estimator(conf, area_ids=[0, 1])
    b = estimator(conf.unsqueeze(1), area_ids=[0, 1])
    assert np.allclose(a.values, b.values)


def test_estimator_defaults_to_every_area(estimator: AreaConfidenceEstimator,
                                          grid: AreaGrid) -> None:
    m = estimator(torch.rand(2, 1, *SMALL_HW))
    assert m.n_areas == len(grid)


def test_estimator_pools_only_the_areas_cells(estimator: AreaConfidenceEstimator,
                                              grid: AreaGrid) -> None:
    """A hot cell must raise its own area's confidence and no other's."""
    conf = torch.zeros(1, 1, *SMALL_HW)
    conf[0, 0, 0, 0] = 0.9
    hot_area = int(grid.cell_area_ids(SMALL_HW)[0, 0])
    m = estimator(conf)
    assert m.confidence("cav0", hot_area) == pytest.approx(0.9)
    others = [
        m.confidence("cav0", a) for a in m.area_ids.tolist() if a != hot_area
    ]
    assert max(others) == 0.0


def test_estimator_validates_geometry(grid: AreaGrid) -> None:
    est = AreaConfidenceEstimator(grid, feature_hw=SMALL_HW)
    with pytest.raises(ValueError, match="disagree"):
        est(torch.rand(2, 1, 4, 4))
    with pytest.raises(ValueError, match="single confidence channel"):
        est(torch.rand(2, 3, *SMALL_HW))
    with pytest.raises(ValueError):
        est(torch.rand(2))
    with pytest.raises(IndexError):
        est(torch.rand(2, 1, *SMALL_HW), area_ids=[9999])


def test_estimator_emits_a_tap(estimator: AreaConfidenceEstimator) -> None:
    stats = StatsTap()
    estimator(torch.rand(2, 1, *SMALL_HW), area_ids=[0, 1], taps=TapSet([stats]))
    assert any(r.location == "lgcp/confidence/per_area" for r in stats.records)


def test_estimator_output_identical_with_and_without_taps(
    estimator: AreaConfidenceEstimator,
) -> None:
    conf = torch.rand(3, 1, *SMALL_HW)
    clean = estimator(conf, area_ids=[0, 1, 2])
    tapped = estimator(conf, area_ids=[0, 1, 2], taps=TapSet([StatsTap()], strict=True))
    assert np.array_equal(clean.values, tapped.values)


# --------------------------------------------------------------------- #
# AreaConfidenceMatrix -- the control-plane injection target
# --------------------------------------------------------------------- #


def test_matrix_lookups() -> None:
    m = AreaConfidenceMatrix(
        values=np.array([[0.8, 0.1], [0.3, 0.9]]),
        area_ids=np.array([5, 9]),
        agent_ids=("cav0", "cav1"),
    )
    assert m.for_area(9).tolist() == [0.1, 0.9]
    assert m.for_agent("cav1").tolist() == [0.3, 0.9]
    assert m.confidence("cav0", 5) == pytest.approx(0.8)
    with pytest.raises(KeyError):
        m.for_area(7)
    with pytest.raises(KeyError):
        m.for_agent("ghost")


def test_replace_values_does_not_mutate_the_clean_matrix() -> None:
    """Fault injection must produce a NEW matrix so clean and corrupted views
    can both be logged and compared."""
    m = AreaConfidenceMatrix(
        values=np.array([[0.8, 0.1]]), area_ids=np.array([0, 1]), agent_ids=("a",)
    )
    corrupted = m.replace_values(np.array([[0.0, 0.0]]))
    assert m.values.tolist() == [[0.8, 0.1]]
    assert corrupted.values.tolist() == [[0.0, 0.0]]
    assert corrupted.agent_ids == m.agent_ids


def test_matrix_validates_shapes() -> None:
    with pytest.raises(ValueError):
        AreaConfidenceMatrix(np.zeros((2, 3)), np.array([0, 1]), ("a", "b"))
    with pytest.raises(ValueError):
        AreaConfidenceMatrix(np.zeros((2, 2)), np.array([0, 1]), ("a",))
