"""
Tests for lgcpbench.selection -- paper Algorithm 1.

Three families of property:
    * Eq. 8 is respected exactly -- every admitted CAV cleared the threshold,
      and early termination never changes the answer;
    * Eq. 9 holds -- exactly one leader per non-orphaned group, always a
      member;
    * the Fig. 3 / Table II trend reproduces -- larger dg gives smaller
      groups and lower area confidence. That trend IS the paper's headline
      trade-off, so if it inverted, the implementation would be wrong in a
      way no shape assertion would catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpbench.observation.recorders import StatsTap
from cpbench.observation.taps import TapSet
from lgcpbench.confidence import AreaConfidenceMatrix, NoisyOrCombiner
from lgcpbench.observation import ControlPlaneTap
from lgcpbench.selection import (
    FirstMemberLeaderElector,
    GreedyGroupSelector,
    Group,
    MinMaxLoadLeaderElector,
    SelectionAlgorithm,
    SelectionResult,
)


def _matrix(values: np.ndarray, n_areas: int = None) -> AreaConfidenceMatrix:
    v, a = values.shape
    return AreaConfidenceMatrix(
        values=values,
        area_ids=np.arange(a),
        agent_ids=tuple(f"cav{i}" for i in range(v)),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)


# --------------------------------------------------------------------- #
# Group dataclass
# --------------------------------------------------------------------- #


def test_group_basics() -> None:
    g = Group(area_id=3, members=("a", "b"), confidence=0.8, leader="b")
    assert g.size == 2 and not g.is_orphaned and g.has_leader
    assert g.transmitting_members == ("a",)


def test_orphaned_group() -> None:
    g = Group(area_id=1, members=(), confidence=0.0)
    assert g.is_orphaned and g.size == 0 and not g.has_leader


def test_leader_must_be_a_member() -> None:
    with pytest.raises(ValueError):
        Group(area_id=0, members=("a",), confidence=0.5, leader="b")


def test_group_rejects_duplicate_members() -> None:
    with pytest.raises(ValueError):
        Group(area_id=0, members=("a", "a"), confidence=0.5)


def test_leader_does_not_transmit_to_itself() -> None:
    """Paper section V-B: only non-leader members send. The leader already
    holds its own features (B9), which is why leader placement changes
    communication cost, not just computation cost."""
    g = Group(area_id=0, members=("a", "b", "c"), confidence=0.9, leader="b")
    assert set(g.transmitting_members) == {"a", "c"}


# --------------------------------------------------------------------- #
# Eq. 8 -- greedy grouping
# --------------------------------------------------------------------- #


def test_every_admitted_cav_cleared_the_threshold(rng: np.random.Generator) -> None:
    """Replays Eq. 8 independently against the produced membership."""
    sel = GreedyGroupSelector(delta_g=0.1, max_group_size=None)
    c = NoisyOrCombiner()
    for _ in range(100):
        n = int(rng.integers(1, 8))
        conf = rng.uniform(0, 1, size=n)
        ids = tuple(f"cav{i}" for i in range(n))
        g = sel.select_area(0, ids, conf)

        by_id = dict(zip(ids, conf))
        current = 0.0
        for m in g.members:
            assert c.gain(current, by_id[m]) >= sel.delta_g - 1e-12
            current = c.update(current, by_id[m])
        assert g.confidence == pytest.approx(current)


def test_early_stop_agrees_with_exhaustive_scan(rng: np.random.Generator) -> None:
    """The guard on the non-increasing-gain argument.

    If the ordering assumption were ever violated, early termination would
    silently drop admissible CAVs -- and nothing else in the pipeline would
    notice. So the two modes are compared directly on random inputs.
    """
    fast = GreedyGroupSelector(delta_g=0.1, max_group_size=None, early_stop=True)
    slow = GreedyGroupSelector(delta_g=0.1, max_group_size=None, early_stop=False)
    for _ in range(200):
        n = int(rng.integers(1, 10))
        conf = rng.uniform(0, 1, size=n)
        ids = tuple(f"cav{i}" for i in range(n))
        assert fast.select_area(0, ids, conf).members == slow.select_area(0, ids, conf).members


def test_members_are_in_descending_confidence_order() -> None:
    """Algorithm 1 line 2 sorts V by area confidence."""
    sel = GreedyGroupSelector(delta_g=0.0, max_group_size=None)
    g = sel.select_area(0, ("a", "b", "c"), np.array([0.3, 0.9, 0.6]))
    assert g.members == ("b", "c", "a")


def test_ties_broken_deterministically() -> None:
    """Reproducibility precondition: a schedule difference must be
    attributable to an injected fault, never to dict ordering."""
    sel = GreedyGroupSelector(delta_g=0.0, max_group_size=None)
    conf = np.array([0.5, 0.5, 0.5])
    first = sel.select_area(0, ("c", "a", "b"), conf).members
    for _ in range(20):
        assert sel.select_area(0, ("c", "a", "b"), conf).members == first
    assert first == ("a", "b", "c")


def test_larger_delta_g_gives_smaller_groups(rng: np.random.Generator) -> None:
    """The Fig. 3 trend: "As the dg increases, the RSU tends to select fewer
    CAVs for each group, resulting in a decrease in the perception accuracy."
    """
    conf = rng.uniform(0, 1, size=(7, 40))
    matrix = _matrix(conf)
    sizes, confidences = [], []
    for dg in (0.0, 0.05, 0.075, 0.1, 0.125, 0.25):
        res = SelectionAlgorithm(
            GreedyGroupSelector(delta_g=dg, max_group_size=None)
        )(matrix)
        sizes.append(res.mean_group_size)
        confidences.append(res.accuracy_proxy)

    assert all(b <= a + 1e-9 for a, b in zip(sizes, sizes[1:])), sizes
    assert all(b <= a + 1e-9 for a, b in zip(confidences, confidences[1:])), confidences
    assert sizes[0] > sizes[-1]  # the trend is real, not flat


def test_delta_g_zero_admits_everyone() -> None:
    sel = GreedyGroupSelector(delta_g=0.0, max_group_size=None)
    g = sel.select_area(0, ("a", "b", "c"), np.array([0.5, 0.4, 0.3]))
    assert g.size == 3


def test_max_group_size_caps_admission() -> None:
    """Assumption B3: dg is the primary control, this bounds the worst case."""
    sel = GreedyGroupSelector(delta_g=0.0, max_group_size=2)
    g = sel.select_area(0, ("a", "b", "c", "d"), np.array([0.9, 0.8, 0.7, 0.6]))
    assert g.size == 2 and g.members == ("a", "b")


def test_area_no_one_can_see_is_orphaned() -> None:
    """The greedy starts from F=0, so the strongest candidate must itself
    clear dg. An area nobody sees well produces an empty group -- a real
    outcome and a headline robustness signal, not an error."""
    sel = GreedyGroupSelector(delta_g=0.5)
    g = sel.select_area(0, ("a", "b"), np.array([0.1, 0.05]))
    assert g.is_orphaned and g.confidence == 0.0


def test_group_confidence_matches_eq2(rng: np.random.Generator) -> None:
    sel = GreedyGroupSelector(delta_g=0.05, max_group_size=None)
    c = NoisyOrCombiner()
    for _ in range(50):
        n = int(rng.integers(1, 7))
        conf = rng.uniform(0, 1, size=n)
        ids = tuple(f"cav{i}" for i in range(n))
        g = sel.select_area(0, ids, conf)
        by_id = dict(zip(ids, conf))
        assert g.confidence == pytest.approx(c.combine([by_id[m] for m in g.members]))


def test_selector_validates_input() -> None:
    sel = GreedyGroupSelector()
    with pytest.raises(ValueError):
        sel.select_area(0, ("a", "b"), np.array([0.5]))
    with pytest.raises(ValueError):
        GreedyGroupSelector(delta_g=-0.1)
    with pytest.raises(ValueError):
        GreedyGroupSelector(max_group_size=0)


def test_select_all_covers_every_area() -> None:
    matrix = _matrix(np.random.default_rng(0).uniform(0, 1, size=(3, 6)))
    groups = GreedyGroupSelector(delta_g=0.05).select_all(matrix)
    assert [g.area_id for g in groups] == matrix.area_ids.tolist()


# --------------------------------------------------------------------- #
# Eq. 9, 10 -- leader election
# --------------------------------------------------------------------- #


def test_exactly_one_leader_per_group(rng: np.random.Generator) -> None:
    """Eq. 9: SUM_j y_i,j = 1 for every group."""
    matrix = _matrix(rng.uniform(0, 1, size=(6, 25)))
    res = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.05))(matrix)
    for g in res.groups:
        if g.is_orphaned:
            assert g.leader is None
        else:
            assert g.leader is not None and g.leader in g.members


def test_loads_match_eq10(rng: np.random.Generator) -> None:
    """L_j = SUM_i y_i,j * |V^_i| * B, recomputed independently."""
    matrix = _matrix(rng.uniform(0, 1, size=(5, 20)))
    res = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.05))(matrix)
    expected = {cav: 0.0 for cav in matrix.agent_ids}
    for g in res.groups:
        if g.leader is not None:
            expected[g.leader] += g.size
    assert res.loads == pytest.approx(expected)


def test_greedy_invariant_leader_had_minimal_load_at_assignment() -> None:
    """Algorithm 1 line 8, replayed: process groups in LPT order and check
    the chosen leader was minimal-load among its own members at that time."""
    groups = [
        Group(0, ("a", "b", "c"), 0.9),
        Group(1, ("a", "b"), 0.8),
        Group(2, ("a",), 0.7),
        Group(3, ("b", "c"), 0.6),
    ]
    elected, _ = MinMaxLoadLeaderElector().elect(groups)
    by_area = {g.area_id: g for g in elected}

    running = {cav: 0.0 for cav in ("a", "b", "c")}
    order = sorted(groups, key=lambda g: (-g.size, g.area_id))
    for g in order:
        chosen = by_area[g.area_id].leader
        best = min(running[m] for m in g.members)
        assert running[chosen] == pytest.approx(best)
        running[chosen] += g.size


def test_min_max_beats_first_member_on_balance() -> None:
    """The ablation isolating contribution C4: identical membership, only
    leader placement differs, so any gap is load balancing alone.

    Constructed so one CAV is the strongest everywhere -- the case where a
    confidence-first policy piles every group onto it.
    """
    groups = [Group(i, ("hub", f"spoke{i}"), 0.9) for i in range(6)]
    _, balanced = MinMaxLoadLeaderElector().elect(groups)
    _, naive = FirstMemberLeaderElector().elect(groups)

    assert max(naive.values()) == 12.0  # every group on the hub
    assert max(balanced.values()) < max(naive.values())


def test_orphaned_groups_get_no_leader_and_no_load() -> None:
    groups = [Group(0, (), 0.0), Group(1, ("a",), 0.5)]
    elected, loads = MinMaxLoadLeaderElector().elect(groups)
    assert elected[0].leader is None
    assert loads["a"] == 1.0


def test_agents_leading_nothing_still_appear_in_loads() -> None:
    groups = [Group(0, ("a",), 0.5)]
    _, loads = MinMaxLoadLeaderElector().elect(groups, agents=("a", "b", "c"))
    assert loads == {"a": 1.0, "b": 0.0, "c": 0.0}


def test_per_area_bits_override() -> None:
    """The paper assumes B uniform; our areas differ slightly in cell count,
    so a true per-area B is supported for the more accurate variant."""
    groups = [Group(0, ("a", "b"), 0.9), Group(1, ("a", "b"), 0.9)]
    _, loads = MinMaxLoadLeaderElector(area_bits={0: 10.0, 1: 100.0}).elect(groups)
    assert sorted(loads.values()) == [20.0, 200.0]


def test_election_is_deterministic(rng: np.random.Generator) -> None:
    matrix = _matrix(rng.uniform(0, 1, size=(5, 15)))
    alg = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.05))
    first = [(g.area_id, g.members, g.leader) for g in alg(matrix).groups]
    for _ in range(10):
        assert [(g.area_id, g.members, g.leader) for g in alg(matrix).groups] == first


def test_makespan_and_imbalance() -> None:
    assert MinMaxLoadLeaderElector.makespan({"a": 3.0, "b": 1.0}) == 3.0
    assert MinMaxLoadLeaderElector.imbalance({"a": 2.0, "b": 2.0}) == pytest.approx(1.0)
    assert MinMaxLoadLeaderElector.imbalance({"a": 0.0, "b": 0.0}) == 1.0
    assert MinMaxLoadLeaderElector.imbalance({}) == 1.0


# --------------------------------------------------------------------- #
# SelectionResult / SelectionAlgorithm
# --------------------------------------------------------------------- #


def test_result_reports_orphaned_areas() -> None:
    res = SelectionResult(
        groups=[Group(0, ("a",), 0.5, "a"), Group(1, (), 0.0), Group(2, (), 0.0)],
        loads={"a": 1.0},
        delta_g=0.075,
    )
    assert res.n_areas == 3 and res.n_orphaned == 2
    assert res.orphaned_area_ids == (1, 2)
    # Eq. 3: orphaned areas contribute 0.0, so losing coverage is penalised
    assert res.accuracy_proxy == pytest.approx(0.5 / 3)


def test_result_record_is_flat_and_loggable() -> None:
    matrix = _matrix(np.random.default_rng(1).uniform(0, 1, size=(4, 10)))
    res = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075))(matrix)
    row = res.as_record()
    assert set(row) >= {
        "n_areas", "n_orphaned_areas", "mean_group_size", "max_group_size",
        "mean_area_confidence", "delta_g", "leader_load_max",
        "leader_load_imbalance", "n_leaders",
    }
    assert all(not isinstance(v, (list, dict, tuple)) for v in row.values())


def test_result_group_lookup() -> None:
    res = SelectionResult(groups=[Group(7, ("a",), 0.5, "a")], loads={"a": 1.0})
    assert res.group_for(7).members == ("a",)
    with pytest.raises(KeyError):
        res.group_for(99)


def test_algorithm_emits_control_plane_taps() -> None:
    """Control-plane decisions are not tensors, so they need ControlPlaneTap.

    corabench's StatsTap ignores non-tensors by contract -- routing groups
    and loads through it would make the entire control plane silently
    unobservable while every other test still passed.
    """
    tap = ControlPlaneTap(retain=True)
    matrix = _matrix(np.random.default_rng(3).uniform(0, 1, size=(4, 8)))
    SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075))(matrix, taps=TapSet([tap]))
    assert {
        "lgcp/selection/groups",
        "lgcp/selection/leaders",
        "lgcp/selection/loads",
    } <= set(tap.locations())


def test_stats_tap_alone_cannot_see_the_control_plane() -> None:
    """Pins the reason ControlPlaneTap exists, so the gap cannot silently
    reopen if corabench's recorder behaviour ever changes."""
    stats = StatsTap()
    matrix = _matrix(np.random.default_rng(3).uniform(0, 1, size=(4, 8)))
    SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075))(matrix, taps=TapSet([stats]))
    assert stats.records == []


def test_control_tap_summarises_groups_and_loads() -> None:
    tap = ControlPlaneTap(retain=True)
    matrix = _matrix(np.array([[0.9, 0.01], [0.8, 0.01]]))
    SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075))(matrix, taps=TapSet([tap]))

    groups = tap.latest("lgcp/selection/groups")
    assert groups is not None
    assert groups.summary["n_orphaned"] == 1  # area 1 is unperceivable
    assert groups.context["delta_g"] == 0.075

    loads = tap.latest("lgcp/selection/loads")
    assert loads is not None and loads.summary["value_max"] >= 1.0


def test_control_tap_csv_export(tmp_path) -> None:
    tap = ControlPlaneTap()
    matrix = _matrix(np.random.default_rng(5).uniform(0, 1, size=(3, 5)))
    SelectionAlgorithm(GreedyGroupSelector())(matrix, taps=TapSet([tap]))
    out = tap.to_csv(tmp_path / "control.csv")
    text = out.read_text()
    assert "location" in text and "lgcp/selection/groups" in text


def test_control_tap_does_not_retain_by_default() -> None:
    """A 30-CAV schedule over hundreds of frames must not silently accumulate."""
    tap = ControlPlaneTap()
    matrix = _matrix(np.random.default_rng(6).uniform(0, 1, size=(3, 5)))
    SelectionAlgorithm(GreedyGroupSelector())(matrix, taps=TapSet([tap]))
    assert tap.records and all(r.payload is None for r in tap.records)


def test_algorithm_output_identical_with_and_without_taps() -> None:
    """The measurement-plane invariant, extended to the control plane:
    observing a decision must not change it."""
    matrix = _matrix(np.random.default_rng(4).uniform(0, 1, size=(5, 12)))
    alg = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075))
    clean = alg(matrix)
    tapped = alg(matrix, taps=TapSet([StatsTap()]))
    assert [(g.area_id, g.members, g.leader) for g in clean.groups] == [
        (g.area_id, g.members, g.leader) for g in tapped.groups
    ]
    assert clean.loads == tapped.loads


def test_falsified_confidence_changes_grouping() -> None:
    """The control-plane fault surface, end to end at this layer.

    A CAV that inflates its reported confidence for an area gets admitted to
    a group it does not deserve -- the falsified-report attack. Nothing in
    selection/ is fault-aware; the corruption happened upstream on the
    matrix, and Algorithm 1 ran exactly as published on it.
    """
    clean = _matrix(np.array([[0.9, 0.9], [0.02, 0.02]]))
    alg = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075, max_group_size=None))
    assert alg(clean).groups[0].members == ("cav0",)

    liar = clean.replace_values(np.array([[0.9, 0.9], [0.95, 0.95]]))
    assert "cav1" in alg(liar).groups[0].members


def test_scales_to_dense_multi_cav(rng: np.random.Generator) -> None:
    """Fig. 7 regime: 30 CAVs, many occupied areas, must stay fast and sane."""
    matrix = _matrix(rng.uniform(0, 1, size=(30, 200)))
    res = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075))(matrix)
    assert res.n_areas == 200
    assert res.max_group_size <= 5
    assert res.makespan > 0
