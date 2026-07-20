"""
Tests for lgcpbench.network -- Algorithm 2, Table I PHY, Eq. 4-5-7-11.

The single most important property in this file is
``test_schedule_is_always_conflict_free``: Algorithm 2's entire purpose is to
produce a schedule no two transmissions collide in. Everything else -- the
latency numbers, the reduction ratios, the whole benchmark -- is meaningless
if that does not hold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cpbench.observation.taps import TapSet
from lgcpbench.network import (
    FusionLatencyModel,
    InterferenceModel,
    LatencyModel,
    MODEL_MFLOPS,
    Packet,
    PathLossModel,
    RateModel,
    ShadowingModel,
    TransmissionScheduler,
    build_packets,
    priority,
    receiver_load,
    sender_load,
)
from lgcpbench.observation import ControlPlaneTap
from lgcpbench.selection import Group


def _positions(n: int, spacing: float = 20.0) -> dict:
    return {f"cav{i}": (i * spacing, 0.0) for i in range(n)}


def _scheduler(positions, **kw) -> TransmissionScheduler:
    im = InterferenceModel(positions, interference_range_m=kw.pop("range_m", 1e6))
    return TransmissionScheduler(im, **kw)


# --------------------------------------------------------------------- #
# PHY -- Table I
# --------------------------------------------------------------------- #


def test_path_loss_matches_table_i() -> None:
    """128.1 + 36.6 log10(d_km): at 1 km the loss is exactly the intercept."""
    pl = PathLossModel()
    assert pl.loss_db(1000.0) == pytest.approx(128.1)
    assert pl.loss_db(100.0) == pytest.approx(128.1 - 36.6)


def test_path_loss_is_monotone_in_distance() -> None:
    pl = PathLossModel()
    d = np.array([10.0, 50.0, 200.0, 1000.0])
    losses = pl.loss_db_array(d)
    assert np.all(np.diff(losses) > 0)


def test_path_loss_inverse_round_trips() -> None:
    pl = PathLossModel()
    for d in (25.0, 100.0, 750.0):
        assert pl.distance_for_loss_m(pl.loss_db(d)) == pytest.approx(d, rel=1e-9)


def test_path_loss_clamps_at_zero_distance() -> None:
    """The log-distance model is not physical below ~1 m; clamp rather than
    emit nonsense (a co-located pair would otherwise get negative loss)."""
    assert math.isfinite(PathLossModel().loss_db(0.0))


def test_sinr_threshold_is_derived_not_hardcoded() -> None:
    """27 Mbps over 8 MHz is 3.375 bit/s/Hz, so the threshold SINR follows
    from Shannon. Changing either config value must keep it consistent."""
    rm = RateModel()
    assert rm.sinr_threshold_db == pytest.approx(9.72, abs=0.01)

    wider = RateModel(subchannel_bandwidth_hz=16e6)
    assert wider.sinr_threshold_db < rm.sinr_threshold_db


def test_rate_model_is_a_step_not_shannon() -> None:
    """Section VI-C: below 27 Mbps the link is disabled; above it, the rate
    is fixed at 27 Mbps -- it does not keep rising with SINR."""
    rm = RateModel(shadowing=ShadowingModel(enabled=False))
    near = rm.link("a", "b", 10.0)
    far = rm.link("a", "b", 200.0)
    assert near.usable and far.usable
    assert near.rate_bps == far.rate_bps == pytest.approx(27e6)
    assert near.shannon_bps > far.shannon_bps  # the margin still differs


def test_link_becomes_unusable_beyond_range() -> None:
    rm = RateModel(shadowing=ShadowingModel(enabled=False))
    assert rm.is_usable("a", "b", 50.0)
    assert not rm.is_usable("a", "b", 50_000.0)


def test_max_range_is_the_usability_boundary() -> None:
    """B6: the derived interference range is exactly where the link dies."""
    rm = RateModel(shadowing=ShadowingModel(enabled=False))
    r = rm.max_range_m()
    assert rm.is_usable("a", "b", r * 0.99)
    assert not rm.is_usable("a", "b", r * 1.01)


def test_shadowing_is_deterministic_and_reciprocal() -> None:
    """A fault study compares clean vs corrupted runs. If shadowing were
    freshly random per call, every comparison would be contaminated by noise
    that looks exactly like a fault effect."""
    s = ShadowingModel(seed=7)
    first = s.for_link("a", "b")
    assert s.for_link("a", "b") == first
    assert s.for_link("b", "a") == first          # reciprocity
    assert ShadowingModel(seed=7).for_link("a", "b") == first   # across instances
    assert ShadowingModel(seed=8).for_link("a", "b") != first   # seed matters


def test_shadowing_can_be_disabled() -> None:
    assert ShadowingModel(enabled=False).for_link("a", "b") == 0.0
    assert ShadowingModel(std_db=0.0).for_link("a", "b") == 0.0


def test_shadowing_distribution_is_roughly_normal() -> None:
    s = ShadowingModel(std_db=8.0, seed=1)
    draws = np.array([s.for_link("a", f"cav{i}") for i in range(2000)])
    assert abs(draws.mean()) < 1.0
    assert 7.0 < draws.std() < 9.0


def test_packet_bits_agrees_with_derivation_d2() -> None:
    """tau = 0.25 ms at 27 Mbps carries 6750 bits. D2 puts an area-restricted
    feature at 256 * ~24 = ~6100 bits. The paper's slot is sized to one area
    packet -- an independent check that D2's 1-bit-per-element reading of the
    "2.16Mb" figure is right."""
    assert RateModel().packet_bits() == pytest.approx(6750.0)
    assert 0.7 < (256 * 24) / RateModel().packet_bits() < 1.3


# --------------------------------------------------------------------- #
# packets and Eq. 11
# --------------------------------------------------------------------- #


def test_build_packets_skips_the_leader() -> None:
    """Section V-B: only non-leader members transmit. The leader already
    holds its own features (B9)."""
    groups = [Group(0, ("a", "b", "c"), 0.9, leader="b")]
    packets = build_packets(groups)
    assert [(p.v_s, p.v_r) for p in packets] == [("a", "b"), ("c", "b")]


def test_build_packets_skips_orphaned_and_leaderless_groups() -> None:
    """A leaderless group is a real fault outcome (leader-failure injection);
    it manifests as an area that never gets aggregated."""
    groups = [Group(0, (), 0.0), Group(1, ("a", "b"), 0.5, leader=None)]
    assert build_packets(groups) == []


def test_build_packets_uses_per_area_payload() -> None:
    groups = [Group(0, ("a", "b"), 0.9, leader="b"), Group(1, ("a", "c"), 0.8, leader="c")]
    packets = build_packets(groups, area_bits={0: 6144, 1: 7168})
    assert {p.area_id: p.bits for p in packets} == {0: 6144, 1: 7168}


def test_packet_ids_are_stable() -> None:
    groups = [Group(1, ("a", "b"), 0.9, leader="b"), Group(0, ("a", "c"), 0.8, leader="c")]
    ids = [(p.id, p.area_id, p.v_s) for p in build_packets(groups)]
    assert ids == [(0, 0, "a"), (1, 1, "a")]


def test_packet_rejects_self_transmission() -> None:
    with pytest.raises(ValueError):
        Packet(0, "a", "a", 0)


def test_packet_schedule_is_frozen_safe() -> None:
    p = Packet(0, "a", "b", 0)
    q = p.schedule(z=1, t=0.5)
    assert not p.is_scheduled and q.is_scheduled
    assert (q.z, q.t) == (1, 0.5)


def test_eq11_priority() -> None:
    """omega(v_s, v_r) = L_s(v_s) + L_r(v_r)."""
    ps = [Packet(0, "a", "L", 0), Packet(1, "b", "L", 0), Packet(2, "a", "M", 1)]
    assert sender_load(ps) == {"a": 2, "b": 1}
    assert receiver_load(ps) == {"L": 2, "M": 1}
    assert priority(ps) == {0: 4, 1: 3, 2: 3}


# --------------------------------------------------------------------- #
# interference -- I_E(p)
# --------------------------------------------------------------------- #


def test_self_interference_blocks_a_busy_cav() -> None:
    im = InterferenceModel(_positions(3), interference_range_m=1.0)
    scheduled = [Packet(0, "cav0", "cav1", 0, z=0, t=0.0)]
    assert im.conflicts(Packet(1, "cav1", "cav2", 1), scheduled)   # cav1 receiving
    assert im.conflicts(Packet(2, "cav2", "cav0", 1), scheduled)   # cav0 sending
    assert im.conflicts(Packet(3, "cav0", "cav2", 1), scheduled)   # one transmitter


def test_disjoint_links_do_not_self_interfere() -> None:
    im = InterferenceModel(_positions(4), interference_range_m=1.0)
    scheduled = [Packet(0, "cav0", "cav1", 0, z=0, t=0.0)]
    assert not im.conflicts(Packet(1, "cav2", "cav3", 1, z=1, t=0.0), scheduled)


def test_co_channel_fires_only_on_a_shared_subchannel() -> None:
    """Dormant under Algorithm 2 (one packet per subchannel per slot), live
    when a fault injector forces two packets onto the same channel."""
    im = InterferenceModel(_positions(4, spacing=5.0), interference_range_m=1e6)
    scheduled = [Packet(0, "cav0", "cav1", 0, z=0, t=0.0)]
    same = Packet(1, "cav2", "cav3", 1, z=0, t=0.0)
    other = Packet(2, "cav2", "cav3", 1, z=1, t=0.0)
    assert im.conflicts(same, scheduled)
    assert not im.conflicts(other, scheduled)


def test_co_channel_respects_the_interference_range() -> None:
    im = InterferenceModel(
        {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (10_000.0, 0.0), "d": (10_001.0, 0.0)},
        interference_range_m=50.0,
    )
    scheduled = [Packet(0, "a", "b", 0, z=0, t=0.0)]
    far = Packet(1, "c", "d", 1, z=0, t=0.0)
    assert not im.conflicts(far, scheduled)   # too distant to collide


def test_explain_names_the_rule() -> None:
    im = InterferenceModel(_positions(3), interference_range_m=1.0)
    conflict = im.explain(
        Packet(1, "cav1", "cav2", 1), [Packet(0, "cav0", "cav1", 0, z=0, t=0.0)]
    )
    assert conflict is not None and conflict.kind == "self"


def test_interference_requires_a_range_source() -> None:
    with pytest.raises(ValueError):
        InterferenceModel(_positions(2))


def test_interference_reports_missing_positions() -> None:
    im = InterferenceModel(_positions(2), interference_range_m=10.0)
    with pytest.raises(KeyError):
        im.distance("cav0", "ghost")


# --------------------------------------------------------------------- #
# Algorithm 2
# --------------------------------------------------------------------- #


def test_schedule_is_always_conflict_free() -> None:
    """THE correctness property of Algorithm 2.

    Every latency number and reduction ratio in the benchmark is meaningless
    if two transmissions in the same slot collide. Verified by replaying the
    interference audit slot by slot on randomised topologies.
    """
    rng = np.random.default_rng(0)
    for trial in range(30):
        n_cav = int(rng.integers(3, 9))
        positions = {f"cav{i}": tuple(rng.uniform(-140, 140, size=2)) for i in range(n_cav)}
        names = list(positions)

        groups, area_bits = [], {}
        for area in range(int(rng.integers(2, 10))):
            size = int(rng.integers(2, min(5, n_cav) + 1))
            members = tuple(rng.choice(names, size=size, replace=False))
            groups.append(Group(area, members, 0.9, leader=members[0]))
            area_bits[area] = 6144

        im = InterferenceModel(positions, interference_range_m=1e6)
        result = TransmissionScheduler(im).schedule(
            build_packets(groups, area_bits=area_bits),
            group_sizes={g.area_id: g.size for g in groups},
            leaders={g.area_id: g.leader for g in groups},
        )
        assert not result.unscheduled, f"trial {trial}: packets left unscheduled"

        by_slot = {}
        for p in result.packets:
            by_slot.setdefault(p.t, []).append(p)
        for slot_time, slot in by_slot.items():
            assert im.audit(slot) == (), f"trial {trial}: collision at t={slot_time}"


def test_every_packet_gets_scheduled_exactly_once() -> None:
    positions = _positions(6)
    groups = [Group(i, (f"cav{i}", f"cav{(i + 1) % 6}", f"cav{(i + 2) % 6}"),
                    0.9, leader=f"cav{i}") for i in range(4)]
    packets = build_packets(groups)
    result = _scheduler(positions).schedule(
        packets,
        group_sizes={g.area_id: g.size for g in groups},
        leaders={g.area_id: g.leader for g in groups},
    )
    assert len(result.packets) == len(packets)
    assert sorted(p.id for p in result.packets) == sorted(p.id for p in packets)


def test_half_duplex_serialises_a_shared_receiver() -> None:
    """One leader receiving from two members cannot do both at once, even
    with 5 free subchannels -- the half-duplex constraint, not spectrum, is
    the binding one here."""
    positions = _positions(3)
    packets = [Packet(0, "cav0", "cav1", 0), Packet(2, "cav2", "cav1", 0)]
    result = _scheduler(positions).schedule(
        packets, group_sizes={0: 3}, leaders={0: "cav1"}
    )
    assert result.n_slots == 2
    assert result.subchannel_utilisation == pytest.approx(2 / (2 * 5))


def test_disjoint_links_transmit_concurrently() -> None:
    """Independent leader/member pairs use separate subchannels in one slot."""
    positions = _positions(4)
    packets = [Packet(0, "cav0", "cav1", 0), Packet(1, "cav2", "cav3", 1)]
    result = _scheduler(positions).schedule(
        packets, group_sizes={0: 2, 1: 2}, leaders={0: "cav1", 1: "cav3"}
    )
    assert result.n_slots == 1
    assert {p.z for p in result.packets} == {0, 1}


def test_concurrency_is_capped_by_subchannel_count() -> None:
    positions = _positions(20)
    packets = [Packet(i, f"cav{2 * i}", f"cav{2 * i + 1}", i) for i in range(10)]
    result = _scheduler(positions, n_subchannels=3).schedule(
        packets,
        group_sizes={i: 2 for i in range(10)},
        leaders={i: f"cav{2 * i + 1}" for i in range(10)},
    )
    assert result.n_slots == math.ceil(10 / 3)


def test_higher_priority_packets_go_first() -> None:
    """Eq. 11: the busiest sender/receiver pair is scheduled earliest, so its
    leader can begin fusing sooner."""
    positions = _positions(5)
    packets = [
        Packet(0, "cav1", "cav0", 0),   # busy leader cav0
        Packet(1, "cav2", "cav0", 0),
        Packet(2, "cav3", "cav0", 0),
        Packet(3, "cav4", "cav1", 1),   # quiet pair
    ]
    result = _scheduler(positions).schedule(
        packets, group_sizes={0: 4, 1: 2}, leaders={0: "cav0", 1: "cav1"}
    )
    times = {p.id: p.t for p in result.packets}
    assert times[0] <= times[3]


def test_empty_packet_set() -> None:
    result = _scheduler(_positions(2)).schedule([], group_sizes={}, leaders={})
    assert result.n_slots == 0 and result.makespan == 0.0
    assert result.subchannel_utilisation == 0.0


def test_group_of_one_is_fusable_immediately() -> None:
    """No packets to await, so the area is ready at t=0. This is why raising
    dg lowers latency as well as accuracy."""
    result = _scheduler(_positions(2)).schedule(
        [], group_sizes={0: 1}, leaders={0: "cav0"}
    )
    assert result.area_ready == {0: 0.0}
    assert result.makespan == pytest.approx(FusionLatencyModel().fusion_time_s(1))


def test_scheduling_is_deterministic() -> None:
    positions = _positions(6)
    groups = [Group(i, (f"cav{i}", f"cav{(i + 3) % 6}"), 0.9, leader=f"cav{i}")
              for i in range(4)]
    packets = build_packets(groups)
    sizes = {g.area_id: g.size for g in groups}
    leaders = {g.area_id: g.leader for g in groups}

    first = None
    for _ in range(5):
        r = _scheduler(positions).schedule(packets, group_sizes=sizes, leaders=leaders)
        signature = [(p.id, p.z, p.t) for p in r.packets]
        first = signature if first is None else first
        assert signature == first


def test_scheduler_emits_control_plane_taps() -> None:
    tap = ControlPlaneTap(retain=True)
    _scheduler(_positions(3)).schedule(
        [Packet(0, "cav0", "cav1", 0)],
        group_sizes={0: 2}, leaders={0: "cav1"}, taps=TapSet([tap]),
    )
    assert {"lgcp/network/priority", "lgcp/network/schedule"} <= set(tap.locations())


def test_schedule_output_identical_with_and_without_taps() -> None:
    positions = _positions(4)
    packets = [Packet(0, "cav0", "cav1", 0), Packet(1, "cav2", "cav3", 1)]
    kw = dict(group_sizes={0: 2, 1: 2}, leaders={0: "cav1", 1: "cav3"})
    clean = _scheduler(positions).schedule(packets, **kw)
    tapped = _scheduler(positions).schedule(packets, taps=TapSet([ControlPlaneTap()]), **kw)
    assert [(p.id, p.z, p.t) for p in clean.packets] == [
        (p.id, p.z, p.t) for p in tapped.packets
    ]
    assert clean.makespan == tapped.makespan


# --------------------------------------------------------------------- #
# fusion overlap -- section V-B's parallelisation claim
# --------------------------------------------------------------------- #


def test_fusion_overlaps_with_other_areas_transmission() -> None:
    """Section V-B: "a leader CAV can fuse received packets during
    transmission once all packets are fully received".

    So the makespan must be less than (all transmission) + (all fusion).
    """
    positions = _positions(6)
    groups = [
        Group(0, ("cav0", "cav1"), 0.9, leader="cav0"),
        Group(1, ("cav2", "cav3"), 0.9, leader="cav2"),
        Group(2, ("cav4", "cav5"), 0.9, leader="cav4"),
    ]
    packets = build_packets(groups)
    sizes = {g.area_id: g.size for g in groups}
    result = _scheduler(positions).schedule(
        packets, group_sizes=sizes, leaders={g.area_id: g.leader for g in groups}
    )
    fm = FusionLatencyModel()
    naive_serial = result.transmission_span + sum(fm.fusion_time_s(s) for s in sizes.values())
    assert result.makespan < naive_serial


def test_concentrating_leadership_lengthens_the_makespan() -> None:
    """Why Eq. 10's min-max balancing matters: identical total work, but one
    leader must serialise its fusion queue."""
    positions = _positions(7)
    spread = [Group(i, (f"cav{i}", f"cav{i + 3}"), 0.9, leader=f"cav{i}") for i in range(3)]
    hub = [Group(i, ("cav6", f"cav{i}"), 0.9, leader="cav6") for i in range(3)]

    def makespan(groups):
        return _scheduler(positions).schedule(
            build_packets(groups),
            group_sizes={g.area_id: g.size for g in groups},
            leaders={g.area_id: g.leader for g in groups},
        ).makespan

    assert makespan(hub) > makespan(spread)


def test_fusion_latency_uses_paper_numbers() -> None:
    """Section VI-C: Where2comm 1400 MFLOPs, CAV 0.1 TFLOPS -> 14 ms/member."""
    fm = FusionLatencyModel.for_model("where2comm")
    assert fm.fusion_time_s(1) == pytest.approx(14e-3)
    assert fm.fusion_time_s(3) == pytest.approx(42e-3)
    assert set(MODEL_MFLOPS) == {"cobevt", "where2comm", "coalign"}


def test_edge_server_is_twenty_times_faster() -> None:
    """The asymmetry the edge-assisted baseline exploits (2 vs 0.1 TFLOPS),
    and that LGCP trades away in exchange for parallelism."""
    cav = FusionLatencyModel.for_model("where2comm")
    edge = FusionLatencyModel.for_model("where2comm", edge=True)
    assert cav.fusion_time_s(4) / edge.fusion_time_s(4) == pytest.approx(20.0)


def test_fusion_model_validation() -> None:
    with pytest.raises(ValueError):
        FusionLatencyModel(capacity_tflops=0.0)
    with pytest.raises(KeyError):
        FusionLatencyModel.for_model("nope")
    assert FusionLatencyModel().fusion_time_s(0) == 0.0


# --------------------------------------------------------------------- #
# Eq. 5 and Eq. 7
# --------------------------------------------------------------------- #


def test_t_delta_matches_eq5() -> None:
    lm = LatencyModel(rate_bps=27e6, n_subchannels=5)
    m = lm.message_bits
    expected = (m["D_init"] + 1 * m["D_info"] + m["D_ts"] + m["D_rep"] + m["D_G"]) / 27e6
    assert lm.t_delta(n_cavs=5) == pytest.approx(expected)


def test_t_delta_grows_in_ceil_steps_with_fleet_size() -> None:
    """The ceil(|V|/Z) term: |V| CAVs report over Z shared subchannels."""
    lm = LatencyModel(rate_bps=27e6, n_subchannels=5)
    assert lm.t_delta(5) == lm.t_delta(1)         # both one round
    assert lm.t_delta(6) > lm.t_delta(5)          # second round begins
    assert lm.t_delta(10) == lm.t_delta(6)


def test_breakdown_sums_to_total() -> None:
    lm = LatencyModel(rate_bps=27e6)
    b = lm.breakdown(n_cavs=5, t_aggregate=0.01, t_fuse=0.028, t_schedule=0.038)
    assert b.total == pytest.approx(b.t_delta + b.t_schedule)


def test_deadline_constraint_7a() -> None:
    lm = LatencyModel(rate_bps=27e6, deadline_s=100e-3)
    assert lm.breakdown(5, 0.01, 0.02, 0.05).deadline_met
    assert not lm.breakdown(5, 0.1, 0.2, 0.5).deadline_met


def test_objective_eq7() -> None:
    lm = LatencyModel(rate_bps=27e6)
    b = lm.breakdown(5, 0.01, 0.02, 0.05)
    assert lm.objective(0.8, b) == pytest.approx(0.8 / b.total)


def test_infeasible_schedules_score_zero_not_merely_low() -> None:
    """Constraint (7a) is a feasibility bound. Collapsing a deadline
    violation to a finite score would let an infeasible schedule win a
    sweep on accuracy alone."""
    lm = LatencyModel(rate_bps=27e6, deadline_s=10e-3)
    assert lm.objective(1.0, lm.breakdown(5, 0.1, 0.2, 0.5)) == 0.0


def test_breakdown_record_is_flat() -> None:
    row = LatencyModel().breakdown(5, 0.01, 0.02, 0.03).as_record()
    assert "t_total_ms" in row and "deadline_met" in row
    assert all(not isinstance(v, (list, dict, tuple)) for v in row.values())


def test_overhead_fraction_flags_a_protocol_dominated_run() -> None:
    """B7's message sizes are assumptions; if they dominate, a reported
    latency reduction says more about them than about the scheduler."""
    lm = LatencyModel(rate_bps=27e6)
    assert lm.breakdown(5, 0.0, 0.0, 1e-6).overhead_fraction > 0.9
    assert lm.breakdown(5, 0.0, 0.0, 1.0).overhead_fraction < 0.01


def test_latency_model_validation() -> None:
    with pytest.raises(ValueError):
        LatencyModel(rate_bps=0)
    with pytest.raises(ValueError):
        LatencyModel(n_subchannels=0)
    with pytest.raises(ValueError):
        TransmissionScheduler(
            InterferenceModel(_positions(2), interference_range_m=10.0), time_slot_s=0
        )


# --------------------------------------------------------------------- #
# scale
# --------------------------------------------------------------------- #


def test_scales_to_thirty_cavs() -> None:
    """The Fig. 7 regime. Algorithm 2 is O(|P|^2), so this is the cost check."""
    rng = np.random.default_rng(11)
    positions = {f"cav{i}": tuple(rng.uniform(-140, 140, size=2)) for i in range(30)}
    names = list(positions)
    groups = []
    for area in range(60):
        members = tuple(rng.choice(names, size=3, replace=False))
        groups.append(Group(area, members, 0.9, leader=members[0]))

    im = InterferenceModel(positions, interference_range_m=1e6)
    result = TransmissionScheduler(im).schedule(
        build_packets(groups),
        group_sizes={g.area_id: g.size for g in groups},
        leaders={g.area_id: g.leader for g in groups},
    )
    assert not result.unscheduled
    assert result.n_slots <= len(result.packets)
    assert result.makespan > 0
