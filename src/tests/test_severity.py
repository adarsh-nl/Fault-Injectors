"""Severity scale: constructibility, per-fault direction, and the A/B split.

The direction test is the one with teeth. A blanket "the number goes up as
severity rises" assertion would PASS VACUOUSLY for eight of the twelve faults,
because their constructor kwarg is the level index itself (1, 2, 3) and not
the physical quantity -- and it would be WRONG for the three quantities that
fall as the fault worsens (fog visibility 300->50 m, beams 16->4, darkness
noise-s 25->5). So direction is asserted per fault, against PHYSICAL.
"""

from __future__ import annotations

import pytest

from src.fault_injectors import severity as sev


def _constructible():
    return sorted(f for f in sev.SEVERITY if f not in sev.UNAVAILABLE)


# ── the table itself ───────────────────────────────────────────────────────

def test_every_fault_has_exactly_levels_1_2_3():
    for fault, levels in sev.SEVERITY.items():
        assert sorted(levels) == [1, 2, 3], f"{fault} levels: {sorted(levels)}"


def test_every_fault_is_classified_and_described():
    for fault in sev.SEVERITY:
        assert fault in sev.GROUP, f"{fault} missing from GROUP"
        assert sev.GROUP[fault] in ("A", "B")
        assert fault in sev.PHYSICAL, f"{fault} missing from PHYSICAL"
        assert sev.describe(fault)


def test_physical_covers_unavailable_faults_too():
    """motion_blur cannot be built but its scale is known; losing it would
    mean re-deriving the mapping later from the MultiCorrupt source."""
    for fault in sev.UNAVAILABLE:
        assert fault in sev.PHYSICAL
        assert sorted(sev.PHYSICAL[fault][2]) == [1, 2, 3]


# ── construction ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("fault", _constructible())
@pytest.mark.parametrize("level", [1, 2, 3])
def test_make_constructs_every_fault_at_every_level(fault, level):
    obj = sev.make(fault, level)
    assert obj is not None


@pytest.mark.parametrize("fault", _constructible())
def test_make_rejects_out_of_range_levels(fault):
    for bad in (0, 4, -1):
        with pytest.raises(ValueError):
            sev.make(fault, bad)


def test_unavailable_faults_raise_with_a_reason():
    for fault, reason in sev.UNAVAILABLE.items():
        with pytest.raises(KeyError) as excinfo:
            sev.make(fault, 1)
        assert "not constructible" in str(excinfo.value)
        assert reason[:30] in str(excinfo.value)


def test_unknown_fault_lists_the_known_ones():
    with pytest.raises(KeyError, match="unknown fault"):
        sev.make("no_such_fault", 1)


# ── direction, per fault ───────────────────────────────────────────────────

@pytest.mark.parametrize("fault", sorted(sev.PHYSICAL))
def test_physical_quantity_moves_in_the_declared_direction(fault):
    quantity, _unit, values, direction = sev.PHYSICAL[fault]
    v1, v2, v3 = values[1], values[2], values[3]
    if direction == "up":
        assert v1 < v2 < v3, (
            f"{fault}: {quantity} declared 'up' but goes {v1} -> {v2} -> {v3}")
    elif direction == "down":
        assert v1 > v2 > v3, (
            f"{fault}: {quantity} declared 'down' but goes {v1} -> {v2} -> {v3}")
    else:
        pytest.fail(f"{fault}: bad direction {direction!r}")


def test_the_three_decreasing_faults_are_exactly_the_expected_ones():
    """Pinned by name. If a fault silently changes direction, the blanket
    'severity goes up' intuition would hide it; this makes it fail loudly."""
    falling = {f for f, (_q, _u, _v, d) in sev.PHYSICAL.items() if d == "down"}
    assert falling == {"fog_camera", "fog_lidar", "beams_reducing", "darkness"}


# ── the Group B calibration, which is what the paper must report ───────────

def test_temporal_misalignment_is_frame_shift_not_probability():
    """Group B. The corruption table specifies a frozen-frame probability;
    this injector has no probability parameter and implements N-step
    staleness instead. Levels anchor to 100/200/400 ms at 10 Hz."""
    assert sev.GROUP["temporal_misalignment"] == "B"
    for level, expected_frames in ((1, 1), (2, 2), (3, 4)):
        kw = sev.SEVERITY["temporal_misalignment"][level]
        assert "mu_delay" in kw and "sigma_jitter" in kw
        # jitter MUST be zero or the shift is redrawn per frame and the level
        # stops meaning a fixed number of frames
        assert kw["sigma_jitter"] == 0.0
        shift = round(kw["mu_delay"] * kw["fps"])
        assert shift == expected_frames, (
            f"level {level}: mu_delay={kw['mu_delay']}s at {kw['fps']}Hz "
            f"gives {shift} frames, expected {expected_frames}")


def test_level_3_lands_on_coras_maximum_latency():
    """The 1-2-4 spacing is deliberate, not a typo for 1-2-3."""
    kw = sev.SEVERITY["temporal_misalignment"][3]
    assert kw["mu_delay"] * 1000 == 400.0


def test_spatial_misalignment_uses_the_two_real_sigmas():
    assert sev.GROUP["spatial_misalignment"] == "B"
    for level, deg in ((1, 1.0), (2, 2.0), (3, 3.0)):
        kw = sev.SEVERITY["spatial_misalignment"][level]
        assert kw["sigma_heading"] == deg
        assert "sigma_xy" in kw
        # no probability parameter exists on PoseErrorInjector
        assert not any(k.startswith("p") for k in kw)


# ── the Group A2 trap: severity is an INDEX, not the quantity ──────────────

@pytest.mark.parametrize("fault,quantity_at_level_1", [
    ("fog_camera", 300), ("fog_lidar", 300), ("snow_camera", 5),
    ("snow_lidar", 5), ("darkness", 25), ("brightness", 0.5),
    ("points_reducing", 70), ("beams_reducing", 16),
])
def test_severity_kwarg_is_the_level_not_the_physical_value(
        fault, quantity_at_level_1):
    """Passing the table's physical value into `severity` raises. This test
    documents that the kwarg is 1/2/3 and the quantity lives in PHYSICAL,
    which is the single easiest mistake to make with this table."""
    assert sev.SEVERITY[fault][1] == {"severity": 1}
    assert sev.PHYSICAL[fault][2][1] == quantity_at_level_1
