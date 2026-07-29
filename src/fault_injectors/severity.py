"""
severity.py
-----------
Three-level severity scale for the fault injectors, and a `make()` factory.

THE DISTINCTION THIS FILE EXISTS TO MAKE EXPLICIT
=================================================
Faults fall into two groups, and the difference belongs in the paper:

GROUP A -- TRANSCRIBED from the corruption table (MultiCorrupt, Hahner et al.,
    arXiv:2402.11677 Table I). Our levels 1/2/3 ARE their levels 1/2/3, so
    results are directly comparable to published MultiCorrupt numbers.

    Two sub-cases, and confusing them is the likeliest bug in this file:

    A1  the injector takes the PHYSICAL QUANTITY directly
        -> pass the table's value  (missing_camera: p_drop_rgb = 0.2)

    A2  the injector takes `severity` -- an INDEX in {1,2,3}, not the quantity
        -> pass the LEVEL, never the value. `FogInjector(severity=300)` raises
           ValueError; `FogInjector(severity=1)` IS the 300 m visibility case.
           MultiCorrupt already calibrated level -> quantity internally, so
           passing the level is faithful BY CONSTRUCTION and no conversion by
           us is wanted. Doing our own visibility->extinction (Koschmieder)
           conversion would CREATE divergence from MultiCorrupt, not remove it:
           their alpha/beta constants already encode that calibration.

GROUP B -- INJECTOR-CALIBRATED. The corruption table's quantity has no
    corresponding parameter, so the severity scale is ours, not theirs.
    Results for these faults are NOT comparable to MultiCorrupt's numbers and
    must be reported as such.

      temporal_misalignment : the table specifies a frozen-frame PROBABILITY.
          The injector implements N-step staleness (a frame SHIFT) and has no
          probability parameter at all. Anchored instead to the millisecond
          latency range V2X-ViT and CoRA report.
      spatial_misalignment  : the table gives "X deg, p=Y". The degrees map
          cleanly to sigma_heading; `p` has no parameter -- see its comment.

WHAT IS NOT HERE
================
motion_blur has a correct level->sigma_t mapping (recorded in PHYSICAL below)
but NO injector class wraps it, so it cannot be constructed. See UNAVAILABLE.

DIRECTION
=========
Severity rising does NOT mean "the number goes up". Three physical quantities
FALL as the fault gets worse -- fog visibility (300 -> 50 m), beams kept
(16 -> 4), and darkness noise-s (25 -> 5) -- while probabilities, snowfall
rate, brightness, frame shift and pose sigma all rise. The authoritative
per-fault answer is the `direction` field of PHYSICAL, and the test asserts
it per fault rather than assuming one blanket direction.

Note also that for the Group A2 faults the CONSTRUCTOR kwarg is always the
level 1/2/3 and therefore always rises; only the physical quantity carries
the real direction. Asserting direction on the kwarg would pass vacuously.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Frame rate of OPV2V and V2XSet. Source: src/datasets/opv2v.py:96
# (`fps = 10.0`), inherited unchanged by V2XSetDataset, and matching all 18
# shipped dataset configs. 1 frame = 100 ms.
FPS = 10.0


# ── The severity table: SEVERITY[fault][level] -> constructor kwargs ────────

SEVERITY: Dict[str, Dict[int, Dict[str, Any]]] = {

    # ---- GROUP A1: physical quantity passed directly -----------------------

    # Corruption table: "Missing Camera -- drop probability 0.2 / 0.4 / 0.6".
    # MissingModalityInjector takes the probability itself, range [0, 1].
    "missing_camera": {
        1: {"p_drop_rgb": 0.2},
        2: {"p_drop_rgb": 0.4},
        3: {"p_drop_rgb": 0.6},
    },

    # ---- GROUP A2: `severity` is a LEVEL INDEX, not the quantity ------------
    # For every entry below, the table's physical value is what MultiCorrupt's
    # severity level MEANS internally; it is quoted from each injector's own
    # docstring in the comment. Passing the physical value would raise.

    # Corruption table: dropout 70 / 80 / 90 %.
    # PointsReductionInjector docstring: "severity 1/2/3 drops p = 70/80/90 %".
    "points_reducing": {1: {"severity": 1}, 2: {"severity": 2},
                        3: {"severity": 3}},

    # Corruption table: 16 / 8 / 4 beams.
    # BeamReductionInjector: _KEEP_FRAC = {1: 0.5, 2: 0.25, 3: 0.125}, i.e.
    # 16/8/4 of a 32-beam sensor.
    # NOTE this injector needs (N,5) with a ring-index column at index 4.
    # OPV2V/V2XSet clouds are (N,4) -- it raises there by design. The table
    # entry is still correct; the input requirement is separate.
    "beams_reducing": {1: {"severity": 1}, 2: {"severity": 2},
                       3: {"severity": 3}},

    # Corruption table: noise s = 25 / 12 / 5.
    # DarknessInjector docstring: "Poisson-Gaussian noise (s=25/12/5)".
    "darkness": {1: {"severity": 1}, 2: {"severity": 2}, 3: {"severity": 3}},

    # Corruption table: HSV V-channel add 0.5 / 0.6 / 0.7.
    # BrightnessInjector docstring: "adding s=0.5/0.6/0.7 to the V channel".
    "brightness": {1: {"severity": 1}, 2: {"severity": 2},
                   3: {"severity": 3}},

    # Corruption table: visibility 300 / 150 / 50 m.
    # FogInjector docstring: "visibility 300/150/50 m".
    # LidarFogInjector docstring: "Severity 1/2/3 -> (alpha,beta) =
    #   (0.02,0.008)/(0.03,0.008)/(0.06,0.05), i.e. the 300/150/50 m
    #   visibility levels of Table I".
    # Both modalities share one calibration, so both are level pass-through.
    # NO Koschmieder conversion is applied or wanted: MultiCorrupt already did
    # it, and redoing it here would diverge from their numbers.
    "fog_camera": {1: {"severity": 1}, 2: {"severity": 2},
                   3: {"severity": 3}},
    "fog_lidar": {1: {"severity": 1}, 2: {"severity": 2},
                  3: {"severity": 3}},

    # Corruption table: snowfall 5 / 35 / 70 mm/h.
    # SnowInjector docstring: "snowfall 5/35/70 mm/h".
    # LidarSnowInjector docstring: "severity 1/2/3 -> (snowfall_rate,
    #   terminal_velocity) = ... 5 / 35 / 70 mm/h equivalent".
    # WARNING: LidarSnowInjector runs ~1-3 MINUTES PER FRAME by its own
    # docstring, and generates particle files on first use per severity.
    # Table-only; do not put it in an interactive notebook.
    "snow_camera": {1: {"severity": 1}, 2: {"severity": 2},
                    3: {"severity": 3}},
    "snow_lidar": {1: {"severity": 1}, 2: {"severity": 2},
                   3: {"severity": 3}},

    # ---- GROUP B: injector-calibrated, NOT comparable to the table ----------

    # CALIBRATED, NOT TRANSCRIBED.
    # The corruption table specifies "frozen frame applied with probability
    # p = 0.2/0.4/0.6". TemporalMisalignmentInjector implements a different
    # (and more faithful) model: N-step staleness, pairing the CURRENT LiDAR
    # with an OLDER image, X~_k = (I_{k-dk}, P_k). It has NO probability
    # parameter -- every frame is shifted; only the magnitude varies.
    #
    # So severity is anchored to the millisecond latency range V2X-ViT and
    # CoRA report instead. mu_delay is in SECONDS; the shift is
    # dk = round(mu_delay * fps), so at 10 Hz:
    #     level 1 -> 0.1 s -> 1 frame -> 100 ms
    #     level 2 -> 0.2 s -> 2 frames -> 200 ms
    #     level 3 -> 0.4 s -> 4 frames -> 400 ms
    # The 1-2-4 spacing is INTENTIONAL, not a typo for 1-2-3: level 3 lands on
    # 400 ms, CoRA's tested maximum latency.
    #
    # sigma_jitter MUST be 0.0. With jitter > 0 the shift is redrawn per frame
    # from Normal(mu, sigma) and will not sit on the intended integer, so the
    # level would no longer mean a fixed number of frames.
    "temporal_misalignment": {
        1: {"mu_delay": 0.1, "sigma_jitter": 0.0, "fps": FPS},
        2: {"mu_delay": 0.2, "sigma_jitter": 0.0, "fps": FPS},
        3: {"mu_delay": 0.4, "sigma_jitter": 0.0, "fps": FPS},
    },

    # CALIBRATED, NOT TRANSCRIBED -- and one half needs confirmation.
    # The corruption table gives two numbers per level: "1 deg, p=0.2".
    #   * the DEGREES map cleanly onto sigma_heading (same unit) -- transcribed.
    #   * `p` has NO corresponding parameter. PoseErrorInjector exposes
    #     sigma_xy, sigma_heading, sigma_z, sigma_rollpitch, distribution,
    #     seed -- there is no probability knob anywhere in it.
    #
    # Read here as sigma_xy in METRES, which makes the pairs
    # (1 deg, 0.2 m) / (2 deg, 0.4 m) / (3 deg, 0.6 m) and matches the
    # V2X-ViT / CoBEVT pose-noise protocol the repo README cites
    # (sigma_xy 0-0.5 m, sigma_heading 0-1 deg). If `p` was meant as a literal
    # probability, this row is injector-calibrated and must be reported as
    # such -- UNCONFIRMED, flagged rather than silently converted.
    "spatial_misalignment": {
        1: {"sigma_heading": 1.0, "sigma_xy": 0.2},
        2: {"sigma_heading": 2.0, "sigma_xy": 0.4},
        3: {"sigma_heading": 3.0, "sigma_xy": 0.6},
    },
}


# ── Faults with a known scale but no constructible injector ────────────────

# motion_blur: the MultiCorrupt backend exists and its calibration matches the
# corruption table exactly -- _mc_image.py:77 has
#     s = [0.06, 0.1, 0.13][severity - 1]
# which is the table's sigma_t 0.06/0.10/0.13. But NO injector class wraps it
# (the package exports 16 injectors; motion blur is not among them), so it
# cannot be built without writing new injector code. Recorded here so the
# scale is not lost, and excluded from make().
UNAVAILABLE: Dict[str, str] = {
    "motion_blur": (
        "no MotionBlurInjector class exists. The MultiCorrupt backend "
        "function is present at _mc_image.py:77 with the correct "
        "severity -> sigma_t mapping (0.06/0.10/0.13), but nothing wraps it. "
        "Add a wrapper mirroring fog.py before using this fault."
    ),
}


# ── Physical quantity per level, and which way it moves ────────────────────
# The test asserts DIRECTION on these, not on the constructor kwargs: for the
# Group A2 faults the kwarg is always 1/2/3 (rising), which would make a
# blanket "value goes up" assertion pass while telling you nothing.

# fault -> (quantity, unit, {level: value}, direction)
#   direction "up"   = physically worse as the number RISES
#   direction "down" = physically worse as the number FALLS
PHYSICAL: Dict[str, tuple] = {
    "missing_camera":        ("drop probability", "",      {1: 0.2, 2: 0.4, 3: 0.6},    "up"),
    "points_reducing":       ("dropout",          "%",     {1: 70, 2: 80, 3: 90},       "up"),
    "beams_reducing":        ("beams kept",       "beams", {1: 16, 2: 8, 3: 4},         "down"),
    "darkness":              ("noise s",          "",      {1: 25, 2: 12, 3: 5},        "down"),
    "brightness":            ("HSV V add",        "",      {1: 0.5, 2: 0.6, 3: 0.7},    "up"),
    "fog_camera":            ("visibility",       "m",     {1: 300, 2: 150, 3: 50},     "down"),
    "fog_lidar":             ("visibility",       "m",     {1: 300, 2: 150, 3: 50},     "down"),
    "snow_camera":           ("snowfall rate",    "mm/h",  {1: 5, 2: 35, 3: 70},        "up"),
    "snow_lidar":            ("snowfall rate",    "mm/h",  {1: 5, 2: 35, 3: 70},        "up"),
    "temporal_misalignment": ("frame shift",      "frames",{1: 1, 2: 2, 3: 4},          "up"),
    "spatial_misalignment":  ("heading sigma",    "deg",   {1: 1.0, 2: 2.0, 3: 3.0},    "up"),
    "motion_blur":           ("sigma_t",          "",      {1: 0.06, 2: 0.10, 3: 0.13}, "up"),
}

# Which group each fault belongs to -- this is the distinction that goes in
# the paper, so it is data, not prose.
GROUP: Dict[str, str] = {
    "missing_camera": "A", "points_reducing": "A", "beams_reducing": "A",
    "darkness": "A", "brightness": "A", "fog_camera": "A",
    "fog_lidar": "A", "snow_camera": "A", "snow_lidar": "A",
    "motion_blur": "A",
    "temporal_misalignment": "B", "spatial_misalignment": "B",
}


# ── Factory ────────────────────────────────────────────────────────────────

def _injector_classes() -> Dict[str, Any]:
    """Imported lazily: the package __init__ pulls cv2/skimage backends."""
    from . import (AgentDropInjector, BeamReductionInjector,  # noqa: F401
                   BrightnessInjector, DarknessInjector, FogInjector,
                   LidarFogInjector, LidarSnowInjector,
                   MissingModalityInjector, PointsReductionInjector,
                   PoseErrorInjector, SnowInjector,
                   TemporalMisalignmentInjector)
    return {
        "missing_camera": MissingModalityInjector,
        "points_reducing": PointsReductionInjector,
        "beams_reducing": BeamReductionInjector,
        "darkness": DarknessInjector,
        "brightness": BrightnessInjector,
        "fog_camera": FogInjector,
        "fog_lidar": LidarFogInjector,
        "snow_camera": SnowInjector,
        "snow_lidar": LidarSnowInjector,
        "temporal_misalignment": TemporalMisalignmentInjector,
        "spatial_misalignment": PoseErrorInjector,
    }


def make(fault: str, level: int, seed: Optional[int] = None, **overrides):
    """Construct `fault` at severity `level` (1, 2 or 3).

    Parameters
    ----------
    fault     : key of SEVERITY.
    level     : 1, 2 or 3.
    seed      : passed through if given; severity does not set a seed, so the
                injector's own default applies otherwise.
    overrides : extra constructor kwargs, merged last.

    Raises
    ------
    KeyError          unknown fault, or a fault listed in UNAVAILABLE.
    ValueError        level not in {1, 2, 3}.
    """
    if fault in UNAVAILABLE:
        raise KeyError(f"{fault!r} is not constructible: {UNAVAILABLE[fault]}")
    if fault not in SEVERITY:
        raise KeyError(f"unknown fault {fault!r}; known: {sorted(SEVERITY)}")
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1, 2 or 3, got {level!r}")

    kwargs = dict(SEVERITY[fault][level])
    if seed is not None:
        kwargs["seed"] = seed
    kwargs.update(overrides)
    return _injector_classes()[fault](**kwargs)


def describe(fault: str) -> str:
    """One-line human summary: group, quantity, the three levels, direction."""
    quantity, unit, values, direction = PHYSICAL[fault]
    triple = " / ".join(str(values[i]) for i in (1, 2, 3))
    suffix = f" {unit}" if unit else ""
    tag = "transcribed" if GROUP[fault] == "A" else "INJECTOR-CALIBRATED"
    return (f"{fault:22s} [{GROUP[fault]}: {tag}]  {quantity} = "
            f"{triple}{suffix}  ({direction})")
