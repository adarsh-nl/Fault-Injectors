"""
Tests for the observation-location registry.

A registry is only worth having if it cannot drift from the code. The
cross-check tests -- run a real model, compare what it *actually* emits
against what is *declared*, in both directions -- are the ones that matter,
and they land with the modules they check (implementation steps 3-9 of
``docs/where2comm_design.md`` section 13). What can be checked before any
model exists is the registry's internal consistency, its template algebra and
its error messages, which is what this module covers.

Stale documentation about injection points is worse than none: a taps config
validated against a drifted registry passes and then records nothing.
"""

from __future__ import annotations

import pytest

from w2cbench.observation.locations import (LOCATIONS, _template,
                                            all_locations, validate_location)


# ------------------------------------------------------------- the registry --

def test_names_are_unique() -> None:
    """Two locations sharing a name would silently overwrite each other in
    every analysis that joins on it. Import-time guarded, pinned here too."""
    assert len(LOCATIONS) == len({loc.name for loc in LOCATIONS.values()})


def test_every_location_is_documented() -> None:
    for name, loc in LOCATIONS.items():
        assert loc.module, f"{name} has no emitting module"
        assert loc.shape_hint, f"{name} has no shape hint"
        assert len(loc.description) > 15, f"{name} has a stub description"
        assert loc.track in ("camera", "lidar", "both"), (
            f"{name} has track={loc.track!r}")


def test_names_follow_the_layer_slash_tensor_convention() -> None:
    for name in LOCATIONS:
        assert name == name.lower(), f"{name} is not lowercase"
        assert " " not in name, f"{name} contains a space"
        assert "/" in name, f"{name} has no layer prefix"


def test_emitters_splits_alternatives() -> None:
    """Several locations are emitted by whichever aggregator is configured;
    the registry lists the alternatives rather than naming one arbitrarily."""
    assert LOCATIONS["fusion/r{k}/output"].emitters() == [
        "AttenFusion", "MaxFusion", "TransformerFusion"]
    assert LOCATIONS["encoder/bev_features"].emitters() == [
        "BEVBackbone", "CameraEncoder"]


# ------------------------------------------------------- templates and tracks --

def test_round_template_expansion_respects_k() -> None:
    """K is configurable (A3), so round-indexed names are templates. An
    off-by-one here would make a taps config for a 3-round run silently miss
    the final round."""
    names = all_locations(rounds=2)
    assert "comm/r0/selection_mask" in names
    assert "comm/r1/selection_mask" in names
    assert "comm/r2/selection_mask" not in names


def test_scale_template_expansion_respects_pyramid_depth() -> None:
    names = all_locations(rounds=1, n_scales=2)
    assert "backbone/feat_s0" in names
    assert "backbone/feat_s1" in names
    assert "backbone/feat_s2" not in names


def test_only_the_encoder_is_track_specific() -> None:
    """The package's central structural claim, as far as the registry can
    check it: everything from the confidence generator on is shared, so a
    camera track costs one encoder rather than a second model.

    The forward-pass check of the same claim is test_track_parity.py, which
    lands with the camera encoder (step 15).
    """
    camera = set(all_locations(track="camera"))
    lidar = set(all_locations(track="lidar"))

    assert "input/points" in lidar and "input/points" not in camera
    assert "input/intrinsics" in camera and "input/intrinsics" not in lidar
    assert "lift/depth_distribution" in camera - lidar

    shared = camera & lidar
    for name in ("encoder/bev_features", "confidence/r0/map",
                 "comm/r0/selection_mask", "comm/r0/bytes",
                 "fusion/r0/softmax", "head/cls_logits"):
        assert name in shared, f"{name} should reach both tracks"


def test_every_post_encoder_location_is_declared_for_both_tracks() -> None:
    """Stronger than the spot-checks above: no location outside the encoder
    layers may be track-specific, or the claim quietly stops holding as the
    registry grows."""
    encoder_layers = ("input/", "encoder/", "backbone/", "lift/")
    offenders = sorted(
        name for name, loc in LOCATIONS.items()
        if loc.track != "both" and not name.startswith(encoder_layers))
    assert not offenders, (
        "track-specific outside the encoder:\n  " + "\n  ".join(offenders))


def test_forward_pass_ordering_is_preserved() -> None:
    """all_locations claims forward-pass order; layer-wise robustness plots
    read it directly, so a reordering would mislabel the x-axis."""
    positions = {name: i for i, name in enumerate(all_locations())}
    ordered = ["input/agent_mask", "encoder/bev_features", "confidence/r0/map",
               "comm/r0/request_map", "comm/r0/selection_mask", "comm/r0/bytes",
               "align/r0/after_warp", "fusion/r0/softmax", "fusion/r0/output",
               "head/cls_logits"]
    for earlier, later in zip(ordered, ordered[1:]):
        assert positions[earlier] < positions[later], (
            f"{earlier} should precede {later}")


def test_the_causal_chain_this_package_exists_for_is_registered() -> None:
    """confidence -> selection -> message size -> bytes: the chain from a
    physical fault to a bandwidth number (design doc section 5.3). If any link
    loses its tap the package cannot demonstrate its central claim, so the
    chain is pinned by name and by order."""
    chain = ["confidence/r{k}/map", "comm/r{k}/selection_mask",
             "comm/r{k}/selected_count", "comm/r{k}/bytes"]
    assert all(name in LOCATIONS for name in chain)
    positions = {name: i for i, name in enumerate(all_locations())}
    concrete = [name.replace("{k}", "0") for name in chain]
    for earlier, later in zip(concrete, concrete[1:]):
        assert positions[earlier] < positions[later]


# -------------------------------------------------------------- validation --

def test_validate_accepts_concrete_and_template_forms() -> None:
    concrete = validate_location("comm/r2/selection_mask")
    template = validate_location("comm/r{k}/selection_mask")
    assert concrete is template


def test_unknown_location_raises_with_neighbours_listed() -> None:
    """This fires on a config the user wrote; the message has to be enough to
    fix it without opening the source."""
    with pytest.raises(KeyError) as excinfo:
        validate_location("comm/r0/selection")
    message = str(excinfo.value)
    assert "unknown observation location" in message
    assert "comm/r{k}/selection_mask" in message


def test_unknown_layer_falls_back_to_listing_everything() -> None:
    with pytest.raises(KeyError, match="unknown observation location"):
        validate_location("nonsense/tensor")


def test_template_normalisation() -> None:
    assert _template("comm/r11/bytes") == "comm/r{k}/bytes"
    assert _template("fusion/r0/softmax") == "fusion/r{k}/softmax"
    assert _template("backbone/feat_s2") == "backbone/feat_s{i}"
    assert _template("head/cls_logits") == "head/cls_logits"


def test_attention_key_location_is_not_mistaken_for_a_round_index() -> None:
    """`fusion/r{k}/k` is the attention key, and `{k}` is the round template.
    The normaliser keys off `/r<digits>/`, so the two cannot collide -- but
    the collision is exactly the kind that would be found late and painfully.
    """
    assert _template("fusion/r3/k") == "fusion/r{k}/k"
    assert validate_location("fusion/r3/k").module == "MultiHeadAttention"
