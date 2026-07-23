"""
Tests for the observation-location registry.

A registry is only worth having if it cannot drift from the code. The
cross-check tests -- run a real model, compare what it *actually* emits
against what is *declared*, in both directions -- are the ones that matter,
and they land with the model (``test_wire.py``). What can be checked before
any model exists is the registry's internal consistency, its template algebra
and its error messages, which is what this module covers.

Stale documentation about injection points is worse than none: a taps config
validated against a drifted registry passes and then records nothing.
"""

from __future__ import annotations

import pytest

from v2xvitbench.observation.locations import (LOCATIONS, _template,
                                               all_locations,
                                               validate_location)


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
        assert loc.track == "lidar", f"{name} has track={loc.track!r}"


def test_names_follow_the_layer_slash_tensor_convention() -> None:
    for name in LOCATIONS:
        assert name == name.lower(), f"{name} is not lowercase"
        assert " " not in name, f"{name} contains a space"
        assert "/" in name, f"{name} has no layer prefix"


def test_emitters_splits_alternatives() -> None:
    """The fused MSwin output is emitted by whichever branch-fusion method is
    configured; the registry lists the alternatives rather than naming one
    arbitrarily."""
    assert LOCATIONS["fusion/l{i}/mswin/out"].emitters() == [
        "PyramidWindowAttention", "SplitAttn"]


# ------------------------------------------------------------- templates --

def test_layer_template_expansion_respects_depth() -> None:
    """Depth is configurable (A2), so layer-indexed names are templates. An
    off-by-one here would make a taps config for a 3-layer run silently miss
    the final layer."""
    names = all_locations(depth=2)
    assert "fusion/l0/hmsa/softmax" in names
    assert "fusion/l1/hmsa/softmax" in names
    assert "fusion/l2/hmsa/softmax" not in names


def test_branch_template_expansion_respects_branch_count() -> None:
    names = all_locations(depth=1, branches=2)
    assert "fusion/l0/mswin/w0/out" in names
    assert "fusion/l0/mswin/w1/out" in names
    assert "fusion/l0/mswin/w2/out" not in names


def test_nested_templates_expand_to_depth_times_branches() -> None:
    """MSwin branch locations carry both indices; every (layer, branch) pair
    must appear or a taps config for the inner branches of a deep model
    silently matches nothing."""
    names = set(all_locations(depth=3, branches=3))
    for i in range(3):
        for j in range(3):
            assert f"fusion/l{i}/mswin/w{j}/softmax" in names


def test_forward_pass_ordering_is_preserved() -> None:
    """all_locations claims forward-pass order; layer-wise robustness plots
    read it directly, so a reordering would mislabel the x-axis."""
    positions = {name: i for i, name in enumerate(all_locations())}
    ordered = ["input/agent_mask", "input/prior_encoding",
               "encoder/pillar_features", "encoder/bev_features",
               "encoder/shrunk", "regroup/features", "rte/embedding",
               "sttf/after_warp", "fusion/l0/hmsa/softmax",
               "fusion/l0/mswin/out", "fusion/l0/ffn/out",
               "fusion/l0/output", "fusion/ego_features", "head/cls_logits"]
    for earlier, later in zip(ordered, ordered[1:]):
        assert positions[earlier] < positions[later], (
            f"{earlier} should precede {later}")


def test_the_two_robustness_mechanisms_are_registered() -> None:
    """The delay encoding and the heterogeneity routing are why this paper is
    benchmarked at all (package docstring); their observation points are
    pinned by name so neither can silently lose its tap."""
    assert "rte/embedding" in LOCATIONS
    assert "input/agent_types" in LOCATIONS
    assert "fusion/l{i}/hmsa/softmax" in LOCATIONS
    assert "input/time_delay" in LOCATIONS


def test_single_track_filter() -> None:
    """The track field exists for cross-paper join parity; V2X-ViT has no
    camera track, so filtering on one returns nothing rather than raising."""
    assert all_locations(track="camera") == []
    assert set(all_locations(track="lidar")) == set(all_locations())


# -------------------------------------------------------------- validation --

def test_validate_accepts_concrete_and_template_forms() -> None:
    concrete = validate_location("fusion/l2/hmsa/softmax")
    template = validate_location("fusion/l{i}/hmsa/softmax")
    assert concrete is template


def test_validate_accepts_doubly_indexed_names() -> None:
    concrete = validate_location("fusion/l1/mswin/w2/rel_pos_bias")
    template = validate_location("fusion/l{i}/mswin/w{j}/rel_pos_bias")
    assert concrete is template


def test_unknown_location_raises_with_neighbours_listed() -> None:
    """This fires on a config the user wrote; the message has to be enough to
    fix it without opening the source."""
    with pytest.raises(KeyError) as excinfo:
        validate_location("rte/embeddings")
    message = str(excinfo.value)
    assert "unknown observation location" in message
    assert "rte/embedding" in message


def test_unknown_layer_falls_back_to_listing_everything() -> None:
    with pytest.raises(KeyError, match="unknown observation location"):
        validate_location("nonsense/tensor")


def test_template_normalisation() -> None:
    assert _template("fusion/l11/output") == "fusion/l{i}/output"
    assert _template("fusion/l0/mswin/w2/q") == "fusion/l{i}/mswin/w{j}/q"
    assert _template("head/cls_logits") == "head/cls_logits"


def test_warp_and_weights_are_not_mistaken_for_branch_indices() -> None:
    """``sttf/before_warp`` and ``fusion/l0/mswin/weights`` contain the letter
    w without being branch-indexed; the normaliser keys off ``/w<digits>/``
    so the two cannot collide -- but the collision is exactly the kind that
    would be found late and painfully."""
    assert _template("sttf/before_warp") == "sttf/before_warp"
    assert _template("fusion/l0/mswin/weights") == "fusion/l{i}/mswin/weights"
    assert validate_location("sttf/before_warp").module == "SpatialTransform"
