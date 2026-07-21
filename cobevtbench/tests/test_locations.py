"""
Tests for the observation-location registry.

A registry is only worth having if it cannot drift from the code. The
cross-check tests below are the ones that matter: they run a real model and
compare what it *actually* emits against what is *declared*. Without them the
registry becomes documentation, and stale documentation about injection
points is worse than none -- a taps config validated against it would still
silently record nothing.
"""

from __future__ import annotations

import pytest
import torch

from cobevtbench.fusion.fusebevt import FuseBEVT
from cobevtbench.observation.locations import (LOCATIONS, _template,
                                               all_locations,
                                               validate_location)
from cpbench.observation import StatsTap, TapSet


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


def test_template_expansion_respects_depth() -> None:
    names = all_locations(depth=2, n_blocks=1)
    assert "fusebevt/d0/local/softmax" in names
    assert "fusebevt/d1/local/softmax" in names
    assert "fusebevt/d2/local/softmax" not in names
    assert "sinbevt/b0/key" in names
    assert "sinbevt/b1/key" not in names


def test_track_filter_separates_the_two_pipelines() -> None:
    """A camera-track tap config matching nothing on a LiDAR run looks
    exactly like a broken tap, so the tracks are declared, not inferred."""
    camera = set(all_locations(track="camera"))
    lidar = set(all_locations(track="lidar"))
    assert "head/seg_logits" in camera and "head/seg_logits" not in lidar
    assert "head/cls_logits" in lidar and "head/cls_logits" not in camera
    # FuseBEVT is shared, which is the whole point of writing it once.
    assert "fusebevt/output" in camera & lidar


def test_forward_pass_ordering_is_preserved() -> None:
    """all_locations claims forward-pass order; layer-wise robustness plots
    read it directly, so a reordering would mislabel the x-axis."""
    names = all_locations()
    positions = {name: i for i, name in enumerate(names)}
    assert positions["input/agent_mask"] < positions["fusebevt/input"]
    assert positions["fusebevt/input"] < positions["fusebevt/pooled"]
    assert positions["fusebevt/pooled"] < positions["fusebevt/output"]
    assert positions["fusebevt/output"] < positions["head/seg_logits"]
    assert positions["fusebevt/d0/local/scores"] < positions["fusebevt/d0/local/softmax"]


# ------------------------------------------------------------- validation --

def test_validate_accepts_concrete_and_template_forms() -> None:
    concrete = validate_location("fusebevt/d2/local/softmax")
    template = validate_location("fusebevt/d{d}/local/softmax")
    assert concrete is template


def test_unknown_location_raises_with_neighbours_listed() -> None:
    """This fires on a config the user wrote; the message has to be enough to
    fix it without opening the source."""
    with pytest.raises(KeyError) as excinfo:
        validate_location("fusebevt/d0/local/attn_weights")
    message = str(excinfo.value)
    assert "unknown observation location" in message
    assert "fusebevt/d{d}/local/softmax" in message


def test_unknown_layer_falls_back_to_listing_everything() -> None:
    with pytest.raises(KeyError, match="unknown observation location"):
        validate_location("nonsense/tensor")


def test_template_normalisation() -> None:
    assert _template("fusebevt/d11/local/softmax") == "fusebevt/d{d}/local/softmax"
    assert _template("sinbevt/b0/local/q") == "sinbevt/b{i}/local/q"
    assert _template("fusebevt/output") == "fusebevt/output"


# --------------------------------------------------- registry vs. reality --

def _emitted_by_fusebevt(depth: int = 2) -> set:
    """Every location a real FuseBEVT forward pass actually emits."""
    tap = StatsTap()
    model = FuseBEVT(dim=32, mlp_dim=64, agent_size=3, window_size=4,
                     dim_head=8, depth=depth).eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 8, 8),
              mask=torch.ones(1, 3, dtype=torch.bool),
              taps=TapSet([tap], strict=True))
    return {record.location for record in tap.records}


def test_every_emitted_location_is_registered() -> None:
    """Catches a tensor that was tapped but never declared -- invisible to
    config validation and to anyone reading the registry to find out what
    they can inject into."""
    unregistered = sorted(
        name for name in _emitted_by_fusebevt()
        if _template(name) not in LOCATIONS)
    assert not unregistered, (
        "emitted but not in the registry:\n  " + "\n  ".join(unregistered))


def test_every_registered_fusebevt_location_is_emitted() -> None:
    """The other direction: a declared location that nothing emits is a
    promise the package does not keep. A taps config naming it would validate
    cleanly and then record nothing.

    Scoped to fusebevt/ because that is what exists so far; the camera and
    LiDAR sections are checked by their own steps as they land.
    """
    emitted = {_template(name) for name in _emitted_by_fusebevt()}
    declared = {name for name in LOCATIONS
                if name.startswith("fusebevt/")
                and name != "fusebevt/roi_mask"}   # emitted by ROICavMask, step 8
    missing = sorted(declared - emitted)
    assert not missing, (
        "registered but never emitted:\n  " + "\n  ".join(missing))


def test_registered_modules_match_the_emitting_class() -> None:
    """The registry names which nn.Module emits each tensor. If that drifts,
    'which layer failed' answers point at the wrong code."""
    tap = StatsTap()
    model = FuseBEVT(dim=32, mlp_dim=64, agent_size=2, window_size=4,
                     dim_head=8, depth=1).eval()
    with torch.no_grad():
        model(torch.randn(1, 2, 32, 8, 8),
              mask=torch.ones(1, 2, dtype=torch.bool),
              taps=TapSet([tap], strict=True))
    for record in tap.records:
        declared = validate_location(record.location)
        assert record.module in declared.emitters(), (
            f"{record.location}: registry says {declared.module}, "
            f"emitted by {record.module}")
