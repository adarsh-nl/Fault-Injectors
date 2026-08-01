"""
Tests for the Stage-1 encoder contract and the LiDAR implementation.

Two things are being pinned here. The first is the contract itself: the
package's claim that a camera track costs one encoder rather than a second
model rests entirely on every encoder returning the same ``(N, D, H, W)``, so
the shape check has to be enforced rather than assumed. The second is the
geometry validation, which guards two mistakes whose symptoms appear several
modules away from their cause.

The registry-vs-reality cross-check -- does the LiDAR encoder emit exactly the
locations it declares, and exactly once -- lives here too, because this is the
first step that produces a real forward pass to check against.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from cpbench.data import GridSpec
from cpbench.observation import StatsTap, TapSet
from w2cbench.models import LidarPillarEncoder, ObservationEncoder
from w2cbench.observation import validate_location


def _spec(voxel: float = 0.8, extent: float = 25.6,
          downsample: int = 2) -> GridSpec:
    """A small but structurally faithful grid: 64x64 pillars, 32x32 features."""
    return GridSpec(voxel_size=(voxel, voxel),
                    point_range=(-extent, -extent, -3.0, extent, extent, 1.0),
                    downsample=downsample)


def _batch(n_agents: int = 2, n_pillars: int = 6) -> dict:
    coords = torch.stack([
        torch.arange(n_pillars) % n_agents,
        torch.arange(n_pillars) % 8,
        torch.arange(n_pillars) % 8,
    ], dim=1)
    return {"features": torch.randn(n_pillars, 8, 10),
            "coords": coords,
            "num_points": torch.full((n_pillars,), 8),
            "record_len": [n_agents]}


# ---------------------------------------------------------------- contract --

class _Constant(ObservationEncoder):
    """Minimal conforming encoder; stands in for the camera track (step 15)."""

    def forward(self, batch, taps=None):
        n = self.total_agents(batch)
        return self.validate_output(
            torch.zeros(n, self.out_channels, *self.feature_hw))


def test_declared_shape_is_available_before_any_forward_pass() -> None:
    """The confidence head, the selection mask and the anchor grid are all
    sized at construction, so the encoder has to declare rather than
    discover."""
    enc = LidarPillarEncoder(_spec(), out_channels=32)
    assert enc.out_channels == 32
    assert enc.feature_hw == (32, 32)


def test_output_is_flat_over_agents_not_padded() -> None:
    """(N, D, H, W), not (B, L, D, H, W). Padding to max_cav is the
    orchestrator's job: inventing empty agents here would feed zero-feature
    maps to the confidence generator, which would produce confidence maps for
    agents that do not exist and select against them."""
    enc = LidarPillarEncoder(_spec(), out_channels=32).eval()
    with torch.no_grad():
        out = enc(_batch(n_agents=3))
    assert out.shape == (3, 32, 32, 32)


def test_total_agents_comes_from_record_len_not_from_the_sensor_tensors() -> None:
    """An agent whose LiDAR was corrupted to zero points still occupies a row.
    Counting rows from the pillar coordinates would silently drop it, turning
    a sensor fault into an agent-dropout fault -- and making the two
    conditions indistinguishable in the results."""
    enc = LidarPillarEncoder(_spec(), out_channels=32).eval()
    batch = _batch(n_agents=3)
    batch["coords"][:, 0] = 0                 # every pillar belongs to agent 0
    with torch.no_grad():
        out = enc(batch)
    assert out.shape[0] == 3                  # still three agents

    # Agents 1 and 2 contributed no points, so their maps are identically
    # zero -- every conv is bias-free and BatchNorm maps 0 to 0. That is the
    # honest encoding of "this agent saw nothing", and it is what lets the
    # confidence generator downstream report low confidence for them rather
    # than never being asked about them at all.
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[2]) == 0
    assert torch.count_nonzero(out[0]) > 0


def test_missing_record_len_names_the_key_and_the_reason() -> None:
    enc = LidarPillarEncoder(_spec(), out_channels=32)
    with pytest.raises(KeyError, match="record_len"):
        enc.total_agents({"features": torch.zeros(1, 8, 10)})


def test_no_points_at_all_still_returns_one_map_per_agent() -> None:
    """A frame in which every collaborator was dropped or fully occluded must
    not change the output rank; downstream stages index the agent axis."""
    enc = LidarPillarEncoder(_spec(), out_channels=32).eval()
    empty = {"features": torch.zeros(0, 8, 10),
             "coords": torch.zeros(0, 3, dtype=torch.long),
             "num_points": torch.zeros(0, dtype=torch.long),
             "record_len": [2]}
    with torch.no_grad():
        out = enc(empty)
    assert out.shape == (2, 32, 32, 32)


# -------------------------------------------------------------- validation --

def test_wrong_grid_names_the_encoder_and_both_shapes() -> None:
    """This fires several modules before the error would otherwise surface,
    and the message has to be enough to fix it without a debugger."""
    class Wrong(ObservationEncoder):
        def forward(self, batch, taps=None):
            return self.validate_output(torch.zeros(1, 8, 5, 4))

    with pytest.raises(ValueError) as excinfo:
        Wrong(out_channels=8, feature_hw=(4, 4))({"record_len": [1]})
    message = str(excinfo.value)
    assert "Wrong" in message and "(5, 4)" in message and "(4, 4)" in message


def test_wrong_channel_count_is_caught() -> None:
    class Wrong(ObservationEncoder):
        def forward(self, batch, taps=None):
            return self.validate_output(torch.zeros(1, 7, 4, 4))

    with pytest.raises(ValueError, match="declares out_channels=8"):
        Wrong(out_channels=8, feature_hw=(4, 4))({"record_len": [1]})


def test_wrong_rank_is_caught() -> None:
    class Wrong(ObservationEncoder):
        def forward(self, batch, taps=None):
            return self.validate_output(torch.zeros(1, 8, 4))

    with pytest.raises(ValueError, match=r"must return \(N, D, H, W\)"):
        Wrong(out_channels=8, feature_hw=(4, 4))({"record_len": [1]})


def test_indivisible_pillar_grid_is_rejected_at_construction() -> None:
    """Otherwise it surfaces as a torch.cat size mismatch from inside the
    backbone, with nothing pointing at the point range in a YAML file."""
    odd = GridSpec(voxel_size=(0.8, 0.8),
                   point_range=(-20.0, -20.0, -3.0, 20.0, 20.0, 1.0))
    with pytest.raises(ValueError, match="stride product"):
        LidarPillarEncoder(odd, out_channels=32)


def test_downsample_must_match_the_first_block_stride() -> None:
    """The trap this check exists for: the backbone upsamples every pyramid
    level back to the FIRST level's resolution, so it produces
    grid_hw // block_strides[0] -- while the anchors, the warp and the
    selection mask are all sized from GridSpec.downsample. When those
    disagree nothing raises on its own; the symptom is a shape error modules
    later, or a silently mismatched anchor grid that lowers AP without ever
    failing."""
    with pytest.raises(ValueError, match="disagrees with"):
        LidarPillarEncoder(_spec(downsample=4), out_channels=32,
                           block_strides=(2, 2, 2))
    # ...and the consistent pairing is accepted
    LidarPillarEncoder(_spec(downsample=4), out_channels=32,
                       block_strides=(4, 2, 2))


def test_declared_feature_hw_matches_what_the_backbone_produces() -> None:
    """The validation above is only worth having if the declaration it
    protects is itself correct, so this checks the two against a real forward
    pass rather than against each other."""
    for downsample, strides in ((2, (2, 2, 2)), (4, (4, 2, 2))):
        enc = LidarPillarEncoder(_spec(downsample=downsample),
                                 out_channels=16, block_strides=strides).eval()
        with torch.no_grad():
            out = enc(_batch())
        assert tuple(out.shape[2:]) == enc.feature_hw == _spec(
            downsample=downsample).feature_hw


# ------------------------------------------------- registry vs. reality --

def _emitted(enc: LidarPillarEncoder) -> Counter:
    tap = StatsTap()
    with torch.no_grad():
        enc.eval()(_batch(), taps=TapSet([tap], strict=True))
    return Counter(record.location for record in tap.records)


def test_encoder_emits_exactly_the_registered_locations() -> None:
    counts = _emitted(LidarPillarEncoder(_spec(), out_channels=32))
    assert set(counts) == {"encoder/pillar_features", "encoder/scatter_bev",
                           "encoder/bev_features"}
    for name in counts:
        validate_location(name)


def test_bev_features_is_emitted_exactly_once() -> None:
    """The wrapper deliberately emits nothing of its own: the tensor's
    producer owns the emit, which on this track is cpbench's BEVBackbone. A
    second emit from the wrapper would put two records under one location in
    one forward pass, corrupting every per-location statistic and the
    clean-vs-faulted layer-wise join."""
    counts = _emitted(LidarPillarEncoder(_spec(), out_channels=32))
    assert counts["encoder/bev_features"] == 1


def test_registered_module_matches_the_emitting_class() -> None:
    """The registry names which nn.Module emits each tensor. If that drifts,
    'which layer failed' answers point at the wrong code."""
    tap = StatsTap()
    enc = LidarPillarEncoder(_spec(), out_channels=32).eval()
    with torch.no_grad():
        enc(_batch(), taps=TapSet([tap], strict=True))
    for record in tap.records:
        declared = validate_location(record.location)
        assert record.module in declared.emitters(), (
            f"{record.location}: registry says {declared.module}, "
            f"emitted by {record.module}")


def test_taps_none_is_the_default_and_costs_nothing() -> None:
    """Training runs with taps off; the hook must be a single is-None check
    and must not change the result."""
    enc = LidarPillarEncoder(_spec(), out_channels=32).eval()
    batch = _batch()
    with torch.no_grad():
        without = enc(batch)
        with_taps = enc(batch, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(without, with_taps)
