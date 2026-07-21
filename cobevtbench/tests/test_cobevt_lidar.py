"""
Tests for the LiDAR track end to end.

This is the first complete model in the package, so these tests cover the
seams between borrowed cpbench parts and CoBEVT's own fusion, and prove the
pipeline runs all the way to an AP number.

The AP value itself is meaningless here -- the model is untrained -- so
nothing asserts on it. What is asserted is that the plumbing from pillars to
scored boxes holds together, because that is what breaks silently.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cobevtbench.models.cobevt_lidar import CoBEVTLidar
from cobevtbench.observation.locations import _template, LOCATIONS
from cpbench.data import AnchorGenerator, BoxDecoder, GridSpec
from cpbench.metrics import DetectionEvaluator
from cpbench.observation import StatsTap, TapSet

# 64x64 pillars (divisible by the backbone's stride product of 8),
# downsample 2 -> a 32x32 feature grid, divisible by the FuseBEVT window 8.
SPEC = GridSpec(voxel_size=(0.8, 0.8),
                point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))


def _model(max_cav: int = 3, **kwargs) -> CoBEVTLidar:
    params = dict(max_cav=max_cav, encoder_out_channels=32, fuse_depth=1,
                  fuse_window=8, fuse_dim_head=8)
    params.update(kwargs)
    return CoBEVTLidar(SPEC, **params).eval()


def _batch(record_len=(2,), max_cav: int = 3, n_pillars: int = 40) -> dict:
    """A minimal but structurally valid pillar batch."""
    rng = np.random.default_rng(0)
    total_agents = sum(record_len)
    agent_idx = rng.integers(0, total_agents, n_pillars)
    rows = rng.integers(0, SPEC.grid_hw[0], n_pillars)
    cols = rng.integers(0, SPEC.grid_hw[1], n_pillars)
    return {
        "features": torch.randn(n_pillars, 8, 9),
        "coords": torch.tensor(np.stack([agent_idx, rows, cols], axis=1)),
        "num_points": torch.full((n_pillars,), 8),
        "record_len": list(record_len),
        "T_agent_to_ego": torch.eye(4).expand(
            len(record_len), max_cav, 4, 4).contiguous(),
    }


# ------------------------------------------------------------------ shapes --

def test_forward_produces_detection_maps() -> None:
    out = _model()(_batch())
    height, width = SPEC.feature_hw
    assert out["cls"].shape == (1, 2, height, width)
    assert out["reg"].shape == (1, 14, height, width)
    assert out["fused"].shape == (1, 32, height, width)


def test_batching_over_scenes_with_different_agent_counts() -> None:
    """Real cooperative data has 2-7 agents per scene. Getting record_len
    handling wrong assigns an agent to the wrong scene, which corrupts every
    number without raising."""
    out = _model()(_batch(record_len=(3, 1, 2)))
    assert out["cls"].shape[0] == 3
    assert out["agent_mask"].shape[:2] == (3, 3)


def test_agent_mask_reflects_the_record_lengths() -> None:
    out = _model(max_cav=4)(_batch(record_len=(2, 1), max_cav=4))
    present = out["agent_mask"].any(dim=-1).any(dim=-1)
    assert present.tolist() == [[True, True, False, False],
                               [True, False, False, False]]


def test_output_is_finite() -> None:
    assert torch.isfinite(_model()(_batch())["cls"]).all()


# -------------------------------------------------------- eager validation --

def test_indivisible_pillar_grid_raises_at_construction() -> None:
    """The failure this replaces is a torch.cat size mismatch from inside the
    backbone, which says nothing about the point_range that caused it."""
    bad = GridSpec(voxel_size=(0.8, 0.8),
                   point_range=(-20.0, -20.0, -3.0, 20.0, 20.0, 1.0))  # 50x50
    with pytest.raises(ValueError) as excinfo:
        CoBEVTLidar(bad, max_cav=2, encoder_out_channels=32)
    message = str(excinfo.value)
    assert "50x50" in message and "stride product 8" in message


def test_indivisible_feature_grid_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="does not divide by the FuseBEVT window"):
        CoBEVTLidar(SPEC, max_cav=2, encoder_out_channels=32, fuse_window=12)


def test_indivisible_head_dim_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="not divisible by fuse_dim_head"):
        CoBEVTLidar(SPEC, max_cav=2, encoder_out_channels=30, fuse_dim_head=8,
                    fuse_window=8)


# ----------------------------------------------------------- architecture --

def test_no_projection_when_widths_already_match() -> None:
    """The paper describes no projection between the encoder and FuseBEVT.
    Adding one unconditionally would be extra parameters it never mentions."""
    assert isinstance(_model().project, torch.nn.Identity)
    projected = _model(fuse_dim=64)
    assert isinstance(projected.project, torch.nn.Conv2d)


def test_fusebevt_is_the_same_class_the_camera_track_uses() -> None:
    """Table 2's claim is that the *same* fusion module works on LiDAR. A
    LiDAR-specific reimplementation would quietly invalidate that."""
    from cobevtbench.fusion.fusebevt import FuseBEVT
    assert isinstance(_model().fuse, FuseBEVT)


def test_gradients_reach_every_parameter() -> None:
    model = CoBEVTLidar(SPEC, max_cav=2, encoder_out_channels=32, fuse_depth=1,
                        fuse_window=8, fuse_dim_head=8)
    out = model(_batch(record_len=(2,), max_cav=2))
    (out["cls"].sum() + out["reg"].sum()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters never used in the forward pass: {missing}"


# ------------------------------------------------------- the AP pipeline --

def test_predictions_decode_into_scored_boxes_and_score() -> None:
    """The gate for this step: pillars in, AP out.

    The model is untrained, so the AP value carries no information and
    nothing asserts on it. What this proves is that the anchor layout, the
    head's channel ordering and the decoder's expectations all agree -- a
    mismatch there produces boxes in the wrong place and an AP that is
    quietly always zero.
    """
    model = _model()
    out = model(_batch())

    anchors = AnchorGenerator(SPEC)
    decoder = BoxDecoder(anchors, score_threshold=0.0, max_boxes=50)
    boxes, scores = decoder(out["cls"][0], out["reg"][0])

    assert boxes.ndim == 2 and boxes.shape[1] == 7
    assert scores.shape == (len(boxes),)
    assert np.isfinite(boxes).all() and np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()

    gt = np.array([[1.0, 2.0, -1.0, 3.9, 1.6, 1.56, 0.0]], dtype=np.float32)
    evaluator = DetectionEvaluator(iou_thresholds=(0.5, 0.7))
    evaluator.add_frame(boxes, scores, gt)
    metrics = evaluator.compute()

    for key in ("ap50", "ap70", "precision50", "recall70", "f1_50"):
        assert key in metrics and np.isfinite(metrics[key])
    assert metrics["n_frames"] == 1


def test_anchor_count_matches_the_head() -> None:
    """A head built for a different anchor count than the decoder assumes
    reshapes without error and scatters boxes across the map."""
    anchors = AnchorGenerator(SPEC)
    model = _model(num_anchors=anchors.num_anchors_per_cell)
    out = model(_batch())
    assert out["cls"].shape[1] == anchors.num_anchors_per_cell
    assert out["reg"].shape[1] == anchors.num_anchors_per_cell * 7


# ------------------------------------------------------------------ taps --

def test_forward_is_identical_with_and_without_taps() -> None:
    """The measurement-plane invariant on a complete model."""
    model = _model()
    batch = _batch()
    with torch.no_grad():
        plain = model(batch)
        tapped = model(batch, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(plain["cls"], tapped["cls"])
    assert torch.equal(plain["reg"], tapped["reg"])


def test_every_emitted_location_is_registered() -> None:
    """Extends the registry cross-check from FuseBEVT alone to the whole
    LiDAR track, now that encoder and head locations are reachable."""
    tap = StatsTap()
    with torch.no_grad():
        _model()(_batch(), taps=TapSet([tap], strict=True))
    unregistered = sorted(
        {r.location for r in tap.records if _template(r.location) not in LOCATIONS})
    assert not unregistered, (
        "emitted but not in the registry:\n  " + "\n  ".join(unregistered))


def test_registered_lidar_locations_are_all_reachable() -> None:
    """The other direction, scoped to the layers this track exercises. A
    declared location nothing emits is a promise the package does not keep.
    """
    tap = StatsTap()
    with torch.no_grad():
        _model()(_batch(), taps=TapSet([tap], strict=True))
    emitted = {_template(r.location) for r in tap.records}
    declared = {
        name for name, loc in LOCATIONS.items()
        if loc.track in ("lidar", "both")
        and name.split("/")[0] in ("input", "encoder", "regroup", "sttf", "head")
        and not (loc.track == "both" and name.startswith("input/images"))
    }
    # compress/ and comm/ are wired in step 8; input/intrinsics is camera-only.
    missing = sorted(declared - emitted)
    assert not missing, "registered but never emitted:\n  " + "\n  ".join(missing)


def test_module_names_match_the_registry() -> None:
    tap = StatsTap()
    with torch.no_grad():
        _model()(_batch(), taps=TapSet([tap], strict=True))
    from cobevtbench.observation.locations import validate_location
    for record in tap.records:
        declared = validate_location(record.location)
        assert record.module in declared.emitters(), (
            f"{record.location}: registry says {declared.module}, "
            f"emitted by {record.module}")
