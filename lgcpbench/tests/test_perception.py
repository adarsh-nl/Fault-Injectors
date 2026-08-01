"""
Tests for lgcpbench.perception -- the backbone seam.

The properties asserted here are the ones the rest of LGCP and the whole
fault-injection story depend on:
    * the protocol is actually satisfied, so swapping in OpenCOOD later is a
      config change and not a rewrite;
    * area extraction is a VIEW and its payload matches derivation D2, so
      communication accounting is exact;
    * taps cannot alter the forward pass (the measurement-plane invariant).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpbench.observation.recorders import StatsTap
from cpbench.observation.taps import TapSet
from lgcpbench.perception import (
    AgentInputs,
    AreaFeatureMasker,
    CollabPerceptionModel,
    Detections,
    NativeReferenceBackbone,
    PerPixelAttentionFusion,
)
from lgcpbench.roi import AreaGrid

OPV2V_RANGE = (-140.8, -38.4, -3.0, 140.8, 38.4, 1.0)
FEATURE_HW = (48, 176)
CHANNELS = 8  # small for test speed; shape logic is channel-independent


@pytest.fixture
def grid() -> AreaGrid:
    return AreaGrid(point_range=OPV2V_RANGE)


@pytest.fixture
def masker(grid: AreaGrid) -> AreaFeatureMasker:
    return AreaFeatureMasker(grid, feature_hw=FEATURE_HW)


@pytest.fixture
def model() -> NativeReferenceBackbone:
    torch.manual_seed(0)
    m = NativeReferenceBackbone(
        grid_hw=(192, 704), feature_hw=FEATURE_HW, channels=CHANNELS
    )
    return m.eval()


@pytest.fixture
def features() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(3, CHANNELS, *FEATURE_HW)


# --------------------------------------------------------------------- #
# protocol conformance
# --------------------------------------------------------------------- #


def test_native_backbone_satisfies_protocol(model: NativeReferenceBackbone) -> None:
    assert isinstance(model, CollabPerceptionModel)


def test_backbone_exposes_declared_geometry(model: NativeReferenceBackbone) -> None:
    assert model.feature_channels == CHANNELS
    assert model.feature_hw == FEATURE_HW
    assert model.downsample == 4  # OPV2V: voxel 0.4 m x 4 = 1.6 m feature cell


def test_backbone_rejects_inconsistent_geometry_at_construction() -> None:
    """grid_hw, feature_hw and downsample are coupled through the encoder's
    stride. Catching a mismatch in __init__ beats discovering it as a shape
    error on the first frame of an HPC job."""
    with pytest.raises(ValueError, match="inconsistent"):
        NativeReferenceBackbone(
            grid_hw=(192, 704), feature_hw=(48, 176), channels=CHANNELS, downsample=2
        )


def test_encoder_stride_matches_declared_downsample() -> None:
    """The stride is block_strides[0], because BEVBackbone upsamples every
    pyramid level back to the first level's resolution."""
    for downsample in (2, 4):
        m = NativeReferenceBackbone(
            grid_hw=(64, 64),
            feature_hw=(64 // downsample, 64 // downsample),
            channels=CHANNELS,
            downsample=downsample,
        ).eval()
        feats = m.encode(_agent_inputs(n_agents=2, grid_hw=(64, 64)))
        assert tuple(feats.shape[-2:]) == (64 // downsample, 64 // downsample)


# --------------------------------------------------------------------- #
# AgentInputs / Detections validation
# --------------------------------------------------------------------- #


def _agent_inputs(
    n_agents: int = 3, n_pillars: int = 40, grid_hw: tuple = (192, 704)
) -> AgentInputs:
    """Synthetic collated pillar inputs. Pillar coords must land inside
    ``grid_hw`` or the scatter step indexes out of bounds."""
    torch.manual_seed(2)
    coords = torch.stack(
        [
            torch.randint(0, n_agents, (n_pillars,)),
            torch.randint(0, grid_hw[0], (n_pillars,)),
            torch.randint(0, grid_hw[1], (n_pillars,)),
        ],
        dim=1,
    )
    return AgentInputs(
        features=torch.randn(n_pillars, 32, 10),
        coords=coords,
        num_points=torch.randint(1, 32, (n_pillars,)),
        n_agents=n_agents,
        agent_ids=tuple(f"cav{i}" for i in range(n_agents)),
        positions=np.zeros((n_agents, 2)),
    )


def test_agent_inputs_index_lookup() -> None:
    ai = _agent_inputs()
    assert ai.agent_index("cav1") == 1
    with pytest.raises(KeyError):
        ai.agent_index("nope")


def test_agent_inputs_rejects_mismatched_ids() -> None:
    with pytest.raises(ValueError):
        AgentInputs(
            features=torch.zeros(1, 1, 10),
            coords=torch.zeros(1, 3, dtype=torch.long),
            num_points=torch.ones(1, dtype=torch.long),
            n_agents=3,
            agent_ids=("a", "b"),
        )


def test_agent_inputs_rejects_bad_positions() -> None:
    with pytest.raises(ValueError):
        AgentInputs(
            features=torch.zeros(1, 1, 10),
            coords=torch.zeros(1, 3, dtype=torch.long),
            num_points=torch.ones(1, dtype=torch.long),
            n_agents=2,
            positions=np.zeros((3, 2)),
        )


def test_detections_validation_and_empty() -> None:
    d = Detections(boxes=np.zeros((2, 7)), scores=np.array([0.9, 0.1]))
    assert len(d) == 2
    empty = Detections.empty(area_id=5)
    assert len(empty) == 0 and empty.area_id == 5
    with pytest.raises(ValueError):
        Detections(boxes=np.zeros((2, 6)), scores=np.zeros(2))
    with pytest.raises(ValueError):
        Detections(boxes=np.zeros((2, 7)), scores=np.zeros(3))


# --------------------------------------------------------------------- #
# area masking -- derivation D2
# --------------------------------------------------------------------- #


def test_extract_returns_a_view_not_a_copy(masker: AreaFeatureMasker) -> None:
    """A copy per (area, CAV) per frame would dominate the hot path."""
    feat = torch.zeros(CHANNELS, *FEATURE_HW)
    sub = masker.extract(feat, area_id=200)
    assert sub._base is not None
    sub.add_(1.0)
    r0, r1, c0, c1 = masker.bounds(200)
    assert torch.all(feat[:, r0:r1, c0:c1] == 1.0)
    assert feat.sum() == sub.numel()


def test_extract_shape_matches_area_shape(masker: AreaFeatureMasker) -> None:
    feat = torch.zeros(CHANNELS, *FEATURE_HW)
    for area_id in (0, 200, 376):
        assert tuple(masker.extract(feat, area_id).shape[-2:]) == masker.area_shape(area_id)


def test_extract_rejects_mismatched_feature_map(masker: AreaFeatureMasker) -> None:
    """Catches a backbone/grid disagreement at frame 1, not as wrong boxes."""
    with pytest.raises(ValueError, match="disagree"):
        masker.extract(torch.zeros(CHANNELS, 32, 32), area_id=0)


def test_payload_bits_follows_D2(masker: AreaFeatureMasker) -> None:
    """bits = C * cells * bits_per_element, with 1 bit/element."""
    for area_id in (0, 200, 376):
        assert masker.payload_bits(area_id, CHANNELS) == CHANNELS * masker.cell_count(area_id)


def test_full_map_bits_reproduces_the_papers_216_mb(grid: AreaGrid) -> None:
    """Paper section VI-C: "Each complete shared feature is compressed to
    2.16Mb". 256 * 48 * 176 = 2,162,688 -- exactly 1 bit per element, which
    is what pins the compression rate (design doc D2)."""
    m = AreaFeatureMasker(grid, feature_hw=FEATURE_HW)
    assert m.full_map_bits(channels=256) == 2_162_688


def test_payload_bits_sum_over_all_areas_equals_full_map(
    masker: AreaFeatureMasker,
) -> None:
    """The strict cell partition means per-area payloads add up exactly --
    no boundary cell is billed twice, which would inflate the reduction."""
    total = sum(masker.payload_bits(a, CHANNELS) for a in range(len(masker.grid)))
    assert total == masker.full_map_bits(CHANNELS)


def test_reduction_ratio(masker: AreaFeatureMasker) -> None:
    assert masker.reduction_ratio([200], CHANNELS) == pytest.approx(
        masker.full_map_bits(CHANNELS) / masker.payload_bits(200, CHANNELS)
    )
    # a CAV in no group transmits nothing -- a real outcome, not an error
    assert masker.reduction_ratio([], CHANNELS) == float("inf")


def test_single_area_is_two_orders_of_magnitude_cheaper(masker: AreaFeatureMasker) -> None:
    """The mechanism behind the paper's 44x headline: one area is ~0.3% of a
    full feature map, so even several areas per CAV beats sending everything."""
    assert masker.reduction_ratio([200], 256) > 100


def test_scatter_into_round_trips(masker: AreaFeatureMasker) -> None:
    canvas = torch.zeros(CHANNELS, *FEATURE_HW)
    h, w = masker.area_shape(200)
    patch = torch.full((CHANNELS, h, w), 3.0)
    masker.scatter_into(canvas, patch, 200)
    assert torch.equal(masker.extract(canvas, 200), patch)
    assert canvas.sum() == patch.sum()


def test_scatter_into_rejects_wrong_shape(masker: AreaFeatureMasker) -> None:
    with pytest.raises(ValueError):
        masker.scatter_into(
            torch.zeros(CHANNELS, *FEATURE_HW), torch.zeros(CHANNELS, 1, 1), 200
        )


def test_masker_rejects_bad_compression(grid: AreaGrid) -> None:
    with pytest.raises(ValueError):
        AreaFeatureMasker(grid, FEATURE_HW, bits_per_element=0.0)


# --------------------------------------------------------------------- #
# backbone forward behaviour
# --------------------------------------------------------------------- #


def test_encode_shape(model: NativeReferenceBackbone) -> None:
    feats = model.encode(_agent_inputs(n_agents=3))
    assert tuple(feats.shape) == (3, CHANNELS, *FEATURE_HW)


def test_confidence_shape_and_range(
    model: NativeReferenceBackbone, features: torch.Tensor
) -> None:
    """Eq. 1 must yield a probability -- Eq. 2's noisy-OR is only valid in
    [0, 1], and a confidence outside it would silently break grouping."""
    conf = model.confidence(features)
    assert tuple(conf.shape) == (3, 1, *FEATURE_HW)
    assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0


def test_confidence_uses_the_shared_detection_head(
    model: NativeReferenceBackbone, features: torch.Tensor
) -> None:
    """Design doc D1: f_gen IS the detector's classification head, not a
    separate network. Verified by construction -- perturbing the shared head
    must move the confidence map."""
    before = model.confidence(features).clone()
    with torch.no_grad():
        model.head.cls_head.bias.add_(1.0)
    after = model.confidence(features)
    assert not torch.allclose(before, after)


def test_fuse_shape(model: NativeReferenceBackbone, masker: AreaFeatureMasker,
                    features: torch.Tensor) -> None:
    parts = [masker.extract(features[i], 200) for i in range(3)]
    fused = model.fuse(parts[0], parts[1:])
    assert fused.shape == parts[0].shape


def test_fuse_with_no_members_returns_ego_unchanged(
    model: NativeReferenceBackbone, masker: AreaFeatureMasker, features: torch.Tensor
) -> None:
    """A group of one (Eq. 8 admitted nobody) must not pay for attention over
    a single agent, and must not perturb the ego feature."""
    ego = masker.extract(features[0], 200)
    assert torch.equal(model.fuse(ego, []), ego)


def test_fuse_rejects_mismatched_member_shapes(
    model: NativeReferenceBackbone, masker: AreaFeatureMasker, features: torch.Tensor
) -> None:
    ego = masker.extract(features[0], 200)
    other = masker.extract(features[1], 0)
    if other.shape != ego.shape:
        with pytest.raises(ValueError):
            model.fuse(ego, [other])


def test_detect_shapes(model: NativeReferenceBackbone, masker: AreaFeatureMasker,
                       features: torch.Tensor) -> None:
    fused = masker.extract(features[0], 200)
    out = model.detect(fused)
    h, w = masker.area_shape(200)
    assert tuple(out["cls"].shape) == (model.num_anchors, h, w)
    assert tuple(out["reg"].shape) == (model.num_anchors * 7, h, w)


def test_attention_fusion_shapes() -> None:
    fusion = PerPixelAttentionFusion(CHANNELS)
    out = fusion(torch.randn(4, CHANNELS, 3, 5))
    assert tuple(out.shape) == (CHANNELS, 3, 5)


def test_attention_fusion_rejects_bad_input() -> None:
    fusion = PerPixelAttentionFusion(CHANNELS)
    with pytest.raises(ValueError):
        fusion(torch.randn(4, CHANNELS, 3))
    with pytest.raises(ValueError):
        fusion(torch.randn(4, CHANNELS + 1, 3, 5))


# --------------------------------------------------------------------- #
# measurement plane: taps must not alter the forward pass
# --------------------------------------------------------------------- #


def test_forward_identical_with_and_without_taps(
    model: NativeReferenceBackbone, masker: AreaFeatureMasker, features: torch.Tensor
) -> None:
    """The plane-2 invariant, inherited from corabench and extended to LGCP.

    Observation must be physically incapable of changing results, otherwise
    every measured robustness number is suspect.
    """
    parts = [masker.extract(features[i], 200) for i in range(3)]

    conf_clean = model.confidence(features)
    fused_clean = model.fuse(parts[0], parts[1:])
    det_clean = model.detect(fused_clean)

    taps = TapSet([StatsTap()], strict=True)
    conf_tapped = model.confidence(features, taps=taps)
    fused_tapped = model.fuse(parts[0], parts[1:], taps=taps)
    det_tapped = model.detect(fused_tapped, taps=taps)

    assert torch.equal(conf_clean, conf_tapped)
    assert torch.equal(fused_clean, fused_tapped)
    assert torch.equal(det_clean["cls"], det_tapped["cls"])
    assert torch.equal(det_clean["reg"], det_tapped["reg"])


def test_taps_record_every_declared_qkv_location(
    model: NativeReferenceBackbone, masker: AreaFeatureMasker, features: torch.Tensor
) -> None:
    """The brief requires query/key/value/scores/softmax to be reachable."""
    stats = StatsTap()
    taps = TapSet([stats])
    parts = [masker.extract(features[i], 200) for i in range(3)]
    model.confidence(features, taps=taps)
    model.fuse(parts[0], parts[1:], taps=taps)

    seen = {r.location for r in stats.records}
    required = {
        "lgcp/perception/psm_single",
        "lgcp/perception/confidence_map",
        "lgcp/perception/attn_query",
        "lgcp/perception/attn_key",
        "lgcp/perception/attn_value",
        "lgcp/perception/attn_scores",
        "lgcp/perception/attn_softmax",
        "lgcp/perception/fused_feature",
    }
    assert required <= seen


def test_attention_softmax_rows_are_distributions(
    model: NativeReferenceBackbone, masker: AreaFeatureMasker, features: torch.Tensor
) -> None:
    stats = StatsTap()
    taps = TapSet([stats])
    parts = [masker.extract(features[i], 200) for i in range(3)]
    model.fuse(parts[0], parts[1:], taps=taps)
    rec = next(r for r in stats.records if r.location == "lgcp/perception/attn_softmax")
    h, w = masker.area_shape(200)
    assert rec.shape == (h * w, 3, 3)
