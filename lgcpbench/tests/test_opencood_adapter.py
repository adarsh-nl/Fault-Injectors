"""
Tests for the OpenCOOD adapter.

Scope, stated plainly
    These run against structural stubs (``perception/opencood/stub.py``) that
    mirror OpenCOOD's submodule names, dict conventions and fusion signatures.
    They verify the ADAPTER's logic: submodule driving, the eval-mode guard,
    checkpoint verification, per-model fusion dispatch, area restriction and
    tap emission.

    They do NOT verify numerical fidelity to OpenCOOD, because OpenCOOD cannot
    run here (Python 3.7, numba==0.49.0, spconv, CUDA). A passing suite means
    the adapter drives the real model correctly; it does not mean reproduced
    Table II numbers are right. Tests needing the real package are marked
    ``@pytest.mark.opencood`` and skip when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpbench.observation.recorders import StatsTap
from cpbench.observation.taps import TapSet
from lgcpbench.perception import AreaFeatureMasker, CollabPerceptionModel
from lgcpbench.perception.opencood import available_core_methods, build_fusion_strategy
from lgcpbench.perception.opencood.adapter import OpenCOODBackbone
from lgcpbench.perception.opencood.stub import StubOpenCOODModel, stub_agent_inputs
from lgcpbench.roi import AreaGrid

GRID_HW = (64, 192)
FEATURE_HW = (16, 48)
CHANNELS = 32
CORE_METHODS = ("point_pillar_where2comm", "point_pillar_cobevt", "point_pillar_coalign")

opencood_only = pytest.mark.skipif(
    pytest.importorskip is None, reason="placeholder"
)


def _backbone(core_method: str = "point_pillar_where2comm", **kw) -> OpenCOODBackbone:
    torch.manual_seed(0)
    model = StubOpenCOODModel(
        core_method=core_method, grid_hw=GRID_HW, feature_hw=FEATURE_HW,
        channels=CHANNELS, **kw,
    )
    adapter = OpenCOODBackbone(
        model=model, core_method=core_method,
        feature_hw=FEATURE_HW, channels=CHANNELS,
    )
    return adapter.eval()


# --------------------------------------------------------------------- #
# protocol conformance
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("core_method", CORE_METHODS)
def test_adapter_satisfies_the_protocol(core_method: str) -> None:
    """Swapping OpenCOOD in must be a config change, not a rewrite."""
    assert isinstance(_backbone(core_method), CollabPerceptionModel)


@pytest.mark.parametrize("core_method", CORE_METHODS)
def test_all_three_paper_models_are_supported(core_method: str) -> None:
    assert core_method in available_core_methods()
    _backbone(core_method)


def test_unknown_core_method_is_rejected_with_guidance() -> None:
    """Each model fuses with a different signature, so a new one genuinely
    needs its own strategy rather than a default."""
    model = StubOpenCOODModel(grid_hw=GRID_HW, feature_hw=FEATURE_HW, channels=CHANNELS)
    with pytest.raises(KeyError, match="fusion strategy"):
        build_fusion_strategy("point_pillar_v2vnet", model)


def test_missing_submodule_fails_at_construction() -> None:
    """The adapter drives submodules directly, so a layout change must fail
    loudly at build time rather than as an AttributeError mid-frame."""
    model = StubOpenCOODModel(grid_hw=GRID_HW, feature_hw=FEATURE_HW, channels=CHANNELS)
    del model.cls_head
    with pytest.raises(AttributeError, match="cls_head"):
        OpenCOODBackbone(model, "point_pillar_where2comm", FEATURE_HW, CHANNELS)


# --------------------------------------------------------------------- #
# the two OpenCOOD footguns
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["encode", "confidence", "detect"])
def test_train_mode_is_refused(method: str) -> None:
    """Where2comm's Communication module takes a DIFFERENT BRANCH in training
    -- a random top-K mask that ignores the configured threshold entirely. Any
    confidence or communication measurement taken in train mode is invalid, so
    the adapter refuses rather than producing a plausible wrong number.
    """
    backbone = _backbone()
    backbone.model.train()

    args = {
        "encode": (stub_agent_inputs(),),
        "confidence": (torch.randn(2, CHANNELS, *FEATURE_HW),),
        "detect": (torch.randn(CHANNELS, 4, 6),),
    }[method]
    with pytest.raises(RuntimeError, match="eval mode"):
        getattr(backbone, method)(*args)


def test_fuse_also_refuses_train_mode() -> None:
    backbone = _backbone()
    backbone.model.train()
    ego = torch.randn(CHANNELS, 4, 6)
    with pytest.raises(RuntimeError, match="eval mode"):
        backbone.fuse(ego, [ego.clone()])


def test_partial_checkpoint_load_is_refused(tmp_path) -> None:
    """``train_utils.load_saved_model`` uses strict=False, so a drifted
    checkpoint leaves randomly-initialised layers in a model that reports
    success. That is the worst possible silent failure for a benchmark.
    """
    model = StubOpenCOODModel(grid_hw=GRID_HW, feature_hw=FEATURE_HW, channels=CHANNELS)
    partial = {k: v for k, v in model.state_dict().items() if "cls_head" not in k}
    path = tmp_path / "partial.pth"
    torch.save(partial, path)

    with pytest.raises(RuntimeError, match="missing"):
        OpenCOODBackbone._load_checkpoint(model, path)


def test_complete_checkpoint_loads(tmp_path) -> None:
    model = StubOpenCOODModel(grid_hw=GRID_HW, feature_hw=FEATURE_HW, channels=CHANNELS)
    path = tmp_path / "full.pth"
    torch.save(model.state_dict(), path)
    OpenCOODBackbone._load_checkpoint(model, path)   # must not raise


def test_checkpoint_accepts_a_wrapped_state_dict(tmp_path) -> None:
    model = StubOpenCOODModel(grid_hw=GRID_HW, feature_hw=FEATURE_HW, channels=CHANNELS)
    path = tmp_path / "wrapped.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 30}, path)
    OpenCOODBackbone._load_checkpoint(model, path)


def test_missing_checkpoint_file_is_reported(tmp_path) -> None:
    model = StubOpenCOODModel(grid_hw=GRID_HW, feature_hw=FEATURE_HW, channels=CHANNELS)
    with pytest.raises(FileNotFoundError):
        OpenCOODBackbone._load_checkpoint(model, tmp_path / "nope.pth")


# --------------------------------------------------------------------- #
# encode / confidence / fuse / detect
# --------------------------------------------------------------------- #


def test_encode_shape() -> None:
    feats = _backbone().encode(stub_agent_inputs(n_agents=3))
    assert tuple(feats.shape) == (3, CHANNELS, *FEATURE_HW)


def test_encode_requires_opencood_preprocessing() -> None:
    """corabench's pillar tensors use a different voxel layout; silently
    reinterpreting one as the other would produce plausible garbage."""
    inputs = stub_agent_inputs()
    object.__setattr__(inputs, "extra", {})
    with pytest.raises(KeyError, match="processed_lidar"):
        _backbone().encode(inputs)


def test_encode_validates_geometry_against_config() -> None:
    backbone = _backbone()
    backbone.feature_hw = (99, 99)
    with pytest.raises(RuntimeError, match="configured for"):
        backbone.encode(stub_agent_inputs())


def test_shrink_layer_is_applied_when_present() -> None:
    backbone = _backbone(shrink=True)
    assert backbone.shrink_flag
    assert tuple(backbone.encode(stub_agent_inputs()).shape[-2:]) == FEATURE_HW


def test_confidence_is_the_shared_head_per_derivation_d1() -> None:
    """D1: f_gen IS the detector's classification head -- sigmoid then max over
    anchors -- not a separate network. Verified by perturbing the shared head.
    """
    backbone = _backbone()
    features = torch.randn(3, CHANNELS, *FEATURE_HW)

    before = backbone.confidence(features).clone()
    with torch.no_grad():
        backbone.model.cls_head.bias.add_(1.0)
    after = backbone.confidence(features)

    assert tuple(before.shape) == (3, 1, *FEATURE_HW)
    assert float(before.min()) >= 0.0 and float(before.max()) <= 1.0
    assert not torch.allclose(before, after)


@pytest.mark.parametrize("core_method", CORE_METHODS)
def test_fuse_preserves_area_shape(core_method: str) -> None:
    """Every fusion strategy must accept an area-restricted stack and return
    the ego's feature at the same spatial size."""
    backbone = _backbone(core_method)
    ego = torch.randn(CHANNELS, 4, 7)
    collab = [torch.randn(CHANNELS, 4, 7) for _ in range(2)]
    assert tuple(backbone.fuse(ego, collab).shape) == (CHANNELS, 4, 7)


@pytest.mark.parametrize("core_method", CORE_METHODS)
def test_group_of_one_returns_ego_unchanged(core_method: str) -> None:
    backbone = _backbone(core_method)
    ego = torch.randn(CHANNELS, 4, 7)
    assert torch.equal(backbone.fuse(ego, []), ego)


def test_fuse_rejects_mismatched_shapes() -> None:
    backbone = _backbone()
    with pytest.raises(ValueError):
        backbone.fuse(torch.randn(CHANNELS, 4, 7), [torch.randn(CHANNELS, 4, 6)])


def test_detect_shapes() -> None:
    out = _backbone().detect(torch.randn(CHANNELS, 4, 7))
    assert tuple(out["cls"].shape) == (2, 4, 7)
    assert tuple(out["reg"].shape) == (14, 4, 7)


def test_detect_rejects_batched_input() -> None:
    with pytest.raises(ValueError):
        _backbone().detect(torch.randn(1, CHANNELS, 4, 7))


# --------------------------------------------------------------------- #
# multi-scale fusion (assumption B12)
# --------------------------------------------------------------------- #


def test_multiscale_uses_the_encoder_width_level() -> None:
    """B12: LGCP restricts to areas on the FINAL feature map, so the fusion
    module whose width matches the encoder output is the one that applies.
    Reproducing true multi-scale fusion would need one backbone pass per area,
    which destroys the encode-once discipline.
    """
    backbone = _backbone("point_pillar_where2comm")
    modules = backbone.model.fusion_net.fuse_modules
    assert isinstance(modules, torch.nn.ModuleList) and len(modules) == 3
    assert backbone.fusion.module is modules[-1]
    assert backbone.fusion.module.channels == CHANNELS


def test_single_scale_configuration_is_supported() -> None:
    from lgcpbench.perception.opencood.stub import StubWhere2commFusionNet

    model = StubOpenCOODModel(grid_hw=GRID_HW, feature_hw=FEATURE_HW, channels=CHANNELS)
    model.fusion_net = StubWhere2commFusionNet(CHANNELS, multi_scale=False)
    backbone = OpenCOODBackbone(
        model, "point_pillar_where2comm", FEATURE_HW, CHANNELS
    ).eval()
    ego = torch.randn(CHANNELS, 4, 7)
    assert tuple(backbone.fuse(ego, [ego.clone()]).shape) == (CHANNELS, 4, 7)


# --------------------------------------------------------------------- #
# integration with the rest of LGCP
# --------------------------------------------------------------------- #


def test_area_restriction_works_on_opencood_features() -> None:
    """The whole point: OpenCOOD features slice by area exactly like native
    ones, so the control plane is unchanged by the backend swap."""
    grid = AreaGrid((-38.4, -12.8, -3.0, 38.4, 12.8, 1.0))
    masker = AreaFeatureMasker(grid, FEATURE_HW)
    backbone = _backbone()

    features = backbone.encode(stub_agent_inputs(n_agents=3))
    area_id = 10
    parts = [masker.extract(features[i], area_id) for i in range(3)]
    fused = backbone.fuse(parts[0], parts[1:])

    assert fused.shape == parts[0].shape
    assert tuple(backbone.detect(fused)["cls"].shape[-2:]) == masker.area_shape(area_id)


def test_adapter_emits_the_full_tap_surface() -> None:
    stats = StatsTap()
    taps = TapSet([stats])
    backbone = _backbone()

    features = backbone.encode(stub_agent_inputs(), taps=taps)
    backbone.confidence(features, taps=taps)
    ego = torch.randn(CHANNELS, 4, 7)
    fused = backbone.fuse(ego, [torch.randn(CHANNELS, 4, 7)], taps=taps)
    backbone.detect(fused, taps=taps)

    seen = {r.location for r in stats.records}
    assert {
        "lgcp/perception/bev_features",
        "lgcp/perception/psm_single",
        "lgcp/perception/confidence_map",
        "lgcp/perception/fused_feature",
        "lgcp/perception/cls_logits",
        "lgcp/perception/reg_map",
    } <= seen


def test_output_identical_with_and_without_taps() -> None:
    """The measurement-plane invariant holds for the OpenCOOD backend too."""
    backbone = _backbone()
    features = torch.randn(3, CHANNELS, *FEATURE_HW)
    ego = torch.randn(CHANNELS, 4, 7)
    collab = [torch.randn(CHANNELS, 4, 7)]

    clean_conf = backbone.confidence(features)
    clean_fused = backbone.fuse(ego, collab)
    tapped_conf = backbone.confidence(features, taps=TapSet([StatsTap()], strict=True))
    tapped_fused = backbone.fuse(ego, collab, taps=TapSet([StatsTap()], strict=True))

    assert torch.equal(clean_conf, tapped_conf)
    assert torch.equal(clean_fused, tapped_fused)


# --------------------------------------------------------------------- #
# requires a real OpenCOOD install
# --------------------------------------------------------------------- #


@pytest.mark.opencood
def test_real_opencood_import() -> None:
    """Skipped unless the Python 3.7 OpenCOOD environment is present.

    This is the boundary of what can be verified here. Everything above tests
    the adapter against stubs; only this path exercises the real package.
    """
    pytest.importorskip("opencood", reason="OpenCOOD needs its own py3.7 env")
    from opencood.tools import train_utils

    assert hasattr(train_utils, "create_model")
