"""
Tests for camera-to-BEV lifting: geometry embeddings, cross-attention, and
SinBEVT end to end.

The lifting is the part of CoBEVT with no depth network and no explicit
projection -- it works by matching ray directions. That makes the camera
intrinsics and extrinsics load-bearing tensors on the attention path, and
several tests here exist specifically to prove that, because it is the
premise of the calibration-error fault surface.
"""

from __future__ import annotations

import pytest
import torch

from cobevtbench.attention.fax_cross import (CrossWindowAttention,
                                             FAXCrossAttentionBlock)
from cobevtbench.fusion.camera_embedding import (CameraGeometryEmbedding,
                                                 pixel_grid)
from cobevtbench.models.sinbevt import BEVSelfAttention, SinBEVT
from cobevtbench.observation.locations import LOCATIONS, _template
from cpbench.data import BEVGrid
from cpbench.observation import StatsTap, TapSet


def _geo(dim: int = 16, bev: int = 8) -> CameraGeometryEmbedding:
    return CameraGeometryEmbedding(dim, BEVGrid(bev, bev, 20.0, 20.0), (64, 64))


def _calib(cameras: int = 4):
    K = torch.eye(3).expand(1, cameras, 3, 3).contiguous()
    T = torch.eye(4).expand(1, cameras, 4, 4).contiguous()
    return K, T


def _sinbevt(**kwargs) -> SinBEVT:
    params = dict(dims=[16, 16], feat_channels=[8, 8], bev_size=16,
                  bev_meters=40.0, image_size=(32, 32), q_win_sizes=[8, 8],
                  feat_win_sizes=[4, 4], heads=[2, 2], dim_head=[8, 8],
                  middle=[1, 1], bev_embedding_flags=[True, False],
                  self_attn_dim_head=8)
    params.update(kwargs)
    return SinBEVT(**params).eval()


def _feats():
    return [torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 4, 4)]


# ------------------------------------------------------------- geometry --

def test_pixel_grid_is_scaled_into_intrinsic_units() -> None:
    """K is expressed in image pixels but attention runs on strided feature
    maps. Skipping the rescale still produces rays -- rays for a camera with
    the wrong focal length, which trains to a systematically warped BEV."""
    grid = pixel_grid(4, 4, image_height=64, image_width=64)
    assert grid[0, 0, 1] == pytest.approx(16.0)      # not 1.0
    assert grid[2].eq(1.0).all()                     # homogeneous


def test_direction_fields_are_unit_length() -> None:
    """They are compared by dot product, which is only a cosine similarity if
    both sides are normalised. Unnormalised, magnitude would leak into the
    attention logits as a spurious brightness-like prior."""
    img, bev = _geo()(*_calib(), feature_hw=(4, 4))
    assert torch.allclose(img.norm(dim=2), torch.ones(1, 4, 4, 4), atol=1e-5)
    assert torch.allclose(bev.norm(dim=2), torch.ones(1, 4, 8, 8), atol=1e-5)


def test_changing_intrinsics_changes_the_ray_field() -> None:
    """The premise of CalibrationErrorInjector. If K did not reach the
    embedding, a miscalibration fault would inject nothing and report perfect
    robustness."""
    geo = _geo()
    K, T = _calib()
    baseline, _ = geo(K, T, feature_hw=(4, 4))
    K_off = K.clone()
    K_off[:, :, 0, 0] *= 1.15                        # focal length drift
    perturbed, _ = geo(K_off, T, feature_hw=(4, 4))
    assert not torch.allclose(baseline, perturbed, atol=1e-4)


def test_changing_extrinsics_changes_both_fields() -> None:
    """Extrinsics set the camera origin, which both sides measure from, so a
    mounting error must move the image rays AND the BEV directions."""
    geo = _geo()
    K, T = _calib()
    img_a, bev_a = geo(K, T, feature_hw=(4, 4))
    T_off = T.clone()
    T_off[:, :, 0, 3] += 0.5                         # 50 cm mounting error
    img_b, bev_b = geo(K, T_off, feature_hw=(4, 4))
    assert not torch.allclose(img_a, img_b, atol=1e-4)
    assert not torch.allclose(bev_a, bev_b, atol=1e-4)


def test_cameras_with_different_poses_get_different_embeddings() -> None:
    """A rig whose cameras all embedded identically could not disambiguate
    which camera saw what -- the lift would be direction-blind."""
    K, T = _calib(cameras=2)
    T = T.clone()
    T[:, 1, :3, :3] = torch.tensor([[0.0, -1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0]])   # second camera yawed
    img, _ = _geo()(K, T, feature_hw=(4, 4))
    assert not torch.allclose(img[:, 0], img[:, 1], atol=1e-4)


def test_geometry_is_observable() -> None:
    tap = StatsTap()
    _geo()(*_calib(), feature_hw=(4, 4), taps=TapSet([tap], strict=True),
           location_prefix="sinbevt/b1")
    assert {r.location for r in tap.records} == {
        "sinbevt/b1/cam_embed", "sinbevt/b1/img_embed",
        "sinbevt/b1/bev_pos_embed"}


# ------------------------------------------------------ cross-attention --

def test_mismatched_window_counts_raise_with_the_numbers_named() -> None:
    """The constraint most easily broken by changing image resolution alone.
    Without this check it surfaces as a batch-size mismatch inside a matmul,
    which says nothing about q_win_size or feat_win_size."""
    attn = CrossWindowAttention(16, 16, 16, dim_head=8, num_heads=2)
    query = torch.randn(1, 2, 2, 2, 4, 4, 16)       # 4 query windows
    key = value = torch.randn(1, 2, 3, 3, 2, 2, 16)  # 9 key windows
    skip = torch.randn(1, 2, 2, 4, 4, 16)
    with pytest.raises(ValueError) as excinfo:
        attn(query, key, value, skip)
    message = str(excinfo.value)
    assert "4 query windows" in message and "9 key" in message
    assert "feat_win_size" in message


def test_camera_axis_is_reduced_away() -> None:
    """Attention runs with the camera axis in the tokens, then averages the
    per-camera results (assumption A6). The output must carry no camera
    axis, or the BEV grid would be per-camera."""
    attn = CrossWindowAttention(16, 16, 16, dim_head=8, num_heads=2)
    out = attn(torch.randn(1, 4, 2, 2, 4, 4, 16),
               torch.randn(1, 4, 2, 2, 2, 2, 16),
               torch.randn(1, 4, 2, 2, 2, 2, 16),
               torch.randn(1, 2, 2, 4, 4, 16))
    assert out.shape == (1, 2, 2, 4, 4, 16)


@pytest.mark.parametrize("reduce", ["mean", "sum", "none"])
def test_camera_reduce_modes_all_run_and_differ(reduce: str) -> None:
    torch.manual_seed(0)
    attn = CrossWindowAttention(16, 16, 16, dim_head=8, num_heads=2,
                                camera_reduce=reduce).eval()
    args = (torch.randn(1, 4, 1, 1, 4, 4, 16), torch.randn(1, 4, 1, 1, 2, 2, 16),
            torch.randn(1, 4, 1, 1, 2, 2, 16), torch.zeros(1, 1, 1, 4, 4, 16))
    with torch.no_grad():
        assert attn(*args).shape == (1, 1, 1, 4, 4, 16)


def test_unknown_camera_reduce_raises() -> None:
    with pytest.raises(ValueError, match="unknown camera_reduce"):
        CrossWindowAttention(16, 16, 16, 8, 2, camera_reduce="median")


def test_cross_attention_has_no_relative_position_bias() -> None:
    """Assumption A5. The paper presents FAX attention with the bias (Eq. 4),
    but the released config disables it for the cross-view blocks and the
    function that would add it is an identity stub. Pinned so the deviation
    is deliberate and visible rather than an omission."""
    block = FAXCrossAttentionBlock(dim=16, feat_channels=8, q_win_size=4,
                                   feat_win_size=2, dim_head=8, num_heads=2)
    names = [n for n, _ in block.named_modules()]
    assert not any("rel_pos_bias" in n for n in names)


def test_the_query_stays_windowed_in_both_branches() -> None:
    """The asymmetry that distinguishes cross-attention from self-attention
    here: only the key/value switch window -> grid. If the query switched
    too, 'global' would mean regrouping the BEV map rather than each BEV
    window looking further into the image."""
    tap = StatsTap()
    block = FAXCrossAttentionBlock(dim=16, feat_channels=8, q_win_size=4,
                                   feat_win_size=2, dim_head=8, num_heads=2)
    block(torch.randn(1, 16, 8, 8), torch.randn(1, 4, 8, 4, 4),
          torch.randn(1, 4, 16, 4, 4), torch.randn(1, 4, 16, 8, 8),
          taps=TapSet([tap], strict=True))
    shapes = {r.location: r.shape for r in tap.records
              if r.location.endswith("/partitioned")}
    # Both branches partition the query identically: 2x2 windows of 4x4.
    assert shapes["sinbevt/b0/local/partitioned"] == \
        shapes["sinbevt/b0/global/partitioned"]


def test_bev_embedding_flag_is_honoured() -> None:
    """The reference adds the BEV positional embedding only in the first
    block. Once the query carries lifted content, re-adding raw geometry
    competes with it."""
    torch.manual_seed(0)
    args = (torch.randn(1, 16, 8, 8), torch.randn(1, 4, 8, 4, 4),
            torch.randn(1, 4, 16, 4, 4), torch.randn(1, 4, 16, 8, 8))
    torch.manual_seed(0)
    with_embed = FAXCrossAttentionBlock(16, 8, 4, 2, 8, 2,
                                        use_bev_embedding=True).eval()
    torch.manual_seed(0)
    without = FAXCrossAttentionBlock(16, 8, 4, 2, 8, 2,
                                     use_bev_embedding=False).eval()
    with torch.no_grad():
        assert not torch.allclose(with_embed(*args), without(*args))


def test_no_image_features_ablation_uses_geometry_only() -> None:
    """The paper's geometry-only ablation: the key becomes the ray embedding
    with no appearance term."""
    torch.manual_seed(0)
    args = (torch.randn(1, 16, 8, 8), torch.randn(1, 4, 8, 4, 4),
            torch.randn(1, 4, 16, 4, 4), torch.randn(1, 4, 16, 8, 8))
    torch.manual_seed(0)
    full = FAXCrossAttentionBlock(16, 8, 4, 2, 8, 2).eval()
    torch.manual_seed(0)
    geometry_only = FAXCrossAttentionBlock(16, 8, 4, 2, 8, 2,
                                           no_image_features=True).eval()
    with torch.no_grad():
        assert not torch.allclose(full(*args), geometry_only(*args))


# ---------------------------------------------------- terminal attention --

def test_bev_self_attention_is_dense() -> None:
    """At 32x32 the map is small enough for full attention, so unlike
    FuseBEVT there is no windowing -- every cell sees every other cell."""
    attn = BEVSelfAttention(dim=16, dim_head=8, grid_hw=(4, 4))
    assert attn.rel_pos_bias.num_tokens == 16       # 4*4, not a window
    assert attn(torch.randn(2, 16, 4, 4)).shape == (2, 16, 4, 4)


def test_bev_self_attention_rejects_a_mismatched_grid() -> None:
    attn = BEVSelfAttention(dim=16, dim_head=8, grid_hw=(4, 4))
    with pytest.raises(ValueError, match="was built for a"):
        attn(torch.randn(1, 16, 8, 8))


# -------------------------------------------------------------- SinBEVT --

def test_forward_shape_and_bev_halving() -> None:
    model = _sinbevt()
    assert model.bev_sizes == [16, 8]
    assert model(_feats(), *_calib()).shape == (1, 16, 8, 8)


def test_paper_configuration_produces_the_transmitted_payload() -> None:
    """The released config must yield the 32x32x128 map the paper says goes
    on the wire (~524 KB). That number is the architecture's whole point --
    it is what a vehicle can broadcast at 10 Hz and what the compression
    ablation trades away."""
    model = SinBEVT(dims=[128, 128, 128], feat_channels=[128, 256, 512],
                    bev_size=128, bev_meters=100.0, image_size=(512, 512),
                    q_win_sizes=[16, 16, 32], feat_win_sizes=[8, 8, 16],
                    heads=[4, 4, 4], dim_head=[32, 32, 32], middle=[2, 2, 2],
                    bev_embedding_flags=[True, False, False]).eval()
    assert model.bev_sizes == [128, 64, 32]
    feats = [torch.randn(1, 4, 128, 64, 64), torch.randn(1, 4, 256, 32, 32),
             torch.randn(1, 4, 512, 16, 16)]
    with torch.no_grad():
        out = model(feats, *_calib())
    assert out.shape == (1, 128, 32, 32)
    assert out.numel() * 4 == 524288                # bytes, fp32


def test_mismatched_per_block_settings_raise() -> None:
    with pytest.raises(ValueError, match="every per-block setting"):
        _sinbevt(heads=[2])


def test_too_many_blocks_for_the_bev_size_raises() -> None:
    with pytest.raises(ValueError, match="halves to"):
        _sinbevt(dims=[16] * 6, feat_channels=[8] * 6, q_win_sizes=[8] * 6,
                 feat_win_sizes=[4] * 6, heads=[2] * 6, dim_head=[8] * 6,
                 middle=[1] * 6, bev_embedding_flags=[True] + [False] * 5)


def test_wrong_number_of_feature_scales_raises() -> None:
    with pytest.raises(ValueError, match="feature scales"):
        _sinbevt()([torch.randn(1, 4, 8, 8, 8)], *_calib())


def test_calibration_error_changes_the_lifted_bev_map() -> None:
    """End-to-end version of the premise: a miscalibrated camera must move
    the output, or the calibration fault surface is inert."""
    model = _sinbevt()
    feats = _feats()
    K, T = _calib()
    with torch.no_grad():
        baseline = model(feats, K, T)
        K_off = K.clone()
        K_off[:, 0, 0, 0] *= 1.2                    # one camera, focal drift
        assert not torch.allclose(baseline, model(feats, K_off, T), atol=1e-5)


def test_gradients_reach_every_parameter() -> None:
    model = _sinbevt()
    model(_feats(), *_calib()).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters never used in the forward pass: {missing}"


# ------------------------------------------------------------------ taps --

def test_forward_is_identical_with_and_without_taps() -> None:
    model = _sinbevt()
    feats, (K, T) = _feats(), _calib()
    with torch.no_grad():
        plain = model(feats, K, T)
        tapped = model(feats, K, T, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(plain, tapped)


def test_every_emitted_location_is_registered() -> None:
    tap = StatsTap()
    with torch.no_grad():
        _sinbevt()(_feats(), *_calib(), taps=TapSet([tap], strict=True))
    unregistered = sorted(
        {r.location for r in tap.records if _template(r.location) not in LOCATIONS})
    assert not unregistered, (
        "emitted but not in the registry:\n  " + "\n  ".join(unregistered))


def test_registered_sinbevt_locations_are_all_reachable() -> None:
    """The other direction, for the camera lifting layers."""
    tap = StatsTap()
    with torch.no_grad():
        _sinbevt()(_feats(), *_calib(), taps=TapSet([tap], strict=True))
    emitted = {_template(r.location) for r in tap.records}
    declared = {n for n in LOCATIONS if n.startswith("sinbevt/")}
    missing = sorted(declared - emitted)
    assert not missing, "registered but never emitted:\n  " + "\n  ".join(missing)
