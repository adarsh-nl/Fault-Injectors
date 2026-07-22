"""
Tests for the camera lift and encoder, and for the two-track parity contract.

The geometry gets the most attention, because it is the part with a *correct*
answer that the loss would not reveal. A lift that transposes its axes, or is
off by half a cell, still trains -- to a model that has learned the wrong
correspondence and will never say so. So the tests place a known pixel at a
known depth through a known calibration and check which BEV cell it lands in.

The second thing pinned here is ``test_track_parity``: the same Where2comm
built with each encoder must emit the identical set of post-encoder tap
locations and produce identically-shaped outputs. That is what turns "only the
encoder is modality-specific" from a claim in a docstring into a checked
property -- and it is the reason a camera track cost one encoder rather than a
second model.
"""

from __future__ import annotations

import math

import pytest
import torch

from cpbench.data import GridSpec
from cpbench.observation import StatsTap, TapSet
from w2cbench.comm import CommunicationGraph, ThresholdSelector
from w2cbench.fusion import AttenFusion, SpatialTransform
from w2cbench.models import (CameraEncoder, DepthSplatLifting,
                             LidarPillarEncoder, SpatialConfidenceGenerator,
                             Where2comm)
from w2cbench.models.lifting import DepthDistributionHead, FrustumSplat
from w2cbench.observation import validate_location

DIM = 16


def _spec(voxel: float = 1.6, extent: float = 25.6) -> GridSpec:
    """32x32 pillars, 16x16 features at 3.2 m per cell -- coarse and fast."""
    return GridSpec(voxel_size=(voxel, voxel),
                    point_range=(-extent, -extent, -3.0, extent, extent, 1.0))


def _pinhole(focal: float = 32.0, size: int = 64) -> torch.Tensor:
    """A square pinhole camera with its principal point at the image centre."""
    k = torch.eye(3)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = k[1, 2] = size / 2.0
    return k


def _camera_to_agent_forward() -> torch.Tensor:
    """Camera +z (forward) -> agent +x; camera +x (right) -> agent -y."""
    e = torch.eye(4)
    e[:3, :3] = torch.tensor([[0.0, 0.0, 1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])
    return e


def _camera_batch(n_agents: int = 2, cameras: int = 3, size: int = 64,
                  samples: int = 1) -> dict:
    return {
        "images": torch.rand(samples, n_agents, cameras, 3, size, size),
        "intrinsics": _pinhole(size=size).expand(
            samples, n_agents, cameras, 3, 3).contiguous(),
        "extrinsics": _camera_to_agent_forward().expand(
            samples, n_agents, cameras, 4, 4).contiguous(),
        "record_len": [n_agents] * samples,
    }


def _camera_encoder(**kwargs) -> CameraEncoder:
    defaults = dict(out_channels=DIM, backbone_arch="resnet18",
                    pretrained=False, id_pick=[1],
                    depth_bins=(4.0, 20.0, 4.0), image_size=(64, 64))
    defaults.update(kwargs)
    return CameraEncoder(_spec(), **defaults).eval()


# ------------------------------------------------------------- depth head --

def test_the_depth_distribution_is_a_distribution() -> None:
    head = DepthDistributionHead(in_channels=8, out_channels=4, depth_bins=6)
    context, depth = head(torch.randn(2, 8, 5, 5))
    assert context.shape == (2, 4, 5, 5) and depth.shape == (2, 6, 5, 5)
    assert torch.allclose(depth.sum(1), torch.ones(2, 5, 5), atol=1e-6)


def test_context_and_depth_come_from_one_projection() -> None:
    """Predicted from the same features; two convolutions would add parameters
    without adding information, and the reference uses one."""
    head = DepthDistributionHead(in_channels=8, out_channels=4, depth_bins=6)
    assert head.project.out_channels == 4 + 6


# -------------------------------------------------------------- geometry --

def test_a_pixel_at_a_known_depth_lands_in_the_expected_bev_cell() -> None:
    """The test the loss cannot substitute for. A transposed or half-cell-off
    lift still trains -- to a model that has learned the wrong correspondence
    and will never report it."""
    spec = _spec()
    splat = FrustumSplat(spec, depth_bins=(8.0, 24.0, 8.0), image_size=(64, 64))
    points = splat.frustum_points(
        _pinhole()[None], _camera_to_agent_forward()[None], feature_hw=(2, 2))

    # (Z, h, w, 3) for the single camera; the principal-point ray is the mean
    # of the four feature-cell centres of a 2x2 map by symmetry.
    centre_ray = points[0].reshape(splat.n_depths, -1, 3).mean(dim=1)
    for index, depth in enumerate((8.0, 16.0)):
        # Straight ahead is +x in the agent frame, at exactly the bin depth.
        assert float(centre_ray[index, 0]) == pytest.approx(depth, abs=1e-4)
        assert float(centre_ray[index, 1]) == pytest.approx(0.0, abs=1e-4)


def test_depth_scales_the_ray_linearly() -> None:
    """Doubling the depth must double the distance, or the frustum is not a
    frustum."""
    splat = FrustumSplat(_spec(), depth_bins=(10.0, 30.0, 10.0),
                         image_size=(64, 64))
    points = splat.frustum_points(
        _pinhole()[None], _camera_to_agent_forward()[None], feature_hw=(2, 2))
    near = points[0, 0].reshape(-1, 3).norm(dim=-1).mean()
    far = points[0, 1].reshape(-1, 3).norm(dim=-1).mean()
    assert float(far / near) == pytest.approx(2.0, abs=1e-3)


def test_the_extrinsic_rotation_is_applied_not_ignored() -> None:
    """A camera mounted facing left must place its features to the left. An
    identity extrinsic would silently put every camera forward."""
    splat = FrustumSplat(_spec(), depth_bins=(10.0, 20.0, 10.0),
                         image_size=(64, 64))
    forward = splat.frustum_points(_pinhole()[None],
                                   _camera_to_agent_forward()[None], (2, 2))
    rotated = _camera_to_agent_forward().clone()
    yaw = math.pi / 2
    spin = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.0],
                         [math.sin(yaw), math.cos(yaw), 0.0],
                         [0.0, 0.0, 1.0]])
    rotated[:3, :3] = spin @ rotated[:3, :3]
    turned = splat.frustum_points(_pinhole()[None], rotated[None], (2, 2))
    assert float(forward[0, 0].reshape(-1, 3).mean(0)[0]) > 5.0    # +x
    assert float(turned[0, 0].reshape(-1, 3).mean(0)[1]) > 5.0     # +y


def test_a_perturbed_focal_length_displaces_every_frustum_point() -> None:
    """How CalibrationErrorInjector reaches the BEV map: K and E enter the
    model here and nowhere else, so a mis-calibrated camera moves features
    without touching image content at all."""
    splat = FrustumSplat(_spec(), depth_bins=(10.0, 20.0, 10.0),
                         image_size=(64, 64))
    clean = splat.frustum_points(_pinhole(focal=32.0)[None],
                                 _camera_to_agent_forward()[None], (4, 4))
    skewed = splat.frustum_points(_pinhole(focal=40.0)[None],
                                  _camera_to_agent_forward()[None], (4, 4))
    assert not torch.allclose(clean, skewed, atol=1e-3)


def test_points_outside_the_grid_are_dropped_not_wrapped() -> None:
    """An out-of-range index would wrap around and deposit a distant object on
    the opposite side of the ego -- a wrong answer that looks like a real
    detection."""
    spec = _spec(extent=6.4)          # a tiny 12.8 m grid
    splat = FrustumSplat(spec, depth_bins=(100.0, 200.0, 100.0),
                         image_size=(64, 64))
    points = splat.frustum_points(_pinhole()[None],
                                  _camera_to_agent_forward()[None], (2, 2))
    frustum = torch.ones(1, DIM, splat.n_depths, 2, 2)
    out = splat(frustum, points, n_agents=1, cameras=1)
    assert float(out.abs().sum()) == 0.0     # everything was 100 m away


def test_each_agents_cameras_accumulate_only_onto_its_own_canvas() -> None:
    """The failure this prevents is silent: one agent's cameras summed onto
    another's map would fabricate observations the ego never received."""
    splat = FrustumSplat(_spec(), depth_bins=(8.0, 16.0, 8.0),
                         image_size=(64, 64))
    points = splat.frustum_points(
        _pinhole().expand(4, 3, 3).contiguous(),
        _camera_to_agent_forward().expand(4, 4, 4).contiguous(), (2, 2))
    frustum = torch.zeros(4, DIM, splat.n_depths, 2, 2)
    frustum[:2] = 1.0                          # only agent 0's two cameras
    out = splat(frustum, points, n_agents=2, cameras=2)
    assert float(out[0].abs().sum()) > 0.0
    assert float(out[1].abs().sum()) == 0.0


def test_invalid_depth_bins_are_rejected() -> None:
    with pytest.raises(ValueError, match="depth_bins must be"):
        FrustumSplat(_spec(), depth_bins=(20.0, 4.0, 1.0), image_size=(64, 64))
    with pytest.raises(ValueError, match="depth_bins must be"):
        FrustumSplat(_spec(), depth_bins=(4.0, 20.0, 0.0), image_size=(64, 64))


# --------------------------------------------------------------- encoder --

def test_the_camera_encoder_matches_the_lidar_encoders_contract() -> None:
    encoder = _camera_encoder()
    assert encoder.out_channels == DIM
    assert encoder.feature_hw == _spec().feature_hw
    out = encoder(_camera_batch(n_agents=2))
    assert out.shape == (2, DIM, *_spec().feature_hw)


def test_ragged_agent_counts_pick_only_real_agents() -> None:
    """A padded slot is an all-zero image; splatting it would deposit a
    genuine-looking empty observation into a real BEV map."""
    batch = _camera_batch(n_agents=3, samples=2)
    batch["record_len"] = [3, 1]
    out = _camera_encoder()(batch)
    assert out.shape[0] == 4


def test_a_channel_mismatch_between_lift_and_encoder_is_named() -> None:
    lift = DepthSplatLifting(in_channels=128, out_channels=8, grid=_spec(),
                             depth_bins=(4.0, 20.0, 4.0), image_size=(64, 64))
    with pytest.raises(ValueError, match="declares out_channels=16"):
        CameraEncoder(_spec(), out_channels=16, backbone_arch="resnet18",
                      pretrained=False, id_pick=[1], lifting=lift)


def test_an_out_of_range_lift_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="lift_level 2 is out of range"):
        CameraEncoder(_spec(), out_channels=DIM, backbone_arch="resnet18",
                      pretrained=False, id_pick=[1], lift_level=2)


def test_gradients_reach_the_backbone_through_the_splat() -> None:
    """index_add_ is differentiable; if it were not, the whole camera track
    would train only its depth head."""
    encoder = _camera_encoder()
    encoder.train()
    encoder(_camera_batch(n_agents=1, cameras=2)).sum().backward()
    grads = {n: p.grad for n, p in encoder.named_parameters()
             if p.grad is not None}
    assert any("backbone" in n and float(g.abs().sum()) > 0
               for n, g in grads.items())
    assert any("head.project" in n for n in grads)


def test_the_camera_encoder_emits_its_registered_locations() -> None:
    tap = StatsTap()
    _camera_encoder()(_camera_batch(n_agents=1, cameras=2),
                      taps=TapSet([tap], strict=True))
    emitted = {r.location for r in tap.records}
    for name in ("backbone/normalised", "backbone/feat_s0",
                 "lift/image_features", "lift/depth_logits",
                 "lift/depth_distribution", "lift/frustum",
                 "lift/frustum_points", "lift/splatted",
                 "encoder/bev_features"):
        assert name in emitted, name
    for record in tap.records:
        assert record.module in validate_location(record.location).emitters(), (
            f"{record.location}: emitted by {record.module}")


def test_bev_features_is_emitted_exactly_once() -> None:
    tap = StatsTap()
    _camera_encoder()(_camera_batch(n_agents=1, cameras=2),
                      taps=TapSet([tap], strict=True))
    assert sum(r.location == "encoder/bev_features" for r in tap.records) == 1


# -------------------------------------------------------- the parity contract --

def _model(encoder) -> Where2comm:
    spec = _spec()
    return Where2comm(
        encoder=encoder,
        confidence=SpatialConfidenceGenerator(in_channels=DIM),
        selector=ThresholdSelector(threshold=0.01),
        aggregator=AttenFusion(dim=DIM),
        warp=SpatialTransform.from_grid_spec(spec),
        graph=CommunicationGraph()).eval()


def _lidar_batch(n_agents: int = 2, n_pillars: int = 8) -> dict:
    coords = torch.stack([torch.arange(n_pillars) % n_agents,
                          torch.arange(n_pillars) % 8,
                          torch.arange(n_pillars) % 8], dim=1)
    return {"features": torch.randn(n_pillars, 4, 9), "coords": coords,
            "num_points": torch.full((n_pillars,), 4),
            "record_len": [n_agents],
            "T_agent_to_ego": torch.eye(4).expand(1, n_agents, 4, 4).contiguous()}


def test_track_parity() -> None:
    """THE two-track contract. The same Where2comm, built with each encoder,
    must emit the identical set of post-encoder tap locations and produce
    identically-shaped outputs.

    This is what makes "only the encoder is modality-specific" a checked
    property rather than a docstring claim -- and it is why the camera track
    cost one encoder instead of a second model.
    """
    spec = _spec()
    lidar = _model(LidarPillarEncoder(spec, out_channels=DIM))
    camera = _model(_camera_encoder())

    def run(model, batch) -> tuple:
        tap = StatsTap()
        out = model(batch, taps=TapSet([tap], strict=True))
        post_encoder = {r.location for r in tap.records
                        if r.location.split("/")[0] in
                        ("confidence", "comm", "align", "fusion", "head")}
        return out, post_encoder

    camera_batch = _camera_batch(n_agents=2)
    camera_batch["T_agent_to_ego"] = torch.eye(4).expand(
        1, 2, 4, 4).contiguous()

    lidar_out, lidar_locations = run(lidar, _lidar_batch(n_agents=2))
    camera_out, camera_locations = run(camera, camera_batch)

    assert lidar_locations == camera_locations, (
        "post-encoder taps differ between tracks:\n  only lidar: "
        f"{sorted(lidar_locations - camera_locations)}\n  only camera: "
        f"{sorted(camera_locations - lidar_locations)}")
    for key in ("cls", "reg", "fused", "confidence", "single_cls"):
        assert lidar_out[key].shape == camera_out[key].shape, key


def test_the_camera_track_runs_the_whole_communication_stack() -> None:
    """Not just the shapes: the protocol actually executes, with bytes
    counted, so the camera track is benchmarkable rather than merely
    constructible."""
    from w2cbench.comm import CommVolumeAccountant

    batch = _camera_batch(n_agents=3)
    batch["T_agent_to_ego"] = torch.eye(4).expand(1, 3, 4, 4).contiguous()
    accountant = CommVolumeAccountant(bytes_per_element=4)
    accountant.start_frame()
    out = _model(_camera_encoder())(batch, accountant=accountant)
    accountant.end_frame(0)
    assert out["cls"].shape[0] == 1
    assert accountant.compute()["bytes_per_frame"] > 0
