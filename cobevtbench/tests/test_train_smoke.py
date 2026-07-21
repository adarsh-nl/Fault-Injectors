"""
End-to-end training smoke tests.

The gate for step 9: both tracks train from synthetic cooperative data, with
no downloads and no GPU, and the loss actually descends.

"Runs without raising" is a weak claim -- a model with a detached graph, a
transposed target or a loss computed on the wrong axis all run happily
forever. So the load-bearing assertions here are that the loss *decreases*
on a batch the model is allowed to overfit, and that the parameters actually
move.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cobevtbench.data.camera import CoBEVTCameraDataset
from cobevtbench.data.collate import camera_collator, lidar_collator
from cobevtbench.data.lidar import CoBEVTLidarDataset
from cobevtbench.models.cobevt_camera import CoBEVTCamera
from cobevtbench.models.cobevt_lidar import CoBEVTLidar
from cobevtbench.training.losses import DetectionLoss, VanillaSegLoss
from cobevtbench.training.trainer import Trainer
from cobevtbench.training.validator import (DetectionValidator,
                                            SegmentationValidator)
from cpbench.data import (AnchorGenerator, BEVGrid, BoxDecoder, GridSpec,
                          SyntheticCameraCooperativeDataset,
                          SyntheticCooperativeDataset, TargetAssigner)
from cpbench.logbook import ExperimentLogger, ExperimentMeta

# The target grid must equal the decoder's output grid exactly:
#   SinBEVT out = bev_size / 2^(n_blocks - 1) = 16 / 2 = 8
#   decoder     = 8 * 2^n_stages              = 8 * 4 = 32
# so the rasterised targets are 32x32 over the same 40 m the model sees.
BEV = BEVGrid(height=32, width=32, h_meters=40.0, w_meters=40.0)
SPEC = GridSpec(voxel_size=(0.8, 0.8),
                point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))

CAMERA_MODEL = dict(
    max_cav=2, image_size=(32, 32), bev_meters=40.0, bev_size=16,
    dims=[16, 16], q_win_sizes=[8, 8], feat_win_sizes=[2, 2], heads=[2, 2],
    dim_head=[8, 8], middle=[1, 1], bev_embedding_flags=[True, False],
    backbone_arch="resnet18", pretrained=False, id_pick=[1, 2],
    fuse_window=4, fuse_dim_head=8, fuse_depth=1, self_attn_dim_head=8,
    decoder_channels=[4, 8])


# ------------------------------------------------------------------ camera --

def _camera_setup(n_frames: int = 4):
    adapter = SyntheticCameraCooperativeDataset(
        n_frames=n_frames, n_agents=2, n_objects=3, image_size=(32, 32))
    dataset = CoBEVTCameraDataset(adapter, BEV, max_cav=2, target="dynamic")
    loader = DataLoader(dataset, batch_size=2, collate_fn=camera_collator(2))
    model = CoBEVTCamera(target="dynamic", **CAMERA_MODEL)
    return dataset, loader, model


def test_camera_dataset_produces_matching_shapes() -> None:
    dataset, loader, model = _camera_setup()
    batch = next(iter(loader))
    out = model(batch)
    assert out["logits"].shape[-2:] == batch["target"].shape[-2:], (
        "decoder output and rasterised target must be the same grid, or the "
        "loss silently trains against a resized label map")
    assert out["logits"].shape[0] == batch["target"].shape[0]


def test_camera_targets_are_valid_class_indices() -> None:
    """An out-of-range label makes cross_entropy raise on GPU and read
    arbitrary memory on some backends."""
    dataset, _, _ = _camera_setup()
    target = dataset[0]["target"]
    assert target.dtype == torch.int64
    assert int(target.min()) >= 0 and int(target.max()) < 2


def test_camera_loss_decreases_when_overfitting_one_batch() -> None:
    """The real check. A detached graph, a transposed target or a loss on the
    wrong axis all 'run' -- none of them descend."""
    _, loader, model = _camera_setup(n_frames=2)
    batch = next(iter(loader))
    loss_fn = VanillaSegLoss(target="dynamic")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    model.train()
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = loss_fn(model(batch)["logits"], batch["target"])["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_camera_trainer_runs_and_moves_parameters(tmp_path) -> None:
    _, loader, model = _camera_setup()
    before = model.head.head.weight.detach().clone()
    trainer = Trainer(
        model=model,
        loss_fn=lambda batch, out: VanillaSegLoss("dynamic")(
            out["logits"], batch["target"]),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        amp=False, log_every=1)
    summary = trainer.train_epoch(loader)
    assert "loss" in summary and np.isfinite(summary["loss"])
    assert not torch.equal(before, model.head.head.weight)


def test_camera_validator_reports_iou(tmp_path) -> None:
    dataset, loader, model = _camera_setup()
    validator = SegmentationValidator(loader, dataset.class_names)
    metrics = validator.run(model, epoch=0)
    assert 0.0 <= validator.score(metrics) <= 1.0
    assert metrics["n_frames"] == len(dataset)


# ------------------------------------------------------------------- lidar --

def _lidar_setup(n_frames: int = 4):
    adapter = SyntheticCooperativeDataset(n_frames=n_frames, n_agents=2,
                                          n_objects=3)
    dataset = CoBEVTLidarDataset(adapter, SPEC, max_cav=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=lidar_collator(2))
    anchors = AnchorGenerator(SPEC)
    model = CoBEVTLidar(SPEC, max_cav=2, encoder_out_channels=32,
                        fuse_depth=1, fuse_window=8, fuse_dim_head=8,
                        num_anchors=anchors.num_anchors_per_cell)
    return dataset, loader, model, anchors


def _detection_loss(assigner: TargetAssigner):
    """Assign targets on the fly, so the smoke test exercises the real
    anchor/encoding path rather than a hand-made target tensor."""
    criterion = DetectionLoss()

    def _loss(batch, out):
        cls_targets, reg_targets = [], []
        for boxes in batch["gt_boxes"]:
            assigned = assigner(boxes if boxes is not None
                                else np.zeros((0, 7), dtype=np.float32))
            cls_targets.append(assigned["cls_target"])
            reg_targets.append(assigned["reg_target"])
        return criterion(out["cls"], out["reg"],
                         torch.stack(cls_targets), torch.stack(reg_targets))
    return _loss


def test_lidar_loss_decreases_when_overfitting_one_batch() -> None:
    _, loader, model, anchors = _lidar_setup(n_frames=2)
    batch = next(iter(loader))
    loss_fn = _detection_loss(TargetAssigner(anchors))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    model.train()
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = loss_fn(batch, model(batch))["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_lidar_trainer_and_validator(tmp_path) -> None:
    dataset, loader, model, anchors = _lidar_setup()
    trainer = Trainer(model=model, loss_fn=_detection_loss(TargetAssigner(anchors)),
                      optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                      amp=False, log_every=1)
    summary = trainer.train_epoch(loader)
    assert np.isfinite(summary["loss"])

    validator = DetectionValidator(loader, BoxDecoder(anchors,
                                                      score_threshold=0.0))
    metrics = validator.run(model, epoch=0)
    assert np.isfinite(validator.score(metrics))


# ---------------------------------------------------------------- plumbing --

def test_collate_offsets_agent_indices_across_scenes() -> None:
    """Without the offset, PointPillarScatter writes every scene's agent 0
    onto the same canvas and the batch silently fuses across scenes."""
    dataset, _, _, _ = _lidar_setup()
    batch = lidar_collator(2)([dataset[0], dataset[1]])
    agent_ids = batch["coords"][:, 0].unique().tolist()
    assert agent_ids == [0, 1, 2, 3]


def test_padded_transform_slots_are_invertible() -> None:
    """The warp inverts the rotation block over the whole padded batch before
    the mask applies. Zero padding is singular, so it would produce NaNs in
    tensors that are then discarded -- the most confusing possible failure."""
    dataset, _, _, _ = _lidar_setup()
    batch = lidar_collator(4)([dataset[0]])
    padded = batch["T_agent_to_ego"][0, 2:]
    assert torch.isfinite(torch.linalg.inv(padded[:, :2, :2])).all()


def test_trainer_skips_a_non_finite_loss(tmp_path) -> None:
    """One bad batch must not poison every parameter through an inf gradient."""
    _, loader, model = _camera_setup(n_frames=2)
    before = [p.detach().clone() for p in model.parameters()]
    trainer = Trainer(
        model=model,
        loss_fn=lambda batch, out: {"loss": out["logits"].sum() * float("nan")},
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        amp=False)
    trainer.train_epoch(loader)
    assert all(torch.equal(a, b)
               for a, b in zip(before, model.parameters()))


def test_checkpoint_round_trips(tmp_path) -> None:
    _, loader, model = _camera_setup(n_frames=2)
    meta = ExperimentMeta(experiment_id="t", experiment_name="t",
                          paper="CoBEVT", architecture="camera",
                          dataset="synthetic", seed=0, deterministic=True)
    with ExperimentLogger(tmp_path, "smoke", meta,
                          logger_names=("cobevtbench",)) as book:
        trainer = Trainer(
            model=model,
            loss_fn=lambda batch, out: VanillaSegLoss("dynamic")(
                out["logits"], batch["target"]),
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            logger_=book, amp=False)
        trainer.train_epoch(loader)
        path = trainer.save("last.pt")
        assert path.exists()

        restored = CoBEVTCamera(target="dynamic", **CAMERA_MODEL)
        resumed = Trainer(model=restored,
                          loss_fn=lambda b, o: {"loss": o["logits"].sum()},
                          optimizer=torch.optim.AdamW(restored.parameters()),
                          amp=False)
        resumed.load(path)
        assert torch.equal(model.head.head.weight, restored.head.head.weight)


def test_training_log_is_not_empty(tmp_path) -> None:
    """The bug fixed in step 2: a hardcoded logger name meant lgcpbench wrote
    an empty training.log for months. Pinned so it cannot regress here."""
    import logging
    meta = ExperimentMeta(experiment_id="t", experiment_name="t",
                          paper="CoBEVT", architecture="camera",
                          dataset="synthetic", seed=0, deterministic=True)
    with ExperimentLogger(tmp_path, "logcheck", meta,
                          logger_names=("cobevtbench",)) as book:
        logging.getLogger("cobevtbench.training.trainer").info("hello")
    assert "hello" in (tmp_path / "logcheck" / "training.log").read_text()
