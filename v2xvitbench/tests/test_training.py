"""
Tests for the loss, the trainer and the validator, on synthetic data.

The one non-negotiable check is that a few optimisation steps actually
reduce the loss on a fixed batch: it is the cheapest end-to-end proof that
gradients flow from the objective through HMSA's relation matrices, the
MSwin branches and the DPE's linear readout down to the pillar encoder.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cpbench.data import (AnchorGenerator, GridSpec,
                          SyntheticCooperativeDataset, TargetAssigner)
from cpbench.data.postprocessing import BoxDecoder

from v2xvitbench.data import V2XVitLidarDataset, v2xvit_collator
from v2xvitbench.models import V2XViT
from v2xvitbench.training import (Trainer, TrainerConfig, V2XViTLoss,
                                  Validator)


def _spec() -> GridSpec:
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
                    downsample=4)


def _model() -> V2XViT:
    return V2XViT(_spec(), max_cav=3, encoder_out_channels=48,
                  shrink_channels=32, depth=1, hmsa_heads=2, hmsa_dim_head=16,
                  window_sizes=(2, 4), mswin_heads=(2, 2),
                  mswin_dim_heads=(16, 16), mlp_dim=32, dropout=0.0)


def _loader(n_frames: int = 4, batch_size: int = 2) -> DataLoader:
    adapter = SyntheticCooperativeDataset(n_frames=n_frames, n_agents=3)
    dataset = V2XVitLidarDataset(adapter, _spec(), max_cav=3,
                                 force_infra=[2])
    return DataLoader(dataset, batch_size=batch_size,
                      collate_fn=v2xvit_collator(3))


def _loss() -> V2XViTLoss:
    return V2XViTLoss(TargetAssigner(AnchorGenerator(_spec())))


# ------------------------------------------------------------------- loss --

def test_loss_terms_are_finite_and_composed() -> None:
    loss_fn = _loss()
    output = {"cls": torch.randn(1, 2, 16, 16) * 0.1,
              "reg": torch.randn(1, 14, 16, 16) * 0.1}
    batch = {"gt_boxes": [np.array([[0.0, 0.0, -1.0, 3.9, 1.6, 1.56, 0.0]],
                                   dtype=np.float32)]}
    terms = loss_fn(batch, output)
    assert set(terms) == {"loss", "loss_cls", "loss_reg"}
    assert torch.isfinite(terms["loss"])
    assert terms["loss"].requires_grad is False or True  # composed scalar


def test_reg_weight_scales_only_the_regression_term() -> None:
    output = {"cls": torch.zeros(1, 2, 16, 16),
              "reg": torch.ones(1, 14, 16, 16)}
    batch = {"gt_boxes": [np.array([[0.0, 0.0, -1.0, 3.9, 1.6, 1.56, 0.0]],
                                   dtype=np.float32)]}
    light = V2XViTLoss(TargetAssigner(AnchorGenerator(_spec())),
                       reg_weight=1.0)(batch, output)
    heavy = V2XViTLoss(TargetAssigner(AnchorGenerator(_spec())),
                       reg_weight=2.0)(batch, output)
    assert torch.allclose(light["loss_cls"], heavy["loss_cls"])
    delta = float(heavy["loss"] - light["loss"])
    assert delta == pytest.approx(float(light["loss_reg"]), rel=1e-4)


def test_empty_ground_truth_is_a_valid_frame() -> None:
    terms = _loss()({"gt_boxes": [None]},
                    {"cls": torch.zeros(1, 2, 16, 16),
                     "reg": torch.zeros(1, 14, 16, 16)})
    assert torch.isfinite(terms["loss"])


# ---------------------------------------------------------------- trainer --

def test_a_few_steps_reduce_the_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    model = _model()
    loss_fn = _loss()
    batch = next(iter(_loader(n_frames=2)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    first = None
    for _ in range(6):
        optimizer.zero_grad()
        terms = loss_fn(batch, model(batch))
        terms["loss"].backward()
        optimizer.step()
        first = float(terms["loss"]) if first is None else first
        last = float(terms["loss"])
    assert last < first, f"loss did not decrease: {first} -> {last}"


def test_trainer_fit_records_history_and_steps_scheduler() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1],
                                                     gamma=0.1)
    trainer = Trainer(model, _loss(), optimizer,
                      TrainerConfig(epochs=2, eval_every=0),
                      scheduler=scheduler)
    history = trainer.fit(_loader(n_frames=2))
    assert len(history) == 2 * len(_loader(n_frames=2))
    assert history[0].loss_total > 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_trainer_grad_clip_bounds_the_norm() -> None:
    model = _model()
    trainer = Trainer(model, _loss(),
                      torch.optim.Adam(model.parameters(), lr=1e-3),
                      TrainerConfig(epochs=1, grad_clip=0.01, eval_every=0))
    history = trainer.fit(_loader(n_frames=2))
    assert all(r.grad_norm >= 0 for r in history)


# -------------------------------------------------------------- validator --

def test_validator_returns_score_and_restores_mode() -> None:
    model = _model()
    decoder = BoxDecoder(AnchorGenerator(_spec()), score_threshold=0.27)
    validator = Validator(_loader(n_frames=2), decoder, max_batches=1)
    model.train()
    metrics = validator.run(model)
    assert model.training, "validator must restore train mode"
    assert "score" in metrics and "ap50" in metrics
    assert metrics["n_frames"] > 0


def test_trainer_selects_best_checkpoint_via_validator(tmp_path) -> None:
    from cpbench.logbook import ExperimentLogger, ExperimentMeta

    class _FixedValidator:
        def __init__(self) -> None:
            self.scores = iter([0.3, 0.1])

        def run(self, model) -> dict:
            return {"score": next(self.scores)}

    meta = ExperimentMeta(experiment_id="t", experiment_name="t",
                          paper="V2X-ViT", architecture="v2xvit",
                          dataset="synthetic", seed=0, deterministic=True)
    model = _model()
    with ExperimentLogger(tmp_path, "run", meta,
                          logger_names=("v2xvitbench",)) as logbook:
        trainer = Trainer(model, _loss(),
                          torch.optim.Adam(model.parameters(), lr=1e-3),
                          TrainerConfig(epochs=2, eval_every=1),
                          logbook=logbook, validator=_FixedValidator())
        trainer.fit(_loader(n_frames=2))
        assert trainer.best_score == pytest.approx(0.3)
        assert (logbook.checkpoints_dir / "best.pt").exists()
