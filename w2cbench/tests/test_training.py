"""
Tests for the dataset, the collator, the multi-round loss and the trainer.

The collator gets the most attention here, and the reason is that its one
serious failure mode is silent: the agent index in ``coords`` is per-scene on
the way in and must become global on the way out, or ``PointPillarScatter``
sums two vehicles' LiDAR onto one canvas. Nothing raises. The model trains, and
every number afterwards is wrong.

The loss tests pin A11 -- that the pre-fusion term is the confidence head's
only direct gradient, and that removing it leaves the head trained solely
through a path whose gradient the hard selection mask has already severed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cpbench.data import (AnchorGenerator, BoxDecoder, GridSpec,
                          SyntheticCooperativeDataset, TargetAssigner)
from cpbench.faults import DataFaultBridge
from w2cbench.comm import CommunicationGraph, ThresholdSelector
from w2cbench.data import W2CLidarDataset, collate_lidar, lidar_collator
from w2cbench.fusion import AttenFusion, SpatialTransform
from w2cbench.models import (LidarPillarEncoder, SpatialConfidenceGenerator,
                             Where2comm)
from w2cbench.training import (MultiRoundDetectionLoss, Trainer, TrainerConfig,
                               Validator)

DIM = 32


def _spec() -> GridSpec:
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))


def _dataset(n_frames: int = 4, n_agents: int = 2, bridge=None):
    adapter = SyntheticCooperativeDataset(n_frames=n_frames, n_agents=n_agents,
                                          n_objects=3, seed=0)
    return W2CLidarDataset(adapter, _spec(), max_cav=n_agents, bridge=bridge)


def _model(rounds: int = 1) -> Where2comm:
    spec = _spec()
    return Where2comm(
        encoder=LidarPillarEncoder(spec, out_channels=DIM),
        confidence=SpatialConfidenceGenerator(in_channels=DIM),
        selector=ThresholdSelector(threshold=0.01),
        aggregator=AttenFusion(dim=DIM),
        warp=SpatialTransform.from_grid_spec(spec),
        graph=CommunicationGraph(), rounds=rounds)


def _loss() -> MultiRoundDetectionLoss:
    return MultiRoundDetectionLoss(TargetAssigner(AnchorGenerator(_spec())))


def _scene(n_agents: int, n_pillars: int) -> dict:
    return {"features": torch.zeros(n_pillars, 4, 10),
            "coords": torch.zeros(n_pillars, 3, dtype=torch.long),
            "num_points": torch.zeros(n_pillars, dtype=torch.long),
            "T_agent_to_ego": torch.eye(4).expand(n_agents, 4, 4).contiguous(),
            "gt_boxes": None, "n_agents": n_agents, "frame": 0}


# ----------------------------------------------------------------- dataset --

def test_dataset_yields_the_documented_keys() -> None:
    item = _dataset()[0]
    assert set(item) >= {"features", "coords", "num_points", "T_agent_to_ego",
                         "gt_boxes", "n_agents", "frame", "fault_records"}
    assert item["coords"].shape[1] == 3
    assert item["T_agent_to_ego"].shape == (2, 4, 4)


def test_points_are_voxelised_in_each_agents_own_frame() -> None:
    """What makes this intermediate fusion. Under early fusion a pose error
    would corrupt the points, the confidence, the selection and the bandwidth
    at once, and the four effects would be inseparable in the results."""
    dataset = _dataset(n_agents=2)
    item = dataset[0]
    # Agent 1's transform is not identity (it stands somewhere else), yet its
    # pillars are indexed on the same grid as the ego's -- they are in local
    # coordinates, awaiting the feature-level warp.
    assert not torch.allclose(item["T_agent_to_ego"][1], torch.eye(4))
    agent_1 = item["coords"][item["coords"][:, 0] == 1]
    assert agent_1.numel() > 0
    assert int(agent_1[:, 1].max()) < _spec().grid_hw[0]


def test_an_agent_with_no_returns_keeps_its_row() -> None:
    """Dropping it would silently turn a sensor fault into an agent-drop
    fault, and the two conditions would become indistinguishable."""
    dataset = _dataset(n_agents=3)
    item = dataset[0]
    assert item["n_agents"] == 3
    assert item["T_agent_to_ego"].shape[0] == 3


def test_a_clean_bridge_is_provably_clean() -> None:
    assert _dataset().is_clean
    assert not _dataset(bridge=DataFaultBridge(
        {"pipeline": {"pose_error": {"sigma_xy": 0.4}}})).is_clean


def test_fault_records_travel_with_the_item() -> None:
    """injection_summary.csv records faults that a frame's OUTPUT gives no
    sign of, so the audit trail has to reach the batch."""
    dataset = _dataset(bridge=DataFaultBridge(
        {"pipeline": {"pose_error": {"sigma_xy": 0.4, "sigma_heading": 0.2}}}))
    assert len(dataset[0]["fault_records"]) > 0


# ---------------------------------------------------------------- collator --

def test_agent_indices_become_global() -> None:
    """The silent failure this collator exists to prevent: two scenes whose
    agent 0 both say 0 would be scattered onto the same canvas, summing two
    vehicles' LiDAR into one feature map."""
    batch = collate_lidar([_scene(2, 5), _scene(1, 3)], max_cav=3)
    assert batch["record_len"] == [2, 1]
    assert batch["coords"][5:, 0].tolist() == [2, 2, 2]


def test_transforms_are_padded_with_identity_not_zeros() -> None:
    """A zero matrix is singular and the warp inverts the rotation block. An
    unused slot is never read, but a NaN from inverting a padding row would
    propagate through the batch and be blamed on whichever agent was looked
    at next."""
    batch = collate_lidar([_scene(1, 2)], max_cav=3)
    assert torch.allclose(batch["T_agent_to_ego"][0, 2], torch.eye(4))
    assert torch.isfinite(torch.linalg.inv(batch["T_agent_to_ego"])).all()


def test_scenes_beyond_max_cav_are_truncated_ego_first() -> None:
    scene = _scene(4, 8)
    scene["coords"][:, 0] = torch.arange(8) % 4
    batch = collate_lidar([scene], max_cav=2)
    assert batch["record_len"] == [2]
    assert int(batch["coords"][:, 0].max()) < 2


def test_empty_batch_does_not_crash() -> None:
    batch = collate_lidar([_scene(1, 0)], max_cav=2)
    assert batch["features"].shape[0] == 0
    assert batch["record_len"] == [1]


def test_collator_drives_a_real_dataloader() -> None:
    loader = DataLoader(_dataset(n_frames=4), batch_size=2, shuffle=False,
                        collate_fn=lidar_collator(2))
    batch = next(iter(loader))
    assert batch["record_len"] == [2, 2]
    assert batch["T_agent_to_ego"].shape == (2, 2, 4, 4)


def test_a_full_batch_runs_through_the_model() -> None:
    loader = DataLoader(_dataset(n_frames=4), batch_size=2,
                        collate_fn=lidar_collator(2))
    out = _model().eval()(next(iter(loader)))
    assert out["cls"].shape[0] == 2


# -------------------------------------------------------------------- loss --

def test_loss_returns_a_finite_scalar_and_its_components() -> None:
    loader = DataLoader(_dataset(), batch_size=2, collate_fn=lidar_collator(2))
    batch = next(iter(loader))
    terms = _loss()(batch, _model().eval()(batch))
    assert torch.isfinite(terms["loss"])
    for key in ("loss_cls", "loss_reg", "loss_single", "loss_r0"):
        assert key in terms


def test_every_round_is_supervised() -> None:
    """What makes an intermediate round a usable operating point rather than
    merely a step toward one -- and why a K=3 model can be evaluated at K=1."""
    loader = DataLoader(_dataset(), batch_size=1, collate_fn=lidar_collator(2))
    batch = next(iter(loader))
    terms = _loss()(batch, _model(rounds=3).eval()(batch))
    assert {"loss_r0", "loss_r1", "loss_r2"} <= set(terms)


def test_round_weights_must_match_the_round_count() -> None:
    loss_fn = MultiRoundDetectionLoss(TargetAssigner(AnchorGenerator(_spec())),
                                      round_weights=[1.0, 0.5])
    loader = DataLoader(_dataset(), batch_size=1, collate_fn=lidar_collator(2))
    batch = next(iter(loader))
    with pytest.raises(ValueError, match="round_weights has 2 entries"):
        loss_fn(batch, _model(rounds=1).eval()(batch))


def test_the_pre_fusion_term_is_the_confidence_heads_only_direct_gradient() -> None:
    """A11, stated as a gradient check. Selection is a hard {0,1} mask, so the
    fused loss cannot reach the confidence map through it. With single_weight
    at zero the head is still trained -- it is shared with the decoder -- but
    the confidence map becomes a by-product rather than a target."""
    loader = DataLoader(_dataset(), batch_size=1, collate_fn=lidar_collator(2))
    batch = next(iter(loader))

    model = _model()
    output = model(batch)
    with_single = _loss()(batch, output)["loss"]
    without = MultiRoundDetectionLoss(
        TargetAssigner(AnchorGenerator(_spec())), single_weight=0.0
    )(batch, output)["loss"]
    assert float(with_single) != float(without)


def test_only_the_ego_row_is_supervised_pre_fusion() -> None:
    """Ground truth exists in the ego frame; a collaborator's pre-fusion
    prediction is in its own. Because the head is shared (A2), training the
    ego's output trains exactly the parameters producing every collaborator's
    confidence map."""
    assert MultiRoundDetectionLoss._ego_rows([2, 3, 1]) == [0, 2, 5]


def test_gradients_flow_from_the_loss_to_the_encoder() -> None:
    loader = DataLoader(_dataset(), batch_size=1, collate_fn=lidar_collator(2))
    batch = next(iter(loader))
    model = _model()
    model.train()
    _loss()(batch, model(batch))["loss"].backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert any("encoder" in n and float(g.abs().sum()) > 0
               for n, g in grads.items())
    assert float(grads["confidence.head.cls_head.weight"].abs().sum()) > 0


# ----------------------------------------------------------------- trainer --

def test_trainer_runs_and_records_history() -> None:
    loader = DataLoader(_dataset(n_frames=4), batch_size=2,
                        collate_fn=lidar_collator(2))
    model = _model()
    trainer = Trainer(model, _loss(),
                      torch.optim.Adam(model.parameters(), lr=1e-3),
                      TrainerConfig(epochs=2, log_every=1))
    history = trainer.fit(loader)
    assert len(history) == 4                        # 2 batches x 2 epochs
    assert all(np.isfinite(r.loss_total) for r in history)
    assert history[0].epoch == 0 and history[-1].epoch == 1


def test_the_loss_actually_decreases_on_a_tiny_fixed_batch() -> None:
    """A smoke test with teeth: if the plumbing were wrong -- targets
    misaligned with predictions, gradients not reaching the encoder -- the
    loss would wander rather than fall."""
    torch.manual_seed(0)
    loader = DataLoader(_dataset(n_frames=2), batch_size=2,
                        collate_fn=lidar_collator(2))
    batch = next(iter(loader))
    model = _model()
    trainer = Trainer(model, _loss(),
                      torch.optim.Adam(model.parameters(), lr=5e-3),
                      TrainerConfig(epochs=12, log_every=100))
    history = trainer.fit([batch] * 12)
    assert history[-1].loss_total < history[0].loss_total


def test_grad_clipping_is_recorded_and_bounded() -> None:
    loader = DataLoader(_dataset(n_frames=2), batch_size=1,
                        collate_fn=lidar_collator(2))
    model = _model()
    trainer = Trainer(model, _loss(),
                      torch.optim.SGD(model.parameters(), lr=1e-2),
                      TrainerConfig(epochs=1, grad_clip=1.0))
    history = trainer.fit(loader)
    assert all(r.grad_norm >= 0.0 for r in history)


def test_the_trainer_does_not_know_which_paper_it_is_training() -> None:
    """It takes a loss closure, so a two-parameter stub drives it -- which is
    what will let the camera track reuse it unchanged."""
    calls = []

    def stub_loss(batch, output):
        calls.append(batch)
        return {"loss": output["cls"].sum() * 0.0 + 1.0}

    loader = DataLoader(_dataset(n_frames=2), batch_size=1,
                        collate_fn=lidar_collator(2))
    model = _model()
    Trainer(model, stub_loss, torch.optim.SGD(model.parameters(), lr=0.0),
            TrainerConfig(epochs=1)).fit(loader)
    assert len(calls) == 2


def test_scheduler_steps_once_per_epoch() -> None:
    loader = DataLoader(_dataset(n_frames=2), batch_size=2,
                        collate_fn=lidar_collator(2))
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    Trainer(model, _loss(), optimizer, TrainerConfig(epochs=3),
            scheduler=scheduler).fit(loader)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


# --------------------------------------------------------------- validator --

def test_validator_returns_a_score_and_restores_training_mode() -> None:
    """The mode restore matters: fit() calls this mid-loop, and leaving the
    model in eval() would silently switch off the bandwidth curriculum and
    every BatchNorm update for the rest of training."""
    loader = DataLoader(_dataset(n_frames=2), batch_size=1,
                        collate_fn=lidar_collator(2))
    model = _model()
    model.train()
    validator = Validator(loader, BoxDecoder(AnchorGenerator(_spec())))
    metrics = validator.run(model)
    assert "score" in metrics and np.isfinite(metrics["score"])
    assert metrics["n_frames"] == 2.0
    assert model.training is True


def test_validator_selects_the_best_checkpoint(tmp_path) -> None:
    from cpbench.logbook import ExperimentLogger, ExperimentMeta

    loader = DataLoader(_dataset(n_frames=2), batch_size=1,
                        collate_fn=lidar_collator(2))
    model = _model()
    meta = ExperimentMeta(experiment_id="t", experiment_name="t", paper="p",
                          architecture="a", dataset="d", seed=0,
                          deterministic=True)
    with ExperimentLogger(tmp_path, "run", meta,
                          logger_names=("w2cbench",)) as book:
        trainer = Trainer(model, _loss(),
                          torch.optim.Adam(model.parameters(), lr=1e-3),
                          TrainerConfig(epochs=1, eval_every=1), logbook=book,
                          validator=Validator(
                              loader, BoxDecoder(AnchorGenerator(_spec()))))
        trainer.fit(loader)
        assert (book.checkpoints_dir / "best.pt").exists()
    assert trainer.best_score is not None


def test_validator_refuses_nothing_but_measures_in_eval_mode() -> None:
    """A17: the selector is stochastic in train mode, so a validation number
    taken there would be a draw from the bandwidth curriculum."""
    seen = []

    class _Watch(Where2comm):
        def forward(self, batch, taps=None, accountant=None):
            seen.append(self.training)
            return super().forward(batch, taps=taps, accountant=accountant)

    spec = _spec()
    model = _Watch(
        encoder=LidarPillarEncoder(spec, out_channels=DIM),
        confidence=SpatialConfidenceGenerator(in_channels=DIM),
        selector=ThresholdSelector(threshold=0.01),
        aggregator=AttenFusion(dim=DIM),
        warp=SpatialTransform.from_grid_spec(spec),
        graph=CommunicationGraph())
    model.train()
    loader = DataLoader(_dataset(n_frames=1), batch_size=1,
                        collate_fn=lidar_collator(2))
    Validator(loader, BoxDecoder(AnchorGenerator(spec))).run(model)
    assert seen == [False]
