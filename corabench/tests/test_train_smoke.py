"""End-to-end smoke: train 1 epoch on synthetic data, validate, benchmark.

CPU-only and small; exercises the whole stack the way the scripts do:
dataset -> bridge -> model -> losses -> trainer -> validator -> clean +
fault benchmark runners -> logger outputs on disk.
"""

import torch

from corabench.data.cooperative import CoRADataset
from corabench.evaluation.benchmark import (CleanBenchmarkRunner,
                                            FaultBenchmarkRunner)
from corabench.faults.bridge import DataFaultBridge
from corabench.logbook.experiment import ExperimentLogger
from corabench.logbook.schema import ExperimentMeta
from corabench.training.losses import CoRALoss
from corabench.training.trainer import Trainer


def _explog(tmp_path):
    meta = ExperimentMeta(
        experiment_id="smoke", experiment_name="smoke", paper="CoRA",
        architecture="cora", dataset="synthetic", seed=0, deterministic=False,
        resolved_config={})
    return ExperimentLogger(tmp_path, "smoke", meta)


def test_train_validate_benchmark(tmp_path, tiny_model, dataset, adapter,
                                  grid):
    device = torch.device("cpu")
    explog = _explog(tmp_path)
    cfg = {"epochs": 1, "batch_size": 2, "num_workers": 0, "amp": False,
           "optimizer": {"lr": 1e-3}, "scheduler": {"milestones": [1]},
           "val_every": 1, "log_every": 1}
    trainer = Trainer(tiny_model, dataset, dataset, CoRALoss(), explog, cfg,
                      device=device)
    val = trainer.fit()
    assert "ap50" in val
    assert (explog.checkpoints_dir / "last.pt").exists()

    def factory(bridge):
        return CoRADataset(adapter, grid, bridge=bridge,
                           anchor_generator=dataset.anchor_generator,
                           target_assigner=dataset.target_assigner,
                           max_points_per_pillar=16, max_pillars=4000)

    clean = CleanBenchmarkRunner(tiny_model, factory, device, explog,
                                 dataset_name="synthetic").run(max_batches=2)
    assert clean.detection["n_frames"] > 0
    assert clean.comm["mb_per_frame"] > 0

    runner = FaultBenchmarkRunner(tiny_model, factory, device, explog, clean,
                                  dataset_name="synthetic",
                                  bridge_kwargs={"seed": 3})
    results = runner.run(
        [{"pose_error": {"sigma_xy": 0.6, "sigma_heading": 5.0}}],
        max_batches=2)
    name, faulted, rob = results[0]
    assert faulted.n_faults > 0
    assert "flip_rate" in rob and "delta_ap70" in rob
    explog.close()
    assert (explog.dir / "fault_statistics.csv").exists()
    assert (explog.dir / "injection_summary.csv").exists()


def test_checkpoint_resume(tmp_path, tiny_model, dataset):
    explog = _explog(tmp_path)
    cfg = {"epochs": 1, "batch_size": 2, "amp": False, "val_every": 5,
           "log_every": 10}
    trainer = Trainer(tiny_model, dataset, None, CoRALoss(), explog, cfg,
                      device=torch.device("cpu"))
    trainer.fit()
    path = trainer.save_checkpoint(0, "resume_test")
    trainer2 = Trainer(tiny_model, dataset, None, CoRALoss(), explog, cfg,
                       device=torch.device("cpu"))
    trainer2.resume(path)
    assert trainer2.start_epoch == 1
    explog.close()
