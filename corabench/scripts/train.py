"""Train entry point (spec §4). Config-driven via cpbench.utils.config:

    python -m corabench.scripts.train                       # synthetic smoke
    python -m corabench.scripts.train dataset=opv2v dataset.root=/path
    python -m corabench.scripts.train model.pac_enabled=false
"""

from __future__ import annotations

import pathlib
import sys

import torch

from cpbench.data import GridSpec
from cpbench.logbook.env import seed_everything
from cpbench.utils.config import load_config

from ..data import CoRADataset
from ..models import CoRAModel
from ..selfcheck import static_source_checks
from ..training import CoRALoss, Trainer

CONFIG_ROOT = (pathlib.Path(__file__).resolve().parents[1] / "configs" / "config.yaml")


def build_dataset(cfg: dict):
    name = cfg["dataset"]["name"]
    if name == "synthetic":
        from cpbench.data.synthetic import SyntheticCooperativeDataset
        return SyntheticCooperativeDataset(
            n_frames=cfg["dataset"].get("n_frames", 32),
            n_agents=cfg["dataset"].get("n_agents", 3))
    from src.datasets import load_dataset
    return load_dataset(name, **cfg["dataset"].get("kwargs", {}))


def main(argv=None) -> int:
    cfg = load_config(CONFIG_ROOT, overrides=argv or sys.argv[1:])
    seed_everything(cfg.get("seed", 2026),
                    cfg.get("deterministic", True))
    static_source_checks()                       # spec §5.1/5.4/5.5

    grid = GridSpec(tuple(cfg["model"]["voxel_size"]),
                    tuple(cfg["model"]["point_range"]))
    model = CoRAModel(
        grid.grid_hw, channels=cfg["model"]["channels"],
        reg_dim=cfg["model"]["reg_dim"],
        cit_strategy=cfg["model"]["cit_strategy"],
        teacher_enabled=cfg["model"]["teacher_enabled"],
        pac_enabled=cfg["model"]["pac_enabled"])
    dataset = CoRADataset(build_dataset(cfg), grid,
                          reg_dim=cfg["model"]["reg_dim"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, dataset,
                      lr=cfg["trainer"]["lr"],
                      weight_decay=cfg["trainer"]["weight_decay"],
                      batch_size=cfg["trainer"]["batch_size"],
                      milestones=cfg["trainer"]["milestones"],
                      grad_clip=cfg["trainer"]["grad_clip"],
                      amp=cfg["trainer"]["amp"],
                      num_workers=cfg["trainer"].get("num_workers", 4),
                      device=device,
                      loss=CoRALoss(reg_dim=cfg["model"]["reg_dim"]))
    for epoch in range(cfg["trainer"]["epochs"]):
        stats = trainer.train_epoch(epoch)
        print("epoch %d: %s" % (epoch, {k: round(v, 4)
                                        for k, v in stats.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
