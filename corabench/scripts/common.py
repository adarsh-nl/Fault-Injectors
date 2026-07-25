"""
common.py
---------
Builders shared by the train / evaluate / benchmark entry points: everything
is constructed from the resolved config dict, so scripts stay thin and no
source edit is ever needed to change an experiment.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset

from src.datasets import load_dataset

from ..data.cooperative import CoRADataset
from cpbench.data.preprocessing import AnchorGenerator, GridSpec, TargetAssigner
from cpbench.data.synthetic import SyntheticCooperativeDataset
from cpbench.faults.bridge import DataFaultBridge
from cpbench.logbook.env import capture_environment, seed_everything
from cpbench.logbook.experiment import ExperimentLogger
from cpbench.logbook.schema import ExperimentMeta
from ..models.cora import CoRAModel
from cpbench.observation.recorders import DriftTap, StatsTap, TensorDumpTap
from cpbench.observation.taps import TapSet

logger = logging.getLogger(__name__)


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def build_grid(ds_cfg: Dict[str, Any]) -> GridSpec:
    g = ds_cfg["grid"]
    return GridSpec(voxel_size=tuple(g["voxel_size"]),
                    point_range=tuple(g["point_range"]),
                    downsample=int(g.get("downsample", 2)))


def build_adapters(ds_cfg: Dict[str, Any],
                   split: Optional[str]) -> List[Any]:
    """Instantiate `src.datasets` adapters for one split."""
    adapter = ds_cfg["adapter"]
    if adapter == "synthetic":
        seed = {"train": 0, "validate": 1, "test": 2}.get(split or "test", 2)
        return [SyntheticCooperativeDataset(
            n_frames=int(ds_cfg.get("n_frames", 12)),
            n_agents=int(ds_cfg.get("n_agents", 3)),
            n_objects=int(ds_cfg.get("n_objects", 4)),
            seed=seed)]
    root = Path(ds_cfg["root"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"dataset root {root} does not exist on this machine. The value "
            f"in configs/dataset/{ds_cfg['name']}.yaml is a placeholder -- "
            f"override it: dataset.root=/path/to/your/{adapter} "
            f"(on the UT HPC the data is read-only under /deepstore/datasets)")
    if adapter in ("opv2v", "v2xset"):
        split_dir = root / split if split else root
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"split directory {split_dir} not found under dataset root "
                f"{root}; check dataset.train_split / dataset.test_split")
        wanted = ds_cfg.get("scenarios")
        scen_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if wanted:
            scen_dirs = [p for p in scen_dirs if p.name in set(wanted)]
        if not scen_dirs:
            raise FileNotFoundError(
            f"no scenario dirs under {split_dir}. If you passed "
            f"dataset.scenarios explicitly, note that YAML 1.1 reads "
            f"underscores as digit separators, so [2021_08_16_22_26_54] "
            f"parses as the integer 20210816222654 rather than the "
            f"directory name. Quote it: dataset.scenarios='[\"NAME\"]'.")
        return [load_dataset(adapter, scenario_dir=str(p)) for p in scen_dirs]
    if adapter == "dair-v2x":
        return [load_dataset(adapter, root=str(root))]
    raise ValueError(f"unknown adapter {adapter!r}")


def build_cora_dataset(ds_cfg: Dict[str, Any], grid: GridSpec,
                       adapters: List[Any],
                       bridge: Optional[DataFaultBridge]):
    """Wrap adapters into CoRADatasets (shared anchors/bridge), concatenated."""
    anchor_gen = AnchorGenerator(grid)
    assigner = TargetAssigner(anchor_gen)
    sets = [CoRADataset(
        adapter, grid, bridge=bridge, anchor_generator=anchor_gen,
        target_assigner=assigner,
        max_points_per_pillar=int(ds_cfg.get("max_points_per_pillar", 32)),
        max_pillars=int(ds_cfg.get("max_pillars", 20000)),
        max_agents=int(ds_cfg.get("max_agents", 5)),
        comm_range_m=float(ds_cfg.get("comm_range_m", 70.0)),
        categories=ds_cfg.get("categories")) for adapter in adapters]
    return sets[0] if len(sets) == 1 else ConcatDataset(sets)


def build_model(cfg: Dict[str, Any], grid: GridSpec) -> CoRAModel:
    m = cfg["model"]
    return CoRAModel(
        grid,
        channels=int(m["channels"]), num_anchors=int(m["num_anchors"]),
        num_classes=int(m["num_classes"]),
        vfe_channels=int(m["vfe_channels"]),
        block_channels=tuple(m["block_channels"]),
        block_strides=tuple(m["block_strides"]),
        block_layers=tuple(m["block_layers"]),
        upsample_channels=int(m["upsample_channels"]),
        cit=m.get("cit"), cssm=m.get("cssm"), lc=m.get("lc"),
        pac=m.get("pac"), fusion=m.get("fusion"),
        teacher_enabled=bool(m.get("teacher_enabled", True)),
        score_threshold=float(m.get("score_threshold", 0.2)))


def build_taps(cfg: Dict[str, Any], out_dir: Path):
    """(TapSet or None, StatsTap or None) from the taps config group."""
    t = cfg.get("taps") or {}
    children, stats_tap = [], None
    if t.get("stats"):
        stats_tap = StatsTap()
        children.append(TapSet([stats_tap],
                               include=t["stats"].get("include")))
    if t.get("dump"):
        d = t["dump"]
        dump = TensorDumpTap(out_dir / "taps",
                             every_n=int(d.get("every_n", 1)),
                             max_dumps=d.get("max_dumps"))
        children.append(TapSet([dump], include=d.get("include")))
    if t.get("drift"):
        children.append(TapSet([DriftTap(t["drift"])]))
    if not children:
        return None, None
    return TapSet(children), stats_tap


def build_experiment(cfg: Dict[str, Any], suffix: str = "") -> ExperimentLogger:
    """Seed, capture environment, open the ExperimentLogger."""
    seed_everything(int(cfg["seed"]), bool(cfg.get("deterministic", True)))
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    short = hashlib.sha1(blob).hexdigest()[:8]
    name = cfg["experiment_name"] + (f"_{suffix}" if suffix else "")
    meta = ExperimentMeta(
        experiment_id=f"{name}_{stamp}_{short}",
        experiment_name=name,
        paper=cfg.get("paper", ""),
        architecture=cfg["model"]["name"],
        dataset=cfg["dataset"]["name"],
        seed=int(cfg["seed"]),
        deterministic=bool(cfg.get("deterministic", True)),
        fault_config=cfg.get("faults", {}),
        tap_config=cfg.get("taps", {}),
        assumptions=cfg["model"].get("assumptions", {}),
        environment=capture_environment(),
        resolved_config=cfg,
        started_at=stamp)
    return ExperimentLogger(cfg.get("results_dir", "results"), name, meta,
                            log_predictions=bool(cfg.get("log_predictions")),
                            logger_names=("corabench",))


def load_checkpoint_into(model: CoRAModel, path: str,
                         device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    logger.info("loaded checkpoint %s (epoch %s)", path, ckpt.get("epoch", "?"))
