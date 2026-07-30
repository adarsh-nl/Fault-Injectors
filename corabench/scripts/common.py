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
from cpbench.utils.paths import require_dataset_root
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
                   split: Optional[str],
                   cfg_data_root: Optional[str] = None) -> List[Any]:
    """Instantiate `src.datasets` adapters for one split.

    ``cfg_data_root`` is the resolved ``data_root`` value, passed only so a
    missing-path error can say which rule produced the path it tried.
    """
    adapter = ds_cfg["adapter"]
    if adapter == "synthetic":
        seed = {"train": 0, "validate": 1, "test": 2}.get(split or "test", 2)
        return [SyntheticCooperativeDataset(
            n_frames=int(ds_cfg.get("n_frames", 12)),
            n_agents=int(ds_cfg.get("n_agents", 3)),
            n_objects=int(ds_cfg.get("n_objects", 4)),
            seed=seed)]
    root = require_dataset_root(ds_cfg["root"], config_value=cfg_data_root,
                                what=f"{ds_cfg['name']} dataset")
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
                       bridge: Optional[DataFaultBridge],
                       reg_dim: int = 7):
    """Wrap adapters into CoRADatasets (shared anchors/bridge), concatenated.

    `reg_dim` must be the SAME value the model is built with -- it decides the
    width of the regression targets, and at >= 8 whether the assigner writes
    the cos channel at all. Callers pass cfg["model"]["reg_dim"];
    assert_reg_dim_consistent() verifies it against the model at startup.
    """
    anchor_gen = AnchorGenerator(grid)
    assigner = TargetAssigner(anchor_gen, reg_dim=reg_dim)
    sets = [CoRADataset(
        adapter, grid, bridge=bridge, anchor_generator=anchor_gen,
        target_assigner=assigner,
        max_points_per_pillar=int(ds_cfg.get("max_points_per_pillar", 32)),
        max_pillars=int(ds_cfg.get("max_pillars", 20000)),
        max_agents=int(ds_cfg.get("max_agents", 5)),
        comm_range_m=float(ds_cfg.get("comm_range_m", 70.0)),
        categories=ds_cfg.get("categories")) for adapter in adapters]
    return sets[0] if len(sets) == 1 else ConcatDataset(sets)


def assert_reg_dim_consistent(model, dataset, loss) -> int:
    """Fail loudly at startup if any producer or consumer of the regression
    channels disagrees about their width.

    This checks AGREEMENT, never a hardcoded value: cobevt / v2xvit /
    where2comm / lgcp run every component at 7 and pass, corabench runs every
    component at 8 and passes, and only a genuine mismatch fails. A mismatch
    is otherwise near-silent -- the gather in PACModule._decode_params happily
    returns 7 of 8 channels without erroring, which drops the cos channel and
    reverts yaw to the 180-degree-ambiguous decode while everything still
    trains and still reports plausible AP.

    Covers every component that PRODUCES or CONSUMES reg channels:
    assigner (produces targets), both DetectionHeads (produce), PAC (produces
    via fuse_reg, consumes in _decode_params), loss (consumes), decoder
    (consumes).
    """
    found = {
        "local_head": model.local_head.reg_dim,
        "lc_head": model.lc_head.reg_dim,
        # PAC derives reg_dim = nreg_ch // num_anchors rather than being told
        # it. Included anyway: PAC's decode indexes reg[:, 6] and reg[:, 7] at
        # FIXED positions, so it REQUIRES reg_dim >= 8 to decode via atan2 and
        # silently falls back to the ambiguous asin path below that. That
        # coupling is documented here rather than left implicit.
        "pac": model.pac.reg_dim,
        "decoder": model.adaptive.decoder.reg_dim,
    }
    if loss is not None:            # evaluate.py has no loss to check
        found["loss"] = loss.reg_dim
    # build_cora_dataset returns a ConcatDataset when a split has several
    # scenarios, so the assigner cannot be reached by plain attribute access.
    subsets = getattr(dataset, "datasets", [dataset]) if dataset is not None else []
    for i, sub in enumerate(subsets):
        assigner = getattr(sub, "target_assigner", None)
        if assigner is not None:
            found[f"assigner[{i}]"] = assigner.reg_dim

    if len(set(found.values())) > 1:
        detail = "\n".join(f"    {k:<14} reg_dim={v}"
                           for k, v in sorted(found.items()))
        raise ValueError(
            "reg_dim mismatch between components that produce or consume the "
            "regression channels:\n" + detail +
            "\nAll must agree. Set model.reg_dim in the config; it is the "
            "single source of truth and is threaded to every site.")
    return next(iter(found.values()))


def build_model(cfg: Dict[str, Any], grid: GridSpec) -> CoRAModel:
    m = cfg["model"]
    return CoRAModel(
        grid,
        reg_dim=int(m.get("reg_dim", 7)),
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
