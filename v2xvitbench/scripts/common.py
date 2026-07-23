"""
common.py
---------
Config -> objects. Every builder the three CLI entry points share.

Design rule
    Nothing in this file makes a decision. It reads resolved config and
    assembles the modules that implement the paper. If a value is not in the
    config, it is not configurable -- no run needs a source edit, so
    anything a sweep might vary lives in ``v2xvitbench/configs``.

    The converse matters just as much: nothing *outside* this file reads
    config. Every model stage takes typed arguments, which is what lets a
    test substitute a stub for any of them and what keeps a config key from
    being silently reinterpreted three modules deep.

Eager validation
    Tap locations, the fusion-stride identity and the MSwin window geometry
    are all checked at load time. A cluster job that dies in the first
    second is cheap; one that runs for six hours and writes an empty
    ``taps.csv`` is not.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from cpbench.data import (AnchorGenerator, BoxDecoder, GridSpec,
                          SyntheticCooperativeDataset, TargetAssigner)
from cpbench.logbook import (ExperimentLogger, ExperimentMeta,
                             capture_environment, seed_everything)
from cpbench.observation import DriftTap, StatsTap, TapSet, TensorDumpTap
from cpbench.utils import load_config

from v2xvitbench.data import V2XVitLidarDataset, v2xvit_collator
from v2xvitbench.evaluation.tester import DetectionTester
from v2xvitbench.faults.registry import build_bridge, build_metadata_bridge
from v2xvitbench.models import V2XViT
from v2xvitbench.observation import validate_location
from v2xvitbench.training.losses import V2XViTLoss
from v2xvitbench.training.trainer import TrainerConfig

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


# -- config -----------------------------------------------------------------

def load(overrides: Optional[List[str]] = None,
         config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Compose the resolved config from groups and CLI overrides."""
    cfg = load_config(config_path or CONFIG_ROOT, overrides)
    validate(cfg)
    return cfg


def validate(cfg: Dict[str, Any]) -> None:
    """Fail now on anything that would otherwise fail late or silently."""
    track = str(cfg["dataset"].get("track", "lidar"))
    if track != "lidar":
        raise ValueError(
            f"dataset track {track!r}: V2X-ViT is LiDAR-only; a camera "
            "dataset cannot drive this model")

    for group in ("stats", "dump"):
        for name in (((cfg.get("taps") or {}).get(group) or {}).get("include")
                     or []):
            if "*" not in name:
                validate_location(name)

    model = cfg["model"]
    strides = list(model["encoder"].get("block_strides", (2, 2, 2)))
    shrink_stride = int(model["shrink"].get("stride", 2))
    declared = int(cfg["dataset"]["grid"].get("downsample", 4))
    expected = int(strides[0]) * shrink_stride
    if declared != expected:
        raise ValueError(
            f"dataset.grid.downsample={declared} disagrees with "
            f"model.encoder.block_strides[0]={strides[0]} x "
            f"model.shrink.stride={shrink_stride} = {expected}; anchors, the "
            "box decoder and the STTF warp are all sized from the fusion "
            "grid, so the identity must hold")

    mswin = model["fusion"]["mswin"]
    lists = (list(mswin["window_sizes"]), list(mswin["heads"]),
             list(mswin["dim_heads"]))
    if len({len(lst) for lst in lists}) != 1:
        raise ValueError(
            "model.fusion.mswin window_sizes, heads and dim_heads must have "
            f"the same length (one entry per branch); got {lists}")

    grid = build_grid_spec(cfg)
    height, width = grid.feature_hw
    for window in mswin["window_sizes"]:
        if height % int(window) or width % int(window):
            raise ValueError(
                f"fused grid {height}x{width} does not divide by MSwin "
                f"window {window}; adjust dataset.grid or "
                "model.fusion.mswin.window_sizes")

    hmsa = model["fusion"]["hmsa"]
    if int(hmsa.get("num_relations", 4)) != int(hmsa.get("num_types", 2)) ** 2:
        raise ValueError(
            "model.fusion.hmsa.num_relations must equal num_types^2 "
            "(ordered (receiver, sender) pairs)")


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    choice = str(cfg.get("device", "auto"))
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def apply_seed(cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))


# -- geometry ---------------------------------------------------------------

def build_grid_spec(cfg: Dict[str, Any]) -> GridSpec:
    """The FUSION GridSpec (downsample = backbone stride x shrink stride)."""
    grid = cfg["dataset"]["grid"]
    return GridSpec(voxel_size=tuple(grid["voxel_size"]),
                    point_range=tuple(grid["point_range"]),
                    downsample=int(grid.get("downsample", 4)))


# -- datasets ---------------------------------------------------------------

_DEFAULT_SPLIT_DIRS = {"train": "train", "val": "validate", "test": "test"}


def build_adapters(cfg: Dict[str, Any],
                   split: Optional[str] = None) -> List[Any]:
    """The ``src.datasets`` adapters for one split, one per scenario.

    Returns a LIST because OPV2V-family datasets ship many independent
    scenario folders, and each becomes its own dataset so that a latency
    fault re-reads earlier frames of the *same* scenario rather than across
    a scenario boundary -- which a single globally-indexed adapter could not
    guarantee.
    """
    dataset = cfg["dataset"]
    name = str(dataset["adapter"])
    if name == "synthetic":
        # The split only varies the seed, so train / val / test are different
        # scenes: an identical val set gives a meaningless validation signal.
        seed = int(cfg["seed"]) + {"train": 0, "val": 1, "test": 2}.get(
            split or "test", 0)
        return [SyntheticCooperativeDataset(
            n_frames=int(dataset.get("n_frames", 16)),
            n_agents=int(dataset.get("n_agents", 3)),
            n_objects=int(dataset.get("n_objects", 4)), seed=seed)]

    if name in ("opv2v", "v2xset"):
        from src.datasets import load_dataset
        root = Path(dataset["root"])
        mapping = dataset.get("split") or _DEFAULT_SPLIT_DIRS
        split_dir = root / mapping.get(split, split) if split else root
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"dataset split directory {split_dir} not found. Set "
                f"dataset.root and check the split map {dataset.get('split')}. "
                "On the UT HPC the data is read-only under /deepstore.")
        wanted = dataset.get("scenarios")
        scenarios = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if wanted:
            scenarios = [p for p in scenarios if p.name in set(wanted)]
        if not scenarios:
            raise FileNotFoundError(f"no scenario directories under {split_dir}")
        return [load_dataset(name, scenario_dir=str(p)) for p in scenarios]

    raise ValueError(
        f"unknown dataset adapter {name!r}; expected synthetic, opv2v or "
        "v2xset")


def build_bridge_for(cfg: Dict[str, Any],
                     overrides: Optional[Dict[str, Any]] = None):
    """The physical bridge for a run. ``overrides={}`` forces a clean bridge.

    The empty-dict override is how a benchmark's clean reference is built:
    ``{}`` is explicit "no faults", distinct from ``None`` which means "fall
    back to the config". Conflating them is how a clean run silently
    inherits a sweep's faults.
    """
    fault_cfg = overrides if overrides is not None else cfg.get("faults")
    return build_bridge(fault_cfg or None,
                        fps=float(cfg["dataset"].get("fps", 10.0)),
                        seed=int(cfg["seed"]))


def build_metadata_for(cfg: Dict[str, Any],
                       overrides: Optional[Dict[str, Any]] = None):
    """The metadata bridge for a run, from the same config block."""
    fault_cfg = overrides if overrides is not None else cfg.get("faults")
    return build_metadata_bridge(fault_cfg or None, seed=int(cfg["seed"]))


def build_dataset(cfg: Dict[str, Any], bridge=None,
                  split: Optional[str] = None):
    """The dataset for one split, wrapping ``bridge`` (clean if None)."""
    dataset_cfg = cfg["dataset"]
    grid = build_grid_spec(cfg)
    sets = [V2XVitLidarDataset(
        adapter, grid, max_cav=int(dataset_cfg.get("max_cav", 5)),
        bridge=bridge, categories=dataset_cfg.get("categories"),
        max_points_per_pillar=int(dataset_cfg.get("max_points_per_pillar", 32)),
        max_pillars=int(dataset_cfg.get("max_pillars", 32000)),
        force_infra=dataset_cfg.get("force_infra"))
        for adapter in build_adapters(cfg, split)]
    if len(sets) == 1:
        return sets[0]
    from torch.utils.data import ConcatDataset
    return ConcatDataset(sets)


def build_collator(cfg: Dict[str, Any]):
    return v2xvit_collator(int(cfg["dataset"].get("max_cav", 5)))


# -- model ------------------------------------------------------------------

def build_model(cfg: Dict[str, Any]) -> V2XViT:
    """Assemble V2XViT from the model config group."""
    model = cfg["model"]
    encoder = model["encoder"]
    shrink = model["shrink"]
    fusion = model["fusion"]
    hmsa = fusion["hmsa"]
    mswin = fusion["mswin"]
    rte = fusion.get("rte") or {}
    head = model["head"]

    return V2XViT(
        build_grid_spec(cfg),
        max_cav=int(cfg["dataset"].get("max_cav", 5)),
        vfe_channels=int(encoder.get("vfe_channels", 64)),
        block_channels=list(encoder.get("block_channels", (64, 128, 256))),
        block_strides=list(encoder.get("block_strides", (2, 2, 2))),
        block_layers=list(encoder.get("block_layers", (3, 5, 8))),
        upsample_channels=int(encoder.get("upsample_channels", 128)),
        encoder_out_channels=int(encoder["out_channels"]),
        shrink_channels=int(shrink["channels"]),
        shrink_stride=int(shrink.get("stride", 2)),
        shrink_kernel=int(shrink.get("kernel", 3)),
        compression_factor=int(model.get("compression", 0)),
        depth=int(fusion["depth"]),
        num_blocks=int(fusion.get("num_blocks", 1)),
        hmsa_heads=int(hmsa["heads"]),
        hmsa_dim_head=int(hmsa["dim_head"]),
        num_types=int(hmsa.get("num_types", 2)),
        num_relations=int(hmsa.get("num_relations", 4)),
        dropout=float(fusion.get("dropout", 0.0)),
        window_sizes=list(mswin["window_sizes"]),
        mswin_heads=list(mswin["heads"]),
        mswin_dim_heads=list(mswin["dim_heads"]),
        relative_pos_embedding=bool(mswin.get("relative_pos_embedding", True)),
        fusion_method=str(mswin.get("fusion_method", "split_attn")),
        mlp_dim=int(fusion.get("mlp_dim", 256)),
        use_rte=bool(rte.get("enabled", True)),
        rte_ratio=int(rte.get("ratio", 2)),
        max_delay=int(rte.get("max_delay", 100)),
        use_roi_mask=bool(fusion.get("use_roi_mask", True)),
        num_anchors=int(head["num_anchors"]),
        num_classes=int(head["num_classes"]))


# -- objective and evaluation -----------------------------------------------

def build_loss(cfg: Dict[str, Any]) -> V2XViTLoss:
    loss = cfg["model"]["loss"]
    head = cfg["model"]["head"]
    return V2XViTLoss(
        TargetAssigner(AnchorGenerator(build_grid_spec(cfg))),
        alpha=float(loss.get("alpha", 0.25)),
        gamma=float(loss.get("gamma", 2.0)),
        cls_weight=float(loss.get("cls_weight", 1.0)),
        reg_weight=float(loss.get("reg_weight", 2.0)),
        num_classes=int(head["num_classes"]))


def build_decoder(cfg: Dict[str, Any]) -> BoxDecoder:
    return BoxDecoder(AnchorGenerator(build_grid_spec(cfg)),
                      score_threshold=float(cfg["model"].get(
                          "score_threshold", 0.27)))


def build_trainer_config(cfg: Dict[str, Any]) -> TrainerConfig:
    trainer = cfg["trainer"]
    return TrainerConfig(
        epochs=int(trainer["epochs"]),
        grad_clip=trainer.get("grad_clip"),
        amp=bool(trainer.get("amp", False)),
        log_every=int(trainer.get("log_every", 20)),
        eval_every=int(trainer.get("eval_every", 1)),
        checkpoint_every=int(trainer.get("checkpoint_every", 0)))


def build_tester_factory(cfg: Dict[str, Any], device: torch.device):
    """A ``(condition, reference) -> DetectionTester`` factory for the runner.

    The runner is fault-agnostic; this closure is where a condition turns
    into a dataset wrapping that condition's plane-1 bridge plus the
    plane-2 metadata bridge the tester applies post-collate.
    """
    collate = build_collator(cfg)
    decoder = build_decoder(cfg)
    max_frames = cfg.get("max_frames")

    def make(condition, reference):
        overrides = condition.config or {}
        dataset = build_dataset(cfg, build_bridge_for(cfg, overrides),
                                split="test")
        return DetectionTester(
            dataset, decoder, collate, device=device, reference=reference,
            metadata_bridge=build_metadata_for(cfg, overrides),
            max_frames=max_frames)
    return make


# -- optimiser --------------------------------------------------------------

def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    trainer = cfg["trainer"]
    name = str(trainer.get("optimizer", "adam")).lower()
    params = dict(lr=float(trainer["lr"]),
                  weight_decay=float(trainer.get("weight_decay", 0.0)))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(),
                                 eps=float(trainer.get("eps", 1e-8)), **params)
    if name == "adam":
        return torch.optim.Adam(model.parameters(),
                                eps=float(trainer.get("eps", 1e-8)), **params)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(),
                               momentum=float(trainer.get("momentum", 0.9)),
                               **params)
    raise ValueError(f"unknown optimizer {name!r}; expected adam, adamw or sgd")


def build_scheduler(cfg: Dict[str, Any], optimizer):
    trainer = cfg["trainer"]
    name = str(trainer.get("scheduler", "none")).lower()
    if name == "none":
        return None
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(trainer.get("lr_steps", [])),
            gamma=float(trainer.get("lr_gamma", 0.1)))
    if name in ("cosine", "cosine_warm"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(trainer["epochs"]),
            eta_min=float(trainer.get("lr_min", 0.0)))
    raise ValueError(f"unknown scheduler {name!r}")


# -- observation ------------------------------------------------------------

def build_taps(cfg: Dict[str, Any], out_dir: Path
               ) -> Tuple[Optional[TapSet], Optional[StatsTap]]:
    """``(TapSet or None, StatsTap or None)`` from the taps config group."""
    taps_cfg = cfg.get("taps") or {}
    children: List[TapSet] = []
    stats_tap: Optional[StatsTap] = None
    if taps_cfg.get("stats"):
        stats_tap = StatsTap()
        children.append(TapSet([stats_tap],
                               include=taps_cfg["stats"].get("include")))
    if taps_cfg.get("dump"):
        dump = taps_cfg["dump"]
        children.append(TapSet(
            [TensorDumpTap(out_dir / "taps", every_n=int(dump.get("every_n", 1)),
                           max_dumps=dump.get("max_dumps"))],
            include=dump.get("include")))
    if taps_cfg.get("drift"):
        children.append(TapSet([DriftTap(taps_cfg["drift"])]))
    if not children:
        return None, None
    return TapSet(children), stats_tap


# -- logging ----------------------------------------------------------------

def build_logger(cfg: Dict[str, Any], suffix: str = "") -> ExperimentLogger:
    apply_seed(cfg)
    name = str(cfg["experiment_name"]) + suffix
    meta = ExperimentMeta(
        experiment_id=f"{name}-{int(cfg['seed'])}",
        experiment_name=name,
        paper=str(cfg["paper"]),
        architecture=str(cfg["model"]["name"]),
        dataset=str(cfg["dataset"]["name"]),
        seed=int(cfg["seed"]),
        deterministic=bool(cfg["deterministic"]),
        fault_config=dict(cfg.get("faults") or {}),
        tap_config=dict(cfg.get("taps") or {}),
        assumptions={k: str(v)
                     for k, v in cfg["model"].get("assumptions", {}).items()},
        environment=capture_environment(),
        resolved_config=cfg,
        started_at=_dt.datetime.now().isoformat(timespec="seconds"))
    return ExperimentLogger(cfg["results_dir"], name, meta,
                            log_predictions=bool(cfg.get("log_predictions",
                                                         False)),
                            logger_names=("v2xvitbench",))
