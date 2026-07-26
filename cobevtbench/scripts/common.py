"""
common.py
---------
Config -> objects. Every builder the three CLI entry points share.

Design rule
    Nothing in this file makes a decision. It reads resolved config and
    assembles the modules that implement the paper. If a value is not in the
    config, it is not configurable -- the brief required that no run need a
    source edit, so anything a sweep might vary lives in
    ``cobevtbench/configs``.

Track dispatch
    Almost every builder branches on ``dataset.track`` (``camera`` or
    ``lidar``). That branch is here and only here, so the CLIs stay
    track-agnostic: ``build_model``, ``build_dataset``, ``build_loss`` and
    ``build_tester_factory`` each return the right object for the configured
    track, and the entry points never mention a track by name.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from cpbench.data import (AnchorGenerator, BEVGrid, BoxDecoder, GridSpec,
                          SyntheticCameraCooperativeDataset,
                          SyntheticCooperativeDataset)
from cpbench.logbook import (ExperimentLogger, ExperimentMeta,
                             capture_environment, seed_everything)
from cpbench.observation import (DriftTap, StatsTap, TapSet, TensorDumpTap)
from cpbench.utils import load_config
from cpbench.utils.paths import missing_root_message

from ..data.camera import CoBEVTCameraDataset
from ..data.collate import camera_collator, lidar_collator
from ..data.lidar import CoBEVTLidarDataset
from ..evaluation.tester import DetectionTester, SegmentationTester
from ..faults.registry import build_bridge
from ..models.cobevt_camera import CoBEVTCamera
from ..models.cobevt_lidar import CoBEVTLidar
from ..training.losses import DetectionLoss, VanillaSegLoss

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


# -- config -----------------------------------------------------------------

def load(overrides: Optional[List[str]] = None,
         config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Compose the resolved config from groups and CLI overrides."""
    return load_config(config_path or CONFIG_ROOT, overrides)


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    choice = str(cfg.get("device", "auto"))
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def apply_seed(cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))


def track(cfg: Dict[str, Any]) -> str:
    """The configured track, cross-checked between dataset and model.

    A camera dataset with a lidar model composes without error and then fails
    deep in a forward pass with an opaque shape mismatch. Catching the
    mismatch here names both sides.
    """
    dataset_track = str(cfg["dataset"].get("track", "camera"))
    model_track = str(cfg["model"].get("track", dataset_track))
    if dataset_track != model_track:
        raise ValueError(
            f"dataset track {dataset_track!r} and model track {model_track!r} "
            "disagree; a camera dataset needs a camera model and a lidar "
            "dataset a lidar model")
    return dataset_track


# -- geometry ---------------------------------------------------------------

def build_grid_spec(cfg: Dict[str, Any]) -> GridSpec:
    grid = cfg["dataset"]["grid"]
    return GridSpec(voxel_size=tuple(grid["voxel_size"]),
                    point_range=tuple(grid["point_range"]),
                    downsample=int(grid.get("downsample", 2)))


def build_bev_grid(cfg: Dict[str, Any]) -> BEVGrid:
    bev = cfg["dataset"]["bev"]
    return BEVGrid(height=int(bev["height"]), width=int(bev["width"]),
                   h_meters=float(bev["h_meters"]),
                   w_meters=float(bev["w_meters"]),
                   offset=float(bev.get("offset", 0.0)))


# -- datasets ---------------------------------------------------------------

# On-disk split-name for each logical split, from dataset.split; falls back
# to the OPV2V/V2XSet layout (train / validate / test).
_DEFAULT_SPLIT_DIRS = {"train": "train", "val": "validate", "test": "test"}


def _split_dir_name(cfg: Dict[str, Any], split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    mapping = cfg["dataset"].get("split") or _DEFAULT_SPLIT_DIRS
    return mapping.get(split, split)


def build_adapters(cfg: Dict[str, Any],
                   split: Optional[str] = None) -> List[Any]:
    """The ``src.datasets`` adapters for one split, one per scenario.

    Returns a LIST, matching the corabench convention: OPV2V/V2XSet ship many
    independent scenario folders, and each becomes its own adapter so that
    ``build_dataset`` can wrap it in its own dataset and ConcatDataset them.
    That keeps a latency fault re-reading earlier frames of the *same*
    scenario rather than across a scenario boundary -- which a single
    globally-indexed adapter could not guarantee.

    Synthetic datasets return a one-element list; the split only varies the
    seed, so train / val / test are different scenes (an identical val set
    would give a meaningless validation signal).
    """
    dataset = cfg["dataset"]
    name = str(dataset["adapter"])
    if name in ("synthetic_camera", "synthetic"):
        seed = int(cfg["seed"]) + {"train": 0, "val": 1, "test": 2}.get(
            split or "test", 0)
        common = dict(n_frames=int(dataset.get("n_frames", 16)),
                      n_agents=int(dataset.get("n_agents", 3)),
                      n_objects=int(dataset.get("n_objects", 4)), seed=seed)
        if name == "synthetic_camera":
            return [SyntheticCameraCooperativeDataset(
                n_cameras=int(dataset.get("n_cameras", 4)),
                image_size=tuple(dataset.get("image_size", (64, 64))),
                fov_degrees=float(dataset.get("fov_degrees", 90.0)), **common)]
        return [SyntheticCooperativeDataset(**common)]

    if name in ("opv2v", "v2xset"):
        from pathlib import Path as _Path

        from src.datasets import load_dataset
        root = _Path(dataset["root"])
        split_name = _split_dir_name(cfg, split)
        split_dir = root / split_name if split_name else root
        if not split_dir.is_dir():
            raise FileNotFoundError(
                missing_root_message(split_dir,
                                     config_value=cfg.get("data_root"),
                                     what=f"{dataset['name']} split directory")
                + f"\n(split map: {dataset.get('split')})")
        wanted = dataset.get("scenarios")
        scenarios = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if wanted:
            scenarios = [p for p in scenarios if p.name in set(wanted)]
        if not scenarios:
            raise FileNotFoundError(f"no scenario directories under {split_dir}")
        max_cams = int(dataset.get("n_cameras", 4))
        return [load_dataset(name, scenario_dir=str(p), max_cams=max_cams)
                for p in scenarios]

    raise ValueError(
        f"unknown dataset adapter {name!r}; expected synthetic_camera, "
        "synthetic, opv2v or v2xset")


def build_bridge_for(cfg: Dict[str, Any],
                     overrides: Optional[Dict[str, Any]] = None):
    """A DataFaultBridge for the run. ``overrides={}`` forces a clean bridge.

    The empty-dict override is how the benchmark's clean reference is built:
    passing ``{}`` is explicit "no faults", distinct from ``None`` which means
    "fall back to the config". Conflating them is how a clean run silently
    inherits a sweep's faults.
    """
    fault_cfg = overrides if overrides is not None else cfg.get("faults")
    fps = float(cfg["dataset"].get("fps", 10.0))
    return build_bridge(fault_cfg or None, fps=fps, seed=int(cfg["seed"]))


def build_dataset(cfg: Dict[str, Any], bridge=None,
                  split: Optional[str] = None):
    """The track's dataset for one split, wrapping ``bridge`` (clean if None).

    Each scenario adapter is wrapped in its own dataset sharing ``bridge``,
    then concatenated. A single scenario returns the dataset directly; many
    return a ``ConcatDataset`` that the DataLoader indexes transparently.
    """
    adapters = build_adapters(cfg, split)
    max_cav = int(cfg["dataset"].get("max_cav", 5))
    categories = cfg["dataset"].get("categories")
    is_camera = track(cfg) == "camera"

    def wrap(adapter):
        if is_camera:
            return CoBEVTCameraDataset(
                adapter, build_bev_grid(cfg), max_cav=max_cav, bridge=bridge,
                target=str(cfg["model"].get("target", "dynamic")),
                categories=categories)
        return CoBEVTLidarDataset(
            adapter, build_grid_spec(cfg), max_cav=max_cav, bridge=bridge,
            categories=categories,
            max_points_per_pillar=int(cfg["dataset"].get("max_points_per_pillar", 32)),
            max_pillars=int(cfg["dataset"].get("max_pillars", 20000)))

    sets = [wrap(adapter) for adapter in adapters]
    if len(sets) == 1:
        return sets[0]
    from torch.utils.data import ConcatDataset
    return ConcatDataset(sets)


def build_collator(cfg: Dict[str, Any]):
    max_cav = int(cfg["dataset"].get("max_cav", 5))
    return (camera_collator(max_cav) if track(cfg) == "camera"
            else lidar_collator(max_cav))


# -- models -----------------------------------------------------------------

def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """The track's model, wired from the model config group."""
    if track(cfg) == "camera":
        return _build_camera_model(cfg)
    return _build_lidar_model(cfg)


def _build_camera_model(cfg: Dict[str, Any]) -> CoBEVTCamera:
    m = cfg["model"]
    sinbevt = m["sinbevt"]
    fuse = m["fusebevt"]
    dataset = cfg["dataset"]
    return CoBEVTCamera(
        target=str(m.get("target", "dynamic")),
        max_cav=int(dataset.get("max_cav", 5)),
        image_size=tuple(dataset.get("image_size", (512, 512))),
        bev_meters=float(sinbevt["bev_meters"]),
        bev_size=int(sinbevt["bev_size"]),
        dims=list(sinbevt["dims"]),
        q_win_sizes=list(sinbevt["q_win_sizes"]),
        feat_win_sizes=list(sinbevt["feat_win_sizes"]),
        heads=list(sinbevt["heads"]),
        dim_head=list(sinbevt["dim_head"]),
        middle=list(sinbevt["middle"]),
        bev_embedding_flags=list(sinbevt["bev_embedding_flags"]),
        self_attn_dim_head=int(sinbevt["self_attn_dim_head"]),
        camera_reduce=str(sinbevt.get("camera_reduce", "mean")),
        no_image_features=bool(sinbevt.get("no_image_features", False)),
        backbone_arch=str(m["backbone"]["arch"]),
        pretrained=bool(m["backbone"]["pretrained"]),
        id_pick=list(m["backbone"]["id_pick"]),
        compression=int(m.get("compression", 0)),
        fuse_depth=int(fuse["depth"]), fuse_window=int(fuse["window"]),
        fuse_dim_head=int(fuse["dim_head"]), fuse_mlp_dim=fuse.get("mlp_dim"),
        fuse_dropout=float(fuse.get("dropout", 0.0)),
        use_local=bool(fuse.get("use_local", True)),
        use_global=bool(fuse.get("use_global", True)),
        pool=str(fuse.get("pool", "mean")),
        decoder_channels=list(m["decoder"]["channels"]),
        upsample_mode=str(m["decoder"].get("upsample_mode", "nearest")),
        head_kernel_size=int(m["head"].get("kernel_size", 3)))


def _build_lidar_model(cfg: Dict[str, Any]) -> CoBEVTLidar:
    m = cfg["model"]
    fuse = m["fusebevt"]
    return CoBEVTLidar(
        build_grid_spec(cfg), max_cav=int(cfg["dataset"].get("max_cav", 5)),
        encoder_out_channels=int(m["encoder"]["out_channels"]),
        block_strides=tuple(m["encoder"].get("block_strides", (2, 2, 2))),
        fuse_dim=fuse.get("dim"), fuse_depth=int(fuse["depth"]),
        fuse_window=int(fuse["window"]), fuse_dim_head=int(fuse["dim_head"]),
        fuse_mlp_dim=fuse.get("mlp_dim"),
        fuse_dropout=float(fuse.get("dropout", 0.0)),
        use_local=bool(fuse.get("use_local", True)),
        use_global=bool(fuse.get("use_global", True)),
        pool=str(fuse.get("pool", "mean")),
        num_anchors=int(m["head"]["num_anchors"]),
        num_classes=int(m["head"]["num_classes"]))


# -- losses -----------------------------------------------------------------

def build_loss(cfg: Dict[str, Any]) -> Callable:
    """A ``(batch, output) -> loss dict`` closure for the Trainer."""
    m = cfg["model"]
    if track(cfg) == "camera":
        loss = m["loss"]
        criterion = VanillaSegLoss(
            target=str(m.get("target", "dynamic")),
            d_weights=float(loss.get("d_weights", 75.0)),
            s_weights=float(loss.get("s_weights", 2.0)),
            l_weights=float(loss.get("l_weights", 4.0)),
            coefficient=loss.get("coefficient"))

        def camera_loss(batch, output):
            return criterion(output["logits"], batch["target"])
        return camera_loss

    loss = m["loss"]
    criterion = DetectionLoss(
        alpha=float(loss.get("alpha", 0.25)),
        gamma=float(loss.get("gamma", 2.0)),
        reg_weight=float(loss.get("reg_weight", 2.0)),
        num_classes=int(m["head"]["num_classes"]))
    assigner = _build_target_assigner(cfg)

    def lidar_loss(batch, output):
        import numpy as np
        cls_targets, reg_targets = [], []
        for boxes in batch["gt_boxes"]:
            assigned = assigner(boxes if boxes is not None
                                else np.zeros((0, 7), dtype=np.float32))
            cls_targets.append(assigned["cls_target"])
            reg_targets.append(assigned["reg_target"])
        return criterion(output["cls"], output["reg"],
                         torch.stack(cls_targets), torch.stack(reg_targets))
    return lidar_loss


def _build_target_assigner(cfg: Dict[str, Any]):
    from cpbench.data import TargetAssigner
    return TargetAssigner(AnchorGenerator(build_grid_spec(cfg)))


# -- evaluation -------------------------------------------------------------

def build_decoder(cfg: Dict[str, Any]) -> BoxDecoder:
    return BoxDecoder(AnchorGenerator(build_grid_spec(cfg)),
                      score_threshold=float(cfg["model"].get("score_threshold",
                                                             0.2)))


def build_tester_factory(cfg: Dict[str, Any], device: torch.device,
                         taps_factory: Optional[Callable] = None):
    """A ``(condition, reference) -> tester`` factory for the benchmark runner.

    The runner is track-agnostic; this closure is where the track choice
    turns into a SegmentationTester or a DetectionTester, each wrapping a
    dataset built around the condition's bridge.
    """
    is_camera = track(cfg) == "camera"
    collate = build_collator(cfg)
    max_frames = cfg.get("max_frames")
    class_names = _class_names(cfg)
    decoder = None if is_camera else build_decoder(cfg)

    def make(condition, reference):
        bridge = build_bridge_for(cfg, overrides=condition.config or {})
        dataset = build_dataset(cfg, bridge, split="test")
        if is_camera:
            return SegmentationTester(dataset, class_names, collate,
                                      device=device, reference=reference,
                                      max_frames=max_frames)
        return DetectionTester(dataset, decoder, collate, device=device,
                               reference=reference, max_frames=max_frames)
    return make


def _class_names(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    from ..data.camera import DYNAMIC_CLASSES, STATIC_CLASSES
    return (STATIC_CLASSES if str(cfg["model"].get("target")) == "static"
            else DYNAMIC_CLASSES)


def metric_kind(cfg: Dict[str, Any]) -> str:
    return "segmentation" if track(cfg) == "camera" else "detection"


# -- taps -------------------------------------------------------------------

def build_taps(cfg: Dict[str, Any], out_dir: Path
               ) -> Tuple[Optional[TapSet], Optional[StatsTap]]:
    """(TapSet or None, StatsTap or None) from the taps config group.

    Every location a dump/stats config names is validated against the
    registry first, so a typo fails in the first second rather than after a
    job finishes with an empty taps.csv.
    """
    from ..observation import validate_location

    t = cfg.get("taps") or {}
    for group in ("stats", "dump"):
        include = (t.get(group) or {}).get("include")
        for name in include or []:
            validate_location(name)

    children, stats_tap = [], None
    if t.get("stats"):
        stats_tap = StatsTap()
        children.append(TapSet([stats_tap], include=t["stats"].get("include")))
    if t.get("dump"):
        d = t["dump"]
        dump = TensorDumpTap(out_dir / "taps", every_n=int(d.get("every_n", 1)),
                             max_dumps=d.get("max_dumps"))
        children.append(TapSet([dump], include=d.get("include")))
    if t.get("drift"):
        children.append(TapSet([DriftTap(t["drift"])]))
    if not children:
        return None, None
    return TapSet(children), stats_tap


# -- optimizer / scheduler --------------------------------------------------

def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    t = cfg["trainer"]
    name = str(t.get("optimizer", "adamw")).lower()
    params = dict(lr=float(t["lr"]),
                  weight_decay=float(t.get("weight_decay", 0.0)))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), eps=float(t.get("eps", 1e-8)),
                                 **params)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), eps=float(t.get("eps", 1e-8)),
                                **params)
    raise ValueError(f"unknown optimizer {name!r}; expected 'adamw' or 'adam'")


def build_scheduler(cfg: Dict[str, Any], optimizer):
    t = cfg["trainer"]
    name = str(t.get("scheduler", "none")).lower()
    if name == "none":
        return None
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(t.get("lr_steps", [])),
            gamma=float(t.get("lr_gamma", 0.1)))
    if name in ("cosine", "cosine_warm"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(t["epochs"]), eta_min=float(t.get("lr_min", 0.0)))
    raise ValueError(f"unknown scheduler {name!r}")


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
        assumptions={k: str(v) for k, v in cfg["model"].get("assumptions", {}).items()},
        environment=capture_environment(),
        resolved_config=cfg,
        started_at=_dt.datetime.now().isoformat(timespec="seconds"))
    return ExperimentLogger(cfg["results_dir"], name, meta,
                            log_predictions=bool(cfg.get("log_predictions", False)),
                            logger_names=("cobevtbench",))
