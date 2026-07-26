"""
common.py
---------
Config -> objects. Every builder the three CLI entry points share.

Design rule
    Nothing in this file makes a decision. It reads resolved config and
    assembles the modules that implement the paper. If a value is not in the
    config, it is not configurable -- the brief required that no run need a
    source edit, so anything a sweep might vary lives in ``w2cbench/configs``.

    The converse matters just as much: nothing *outside* this file reads
    config. Every model stage takes typed arguments, which is what lets a test
    substitute a stub for any of them and what keeps a config key from being
    silently reinterpreted three modules deep.

Eager validation
    Tap locations, the encoder/grid stride agreement and the selector's
    required arguments are all checked at load time. A cluster job that dies in
    the first second is cheap; one that runs for six hours and writes an empty
    ``taps.csv`` is not.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from cpbench.data import (AnchorGenerator, BoxDecoder, GridSpec,
                          SyntheticCooperativeDataset, TargetAssigner)
from cpbench.logbook import (ExperimentLogger, ExperimentMeta,
                             capture_environment, seed_everything)
from cpbench.observation import DriftTap, StatsTap, TapSet, TensorDumpTap
from cpbench.utils import load_config
from cpbench.utils.paths import missing_root_message

from ..comm import (CommunicationGraph, GaussianSmoother, make_selector)
from ..data.collate import camera_collator, lidar_collator
from ..data.lidar import W2CLidarDataset
from ..evaluation.tester import DetectionTester
from ..faults.registry import build_bridge, build_protocol_bridge
from ..fusion import SpatialTransform, make_aggregator
from ..models import (LidarPillarEncoder, SpatialConfidenceGenerator,
                      Where2comm)
from ..observation import validate_location
from ..training.losses import MultiRoundDetectionLoss
from ..training.trainer import TrainerConfig

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
    data_track = str(cfg["dataset"].get("track", "lidar"))
    model_track = str(cfg["model"].get("track", data_track))
    if data_track != model_track:
        raise ValueError(
            f"dataset track {data_track!r} and model track {model_track!r} "
            "disagree; a lidar dataset needs a lidar model")

    for group in ("stats", "dump"):
        for name in ((cfg.get("taps") or {}).get(group) or {}).get("include") or []:
            if "*" not in name:
                validate_location(name)

    comm = cfg["model"]["communication"]
    kind = str(comm["selector"])
    if kind == "topk" and comm.get("topk") is None:
        raise ValueError(
            "model.communication.selector=topk needs model.communication.topk")
    if kind == "budget" and comm.get("budget_bytes") is None:
        raise ValueError("model.communication.selector=budget needs "
                         "model.communication.budget_bytes")

    if int(comm.get("rounds", 1)) > 1 and kind == "threshold":
        # Not an error -- a warning, because whether it bites depends on the
        # checkpoint rather than the config.
        #
        # Round k>0 selects on C_i (x) R_j, and R_j <= 1, so the round-k
        # priority is ALWAYS <= the round-0 priority. Against a fixed
        # threshold a later round can therefore only ever select a SUBSET of
        # what round 0 sent -- and when confidence sits near the bar, that
        # subset is empty and multi-round self-extinguishes after round 0.
        #
        # A well-trained model is fine: C_i ~ 0.9 against R_j ~ 0.9 gives 0.81,
        # far above 0.01. An undertrained one sits at the focal prior 0.010051
        # against a threshold of 0.01, and 0.010051 * 0.99 = 0.00995 falls
        # below it. The rounds then run, cost their compute, and transmit
        # nothing but self-links.
        logger.warning(
            "rounds=%d with selector=threshold: round k>0 selects on "
            "C_i (x) R_j <= C_i, so it can only ever select a subset of round "
            "0 -- and with confidence near the threshold (%.4g), nothing at "
            "all. Multi-round is well posed under selector=topk or "
            "selector=budget, where k cells are kept regardless of magnitude. "
            "Check comm/r1/selected_count before trusting a multi-round "
            "result.", int(comm["rounds"]), float(comm.get("threshold", 0.01)))


def track(cfg: Dict[str, Any]) -> str:
    """``"lidar"`` or ``"camera"``, cross-checked between dataset and model.

    The one branch in the package. It lives here so the CLIs stay
    track-agnostic and so a camera dataset paired with a lidar model fails by
    name rather than as an opaque shape error inside the encoder.
    """
    return str(cfg["dataset"].get("track", "lidar"))


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    choice = str(cfg.get("device", "auto"))
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def apply_seed(cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))


# -- geometry ---------------------------------------------------------------

def build_grid_spec(cfg: Dict[str, Any]) -> GridSpec:
    grid = cfg["dataset"]["grid"]
    return GridSpec(voxel_size=tuple(grid["voxel_size"]),
                    point_range=tuple(grid["point_range"]),
                    downsample=int(grid.get("downsample", 2)))


# -- datasets ---------------------------------------------------------------

_DEFAULT_SPLIT_DIRS = {"train": "train", "val": "validate", "test": "test"}


def build_adapters(cfg: Dict[str, Any],
                   split: Optional[str] = None) -> List[Any]:
    """The ``src.datasets`` adapters for one split, one per scenario.

    Returns a LIST because OPV2V-family datasets ship many independent
    scenario folders, and each becomes its own dataset so that a latency fault
    re-reads earlier frames of the *same* scenario rather than across a
    scenario boundary -- which a single globally-indexed adapter could not
    guarantee.
    """
    dataset = cfg["dataset"]
    name = str(dataset["adapter"])
    if name == "synthetic_camera":
        seed = int(cfg["seed"]) + {"train": 0, "val": 1, "test": 2}.get(
            split or "test", 0)
        from cpbench.data import SyntheticCameraCooperativeDataset
        return [SyntheticCameraCooperativeDataset(
            n_frames=int(dataset.get("n_frames", 16)),
            n_agents=int(dataset.get("n_agents", 3)),
            n_objects=int(dataset.get("n_objects", 4)),
            n_cameras=int(dataset.get("n_cameras", 4)),
            image_size=tuple(dataset.get("image_size", (64, 64))),
            seed=seed)]

    if name == "synthetic":
        # The split only varies the seed, so train / val / test are different
        # scenes: an identical val set gives a meaningless validation signal.
        seed = int(cfg["seed"]) + {"train": 0, "val": 1, "test": 2}.get(
            split or "test", 0)
        return [SyntheticCooperativeDataset(
            n_frames=int(dataset.get("n_frames", 16)),
            n_agents=int(dataset.get("n_agents", 3)),
            n_objects=int(dataset.get("n_objects", 4)), seed=seed)]

    if name in ("opv2v", "v2xset", "dair_v2x"):
        from src.datasets import load_dataset
        root = Path(dataset["root"])
        mapping = dataset.get("split") or _DEFAULT_SPLIT_DIRS
        split_dir = root / mapping.get(split, split) if split else root
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
        return [load_dataset(name, scenario_dir=str(p)) for p in scenarios]

    raise ValueError(
        f"unknown dataset adapter {name!r}; expected synthetic, "
        "synthetic_camera, opv2v, v2xset or dair_v2x")


def build_bridge_for(cfg: Dict[str, Any],
                     overrides: Optional[Dict[str, Any]] = None):
    """The physical bridge for a run. ``overrides={}`` forces a clean bridge.

    The empty-dict override is how a benchmark's clean reference is built:
    ``{}`` is explicit "no faults", distinct from ``None`` which means "fall
    back to the config". Conflating them is how a clean run silently inherits
    a sweep's faults.
    """
    fault_cfg = overrides if overrides is not None else cfg.get("faults")
    return build_bridge(fault_cfg or None,
                        fps=float(cfg["dataset"].get("fps", 10.0)),
                        seed=int(cfg["seed"]))


def build_protocol_for(cfg: Dict[str, Any],
                       overrides: Optional[Dict[str, Any]] = None):
    """The protocol bridge for a run, from the same config block."""
    fault_cfg = overrides if overrides is not None else cfg.get("faults")
    return build_protocol_bridge(fault_cfg or None, seed=int(cfg["seed"]))


def build_dataset(cfg: Dict[str, Any], bridge=None,
                  split: Optional[str] = None):
    """The dataset for one split, wrapping ``bridge`` (clean if None)."""
    dataset_cfg = cfg["dataset"]
    grid = build_grid_spec(cfg)
    if track(cfg) == "camera":
        from ..data.camera import W2CCameraDataset
        sets = [W2CCameraDataset(
            adapter, grid, max_cav=int(dataset_cfg.get("max_cav", 5)),
            bridge=bridge, camera_names=dataset_cfg.get("camera_names"),
            categories=dataset_cfg.get("categories"))
            for adapter in build_adapters(cfg, split)]
        if len(sets) == 1:
            return sets[0]
        from torch.utils.data import ConcatDataset
        return ConcatDataset(sets)
    sets = [W2CLidarDataset(
        adapter, grid, max_cav=int(dataset_cfg.get("max_cav", 5)),
        bridge=bridge, categories=dataset_cfg.get("categories"),
        max_points_per_pillar=int(dataset_cfg.get("max_points_per_pillar", 32)),
        max_pillars=int(dataset_cfg.get("max_pillars", 20000)))
        for adapter in build_adapters(cfg, split)]
    if len(sets) == 1:
        return sets[0]
    from torch.utils.data import ConcatDataset
    return ConcatDataset(sets)


def build_collator(cfg: Dict[str, Any]):
    max_cav = int(cfg["dataset"].get("max_cav", 5))
    return (camera_collator(max_cav) if track(cfg) == "camera"
            else lidar_collator(max_cav))


# -- model ------------------------------------------------------------------

def build_encoder(cfg: Dict[str, Any]):
    """The track's Stage-1 encoder -- the only modality-specific component."""
    enc = cfg["model"]["encoder"]
    if track(cfg) == "camera":
        from ..models.encoder_camera import CameraEncoder
        backbone = enc.get("backbone") or {}
        lift = enc.get("lifting") or {}
        return CameraEncoder(
            build_grid_spec(cfg), out_channels=int(enc["out_channels"]),
            backbone_arch=str(backbone.get("arch", "resnet34")),
            pretrained=bool(backbone.get("pretrained", True)),
            id_pick=list(backbone.get("id_pick", (1, 2, 3))),
            lift_level=int(lift.get("level", 0)),
            depth_bins=tuple(lift.get("depth_bins", (4.0, 45.0, 1.0))),
            image_size=tuple(cfg["dataset"].get("image_size", (160, 416))))
    return LidarPillarEncoder(
        build_grid_spec(cfg),
        vfe_channels=int(enc.get("vfe_channels", 64)),
        block_channels=list(enc.get("block_channels", (64, 128, 256))),
        block_strides=list(enc.get("block_strides", (2, 2, 2))),
        block_layers=list(enc.get("block_layers", (3, 5, 5))),
        upsample_channels=int(enc.get("upsample_channels", 128)),
        out_channels=int(enc["out_channels"]))


def build_model(cfg: Dict[str, Any]) -> Where2comm:
    """Assemble Where2comm from the model config group."""
    model_cfg = cfg["model"]
    comm = model_cfg["communication"]
    fusion = model_cfg["fusion"]
    head = model_cfg["head"]

    encoder = build_encoder(cfg)
    channels = encoder.out_channels

    smooth = model_cfg["confidence"].get("gaussian_smooth") or {}
    smoother = (GaussianSmoother(
        k_size=int(smooth.get("k_size", 3)),
        sigma=float(smooth.get("c_sigma", 1.0)),
        normalize=bool(smooth.get("normalize", True)),
        padding_mode=str(smooth.get("padding_mode", "zeros")))
        if smooth.get("enabled", True) else None)

    selector = make_selector(
        str(comm["selector"]), channels=channels,
        threshold=float(comm.get("threshold", 0.01)),
        k=comm.get("topk"), budget_bytes=comm.get("budget_bytes"),
        bytes_per_element=int(comm.get("bytes_per_element", 4)),
        self_mask=str(comm.get("self_mask", "ones")),
        train_bandwidth=tuple(comm.get("train_bandwidth", (0.1, 1.0))))

    return Where2comm(
        encoder=encoder,
        confidence=SpatialConfidenceGenerator(
            in_channels=channels, num_anchors=int(head["num_anchors"]),
            num_classes=int(head["num_classes"]), smoother=smoother),
        selector=selector,
        aggregator=make_aggregator(
            str(fusion["aggregator"]), dim=channels,
            heads=int(fusion.get("heads", 8)),
            with_spe=bool(fusion.get("with_spe", False)),
            with_scm=bool(fusion.get("with_scm", True)),
            mlp_dim=fusion.get("mlp_dim"),
            dropout=float(fusion.get("dropout", 0.0))),
        warp=SpatialTransform.from_grid_spec(build_grid_spec(cfg)),
        graph=CommunicationGraph(
            broadcast_round_zero=bool(comm.get("broadcast_round_zero", True))),
        rounds=int(comm.get("rounds", 1)),
        use_request_map=bool(comm.get("use_request_map", True)))


# -- objective and evaluation -----------------------------------------------

def build_loss(cfg: Dict[str, Any]) -> MultiRoundDetectionLoss:
    loss = cfg["model"]["loss"]
    head = cfg["model"]["head"]
    return MultiRoundDetectionLoss(
        TargetAssigner(AnchorGenerator(build_grid_spec(cfg))),
        alpha=float(loss.get("alpha", 0.25)),
        gamma=float(loss.get("gamma", 2.0)),
        reg_weight=float(loss.get("reg_weight", 2.0)),
        num_classes=int(head["num_classes"]),
        single_weight=float(loss.get("single_weight", 1.0)),
        round_weights=loss.get("round_weights"))


def build_decoder(cfg: Dict[str, Any]) -> BoxDecoder:
    return BoxDecoder(AnchorGenerator(build_grid_spec(cfg)),
                      score_threshold=float(cfg["model"].get(
                          "score_threshold", 0.2)))


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

    The runner is fault-agnostic; this closure is where a condition turns into
    a dataset wrapping that condition's bridges.
    """
    collate = build_collator(cfg)
    decoder = build_decoder(cfg)
    max_frames = cfg.get("max_frames")
    bytes_per_element = int(cfg["model"]["communication"].get(
        "bytes_per_element", 4))

    def make(condition, reference):
        overrides = condition.config or {}
        dataset = build_dataset(cfg, build_bridge_for(cfg, overrides),
                                split="test")
        return DetectionTester(
            dataset, decoder, collate, device=device, reference=reference,
            protocol=build_protocol_for(cfg, overrides),
            bytes_per_element=bytes_per_element, max_frames=max_frames)
    return make


def bandwidth_sweep(cfg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The accuracy-versus-bandwidth axis, or None to evaluate at one point."""
    return cfg.get("bandwidth_sweep") or None


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
                            logger_names=("w2cbench",))
