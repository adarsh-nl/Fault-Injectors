"""
common.py
---------
Config -> objects. Every builder the CLI entry points share.

Design rule
    Nothing in this file makes a decision. It reads resolved config and
    assembles the modules that implement the paper. If a value is not in the
    config, it is not configurable -- and the requirement was that nothing
    should need a source edit, so anything a sweep might want to vary lives
    in ``lgcpbench/configs``.

Model registry
    ``build_backbone`` dispatches on ``model.backend``. The native backend is
    always available; the OpenCOOD backend is imported lazily and fails with
    an actionable message rather than at import time, because it needs a
    separate Python 3.7 environment that the core deliberately does not
    depend on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from cpbench.data.preprocessing import AnchorGenerator, GridSpec
from cpbench.faults.bridge import DataFaultBridge
from cpbench.logbook.env import capture_environment, seed_everything
from cpbench.logbook.experiment import ExperimentLogger
from cpbench.logbook.schema import ExperimentMeta
from cpbench.observation.recorders import StatsTap
from cpbench.observation.taps import TapSet
from cpbench.utils.config import load_config

from ..confidence import AreaConfidenceEstimator
from ..data import LGCPDataset
from ..faults import ControlPlaneFaultBridge
from ..metrics import LGCPEvaluator
from ..network import (
    FusionLatencyModel,
    InterferenceModel,
    LatencyModel,
    PathLossModel,
    RateModel,
    ShadowingModel,
    TransmissionScheduler,
)
from ..observation import ControlPlaneTap
from ..orchestration import GlobalViewAggregator, LGCPPipeline, RSUController
from ..perception import AreaFeatureMasker, NativeReferenceBackbone
from ..perception.decode import AreaBoxDecoder
from ..roi import AreaGrid, make_occupancy_estimator
from ..selection import (
    FirstMemberLeaderElector,
    GreedyGroupSelector,
    MinMaxLoadLeaderElector,
    SelectionAlgorithm,
)

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load(overrides: Optional[List[str]] = None,
         config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and compose the config tree."""
    return load_config(config_path or CONFIG_ROOT, overrides)


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    """``auto`` picks CUDA when present."""
    want = str(cfg.get("device", "auto"))
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(want)


# --------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------- #


def build_grid_spec(cfg: Dict[str, Any]) -> GridSpec:
    """BEV geometry, shared by the backbone, the areas and the anchors.

    Validates the divisibility constraint here rather than letting it surface
    as an opaque shape error inside the backbone twenty minutes into a job.
    """
    g = cfg["dataset"]["grid"]
    spec = GridSpec(
        voxel_size=tuple(g["voxel_size"]),
        point_range=tuple(g["point_range"]),
        downsample=int(g["downsample"]),
    )
    total_stride = int(g["downsample"]) * 4
    h0, w0 = spec.grid_hw
    if h0 % total_stride or w0 % total_stride:
        raise ValueError(
            f"dataset.grid gives a {h0} x {w0} canvas, which must divide by "
            f"{total_stride} (downsample x the 3-level pyramid). Adjust "
            f"point_range so each extent is a multiple of "
            f"{g['voxel_size'][0] * total_stride} m."
        )
    return spec


def build_area_grid(cfg: Dict[str, Any], spec: GridSpec) -> AreaGrid:
    return AreaGrid.from_grid_spec(spec, area_size_m=cfg["lgcp"]["roi"]["area_size_m"])


# --------------------------------------------------------------------- #
# perception
# --------------------------------------------------------------------- #


def build_backbone(cfg: Dict[str, Any], spec: GridSpec, device: torch.device):
    """Dispatch on ``model.backend``."""
    m = cfg["model"]
    backend = str(m.get("backend", "native"))

    if backend == "native":
        model = NativeReferenceBackbone(
            grid_hw=spec.grid_hw,
            feature_hw=spec.feature_hw,
            channels=int(m["channels"]),
            num_anchors=int(m["num_anchors"]),
            num_classes=int(m["num_classes"]),
            downsample=int(m["downsample"]),
            use_projections=bool(m.get("use_projections", True)),
        )
        return model.to(device).eval()

    if backend == "opencood":
        try:
            from ..perception.opencood.adapter import OpenCOODBackbone
        except ImportError as exc:
            raise ImportError(
                "the OpenCOOD backend needs the separate Python 3.7 environment "
                "(numba==0.49.0, spconv, CUDA). Build it with "
                "slurm/opencood_env.sbatch, or run with model=native for the "
                "CPU reference backbone. Original error: " + str(exc)
            ) from exc
        return OpenCOODBackbone.from_config(m, spec, device)

    raise KeyError(f"unknown model.backend {backend!r}; expected 'native' or 'opencood'")


def build_decoder(cfg: Dict[str, Any], spec: GridSpec, grid: AreaGrid) -> AreaBoxDecoder:
    m = cfg["model"]
    return AreaBoxDecoder(
        AnchorGenerator(spec),
        grid,
        spec.feature_hw,
        score_threshold=float(m.get("score_threshold", 0.2)),
        nms_iou=m.get("nms_iou", 0.15),
    )


# --------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------- #


def build_scheduler(cfg: Dict[str, Any], seed: int) -> TransmissionScheduler:
    n = cfg["network"]
    rate = RateModel(
        tx_power_dbm=float(n["tx_power_dbm"]),
        noise_power_dbm=float(n["noise_power_dbm"]),
        subchannel_bandwidth_hz=float(n["subchannel_bandwidth_hz"]),
        rate_threshold_bps=float(n["rate_threshold_bps"]),
        fixed_rate_bps=float(n["fixed_rate_bps"]),
        path_loss=PathLossModel(
            intercept_db=float(n["path_loss_intercept_db"]),
            slope_db=float(n["path_loss_slope_db"]),
        ),
        shadowing=ShadowingModel(
            std_db=float(n["shadowing_std_db"]),
            seed=seed,
            enabled=bool(n.get("shadowing_enabled", True)),
        ),
    )
    interference = InterferenceModel(
        positions={},                       # refreshed per frame
        rate_model=rate,
        interference_range_m=n.get("interference_range_m"),
    )
    return TransmissionScheduler(
        interference=interference,
        fusion_model=FusionLatencyModel(
            mflops_per_member=float(cfg["model"]["mflops"]),
            capacity_tflops=float(cfg["lgcp"]["latency"]["cav_capacity_tflops"]),
        ),
        n_subchannels=int(n["subchannels_Z"]),
        time_slot_s=float(n["time_slot_s"]),
    )


def build_latency_model(cfg: Dict[str, Any]) -> LatencyModel:
    lat = cfg["lgcp"]["latency"]
    return LatencyModel(
        rate_bps=float(cfg["network"]["fixed_rate_bps"]),
        n_subchannels=int(cfg["network"]["subchannels_Z"]),
        message_bits={k: float(v) for k, v in lat["message_bits"].items()},
        deadline_s=float(lat["deadline_T_ms"]) / 1e3,
    )


# --------------------------------------------------------------------- #
# control plane
# --------------------------------------------------------------------- #


def build_rsu(cfg: Dict[str, Any], spec: GridSpec, grid: AreaGrid,
              seed: int) -> RSUController:
    l = cfg["lgcp"]
    electors = {
        "min_max_load": MinMaxLoadLeaderElector,
        "first_member": FirstMemberLeaderElector,
    }
    policy = str(l["selection"]["leader_policy"])
    if policy not in electors:
        raise KeyError(
            f"unknown leader_policy {policy!r}; expected one of {sorted(electors)}"
        )

    return RSUController(
        grid=grid,
        occupancy=make_occupancy_estimator(
            l["roi"]["occupancy_source"],
            **({"dilate_rings": int(l["roi"]["occupancy_dilate_rings"])}
               if l["roi"]["occupancy_source"] != "all" else {}),
        ),
        confidence=AreaConfidenceEstimator(
            grid, spec.feature_hw, pooling=str(l["confidence"]["pooling"])
        ),
        selection=SelectionAlgorithm(
            GreedyGroupSelector(
                delta_g=float(l["confidence"]["delta_g"]),
                max_group_size=l["confidence"]["max_group_size"],
            ),
            electors[policy](),
        ),
        scheduler=build_scheduler(cfg, seed),
        latency=build_latency_model(cfg),
        aggregator=GlobalViewAggregator(
            mode=str(l["rsu"]["aggregation"]),
            nms_iou=float(l["rsu"]["nms_iou"]),
            grid=grid,
        ),
    )


def build_control_bridge(
    cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None
) -> Optional[ControlPlaneFaultBridge]:
    """The control plane's fault surface (plane 3).

    ``overrides`` replaces the injector set for one sweep condition. An
    explicit ``{}`` forces a clean control plane, which is what the reference
    condition needs -- ``None`` means "fall back to config".
    """
    f = dict(cfg["faults"])
    pipeline_cfg = overrides if overrides is not None else f.get("control_pipeline") or {}
    if not pipeline_cfg:
        return None
    return ControlPlaneFaultBridge(
        {"pipeline": pipeline_cfg, "seed": int(f.get("seed", cfg["seed"]))}
    )


def build_pipeline(cfg: Dict[str, Any], device: Optional[torch.device] = None,
                   control_faults: Optional[ControlPlaneFaultBridge] = None
                   ) -> Tuple[LGCPPipeline, GridSpec]:
    """Assemble everything. Returns (pipeline, grid spec)."""
    device = device or resolve_device(cfg)
    seed = int(cfg["seed"])
    spec = build_grid_spec(cfg)
    grid = build_area_grid(cfg, spec)
    pipeline = LGCPPipeline(
        backbone=build_backbone(cfg, spec, device),
        rsu=build_rsu(cfg, spec, grid, seed),
        masker=AreaFeatureMasker(grid, spec.feature_hw),
        decoder=build_decoder(cfg, spec, grid),
        control_faults=control_faults,
    )
    return pipeline, spec


# --------------------------------------------------------------------- #
# data + faults
# --------------------------------------------------------------------- #


def build_adapter(cfg: Dict[str, Any]):
    """Build the underlying ``src.datasets`` adapter."""
    d = cfg["dataset"]
    kind = str(d["adapter"])
    if kind == "synthetic":
        from cpbench.data.synthetic import SyntheticCooperativeDataset

        return SyntheticCooperativeDataset(
            n_frames=int(d["n_frames"]),
            n_agents=int(d["n_agents"]),
            n_objects=int(d["n_objects"]),
            seed=int(cfg["seed"]),
        )

    from src.datasets import load_dataset

    root = d.get("root")
    if not root:
        raise ValueError(f"dataset.root is required for adapter {kind!r}")
    return load_dataset(kind, root)


def build_bridge(cfg: Dict[str, Any],
                 overrides: Optional[Dict[str, Any]] = None) -> Optional[DataFaultBridge]:
    """The corruption plane. ``overrides`` replaces the pipeline for a sweep."""
    f = dict(cfg["faults"])
    pipeline_cfg = overrides if overrides is not None else f.get("pipeline") or {}
    if not pipeline_cfg:
        return None
    return DataFaultBridge(
        {
            "pipeline": pipeline_cfg,
            "agent_scope": f.get("agent_scope", "non-ego"),
        },
        fps=float(cfg["dataset"].get("fps", 10.0)),
        seed=int(f.get("seed", cfg["seed"])),
    )


def build_dataset(cfg: Dict[str, Any], spec: GridSpec, adapter,
                  bridge: Optional[DataFaultBridge] = None) -> LGCPDataset:
    d = cfg["dataset"]
    return LGCPDataset(
        adapter=adapter,
        grid=spec,
        bridge=bridge,
        max_points_per_pillar=int(d["max_points_per_pillar"]),
        max_pillars=int(d["max_pillars"]),
        max_agents=int(d["max_agents"]),
        comm_range_m=float(d["comm_range_m"]),
        categories=d.get("categories"),
    )


# --------------------------------------------------------------------- #
# observation + logging
# --------------------------------------------------------------------- #


def build_taps(cfg: Dict[str, Any]):
    """Returns (TapSet | None, ControlPlaneTap | None, StatsTap | None)."""
    t = cfg["taps"]
    taps, control, stats = [], None, None
    if t.get("control"):
        control = ControlPlaneTap()
        taps.append(control)
    if t.get("stats"):
        stats = StatsTap()
        taps.append(TapSet([stats], include=t["stats"].get("include")))
    if not taps:
        return None, None, None
    return TapSet(taps), control, stats


def build_evaluator(cfg: Dict[str, Any], pipeline: LGCPPipeline) -> LGCPEvaluator:
    return LGCPEvaluator(
        pipeline,
        score_threshold=float(cfg["model"].get("score_threshold", 0.2)),
        keep_predictions=bool(cfg.get("log_predictions", False)),
        interference=pipeline.rsu.scheduler.interference if pipeline.rsu.scheduler else None,
    )


def build_logger(cfg: Dict[str, Any], suffix: str = "") -> ExperimentLogger:
    """Open the results tree for this run."""
    name = str(cfg["experiment_name"]) + suffix
    meta = ExperimentMeta(
        experiment_id=f"{name}-{int(cfg['seed'])}",
        experiment_name=name,
        paper=str(cfg["paper"]),
        architecture=str(cfg["model"]["name"]),
        dataset=str(cfg["dataset"]["name"]),
        seed=int(cfg["seed"]),
        deterministic=bool(cfg["deterministic"]),
        fault_config=dict(cfg["faults"]),
        tap_config=dict(cfg["taps"]),
        assumptions={k: str(v) for k, v in cfg.get("assumptions", {}).items()},
        environment=capture_environment(),
        resolved_config=cfg,
    )
    return ExperimentLogger(cfg["results_dir"], name, meta,
                            log_predictions=bool(cfg.get("log_predictions", False)),
                            logger_names=("lgcpbench",))


def apply_seed(cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))
