"""
adapter.py
----------
Run a pretrained OpenCOOD model as an LGCP ``CollabPerceptionModel``.

What this does and does not do
    It DRIVES OpenCOOD's submodules -- ``pillar_vfe``, ``scatter``,
    ``backbone``, ``shrink_conv``, ``cls_head``, ``reg_head``, ``fusion_net``
    -- rather than calling ``forward(data_dict)``. That is not stylistic:
    ``forward`` encodes every agent, fuses across ALL of them on the FULL BEV
    map, and detects, in one call. LGCP needs confidence before fusion, fusion
    at a leader over a group restricted to one area, and detection per area.
    No monolithic forward can express that ordering.

    Because only the composition changes and never the weights, a pretrained
    checkpoint is used exactly as trained.

Two OpenCOOD behaviours that silently corrupt results
    Both were verified in the OpenCOOD sources and are asserted here rather
    than trusted.

    1. ``Communication.forward`` takes a DIFFERENT BRANCH in train vs eval.
       In training the communication mask is random top-K
       (``K = int(H*W*random.uniform(0, 1))``) and the configured ``threshold``
       is ignored entirely. Any communication-volume or confidence measurement
       taken in train mode is meaningless. The adapter refuses to run unless
       the model is in eval mode.

    2. ``train_utils.load_saved_model`` returns ``(initial_epoch, model)`` --
       not just the model -- and loads with ``strict=False``. A checkpoint
       whose keys have drifted therefore loads PARTIALLY and SILENTLY, leaving
       randomly-initialised layers in a model that looks trained. The adapter
       verifies the loaded key set and fails loudly.

Environment
    OpenCOOD is locked to Python 3.7 (``numba==0.49.0``), needs ``spconv``
    (unpinned, installed out-of-band) and CUDA. It cannot share the core
    environment, which is why the import is lazy and why the native reference
    backbone exists. See ``slurm/opencood_env.sbatch``.

VERIFICATION STATUS
    This adapter was written against OpenCOOD sources read at
    ``github.com/DerrickXuNu/OpenCOOD@main`` and is unit-tested against a stub
    that mirrors that module structure. It has NOT been executed against a
    real OpenCOOD install or real pretrained weights -- that requires the
    py3.7 + spconv + CUDA environment. Treat the first cluster run as
    integration testing, not as a regression check.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cpbench.observation.taps import TapProtocol, emit

from ..protocol import AgentInputs
from .fusion import FusionStrategy, build_fusion_strategy

logger = logging.getLogger(__name__)


class OpenCOODBackbone(nn.Module):
    """A pretrained OpenCOOD model, orchestrated by LGCP.

    Purpose
        Reproduce the paper's Table II by running the models it evaluates --
        Where2comm, CoBEVT, CoAlign -- inside the LGCP protocol.

    Inputs
    ------
    model         an instantiated OpenCOOD model (from ``create_model``).
    core_method   the ``model.core_method`` string, selecting the fusion
                  strategy.
    feature_hw    (H, W) of the BEV feature map after any shrink layer.
    channels      C of that map.

    Outputs (per CollabPerceptionModel)
    -----------------------------------
    encode      AgentInputs      -> (V, C, H, W)
    confidence  (V, C, H, W)     -> (V, 1, H, W)
    fuse        (C,h,w) + [...]  -> (C, h, w)
    detect      (C, h, w)        -> {"cls": (A, h, w), "reg": (A*7, h, w)}
    """

    def __init__(
        self,
        model: nn.Module,
        core_method: str,
        feature_hw: Tuple[int, int],
        channels: int,
        num_anchors: int = 2,
    ) -> None:
        super().__init__()
        self.model = model
        self.core_method = core_method
        self.feature_hw = (int(feature_hw[0]), int(feature_hw[1]))
        self.feature_channels = int(channels)
        self.num_anchors = int(num_anchors)

        for name in ("pillar_vfe", "scatter", "backbone", "cls_head", "reg_head"):
            if not hasattr(model, name):
                raise AttributeError(
                    f"OpenCOOD model {type(model).__name__} has no `{name}`; "
                    f"the adapter drives submodules directly and cannot proceed"
                )
        self.fusion: FusionStrategy = build_fusion_strategy(core_method, model)
        self.shrink_flag = bool(getattr(model, "shrink_flag", False))

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, model_cfg: Dict[str, Any], spec, device) -> "OpenCOODBackbone":
        """Build from ``configs/model/<name>.yaml``.

        Requires ``hypes_yaml`` (the OpenCOOD config) and, for meaningful
        numbers, ``checkpoint``.
        """
        try:
            from opencood.hypes_yaml.yaml_utils import load_yaml
            from opencood.tools import train_utils
        except ImportError as exc:  # pragma: no cover - needs the py3.7 env
            raise ImportError(
                "OpenCOOD is not importable. It requires its own Python 3.7 "
                "environment (numba==0.49.0, spconv, CUDA); build it with "
                "slurm/opencood_env.sbatch. Use model=native for the CPU "
                "reference backbone. Original error: " + str(exc)
            ) from exc

        hypes_path = model_cfg.get("hypes_yaml")
        if not hypes_path:
            raise ValueError(
                f"model.hypes_yaml is required for the opencood backend "
                f"(point it at opencood/hypes_yaml/{model_cfg['core_method']}.yaml)"
            )
        hypes = load_yaml(str(hypes_path))
        model = train_utils.create_model(hypes)

        checkpoint = model_cfg.get("checkpoint")
        if checkpoint:
            cls._load_checkpoint(model, Path(checkpoint))
        else:
            logger.warning(
                "no model.checkpoint given: the OpenCOOD model is RANDOMLY "
                "INITIALISED and its detection numbers will be meaningless. "
                "Set model.checkpoint to reproduce Table II."
            )

        model = model.to(device).eval()
        return cls(
            model=model,
            core_method=str(model_cfg["core_method"]),
            feature_hw=spec.feature_hw,
            channels=int(model_cfg["channels"]),
            num_anchors=int(model_cfg.get("num_anchors", 2)),
        )

    @staticmethod
    def _load_checkpoint(model: nn.Module, path: Path) -> None:
        """Load weights, refusing a silent partial load.

        ``train_utils.load_saved_model`` uses ``strict=False``, so a drifted
        checkpoint leaves randomly-initialised layers in a model that reports
        success. We call ``load_state_dict`` ourselves and inspect the result.
        """
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")

        state = torch.load(str(path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        incompatible = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing:
            raise RuntimeError(
                f"checkpoint {path} is missing {len(missing)} parameter(s) the "
                f"model expects, e.g. {missing[:5]}. OpenCOOD loads with "
                f"strict=False, which would leave these randomly initialised in "
                f"a model that looks trained. Refusing to continue."
            )
        if unexpected:
            logger.warning(
                "checkpoint %s has %d unexpected key(s), e.g. %s; these were "
                "ignored", path, len(unexpected), unexpected[:5],
            )
        logger.info("loaded OpenCOOD checkpoint %s", path)

    # ------------------------------------------------------------------ #
    # guards
    # ------------------------------------------------------------------ #

    def _assert_eval(self, what: str) -> None:
        if self.model.training:
            raise RuntimeError(
                f"{what} requires the OpenCOOD model to be in eval mode. "
                f"Where2comm's Communication module takes a different branch "
                f"in training -- a random top-K mask that ignores the "
                f"configured threshold entirely -- so any confidence or "
                f"communication measurement taken in train mode is invalid. "
                f"Call model.eval() first."
            )

    # ------------------------------------------------------------------ #
    # protocol
    # ------------------------------------------------------------------ #

    def encode(
        self, inputs: AgentInputs, *, taps: Optional[TapProtocol] = None
    ) -> torch.Tensor:
        """Per-CAV BEV features, encoded once per frame.

        Reads OpenCOOD's own preprocessor output from ``inputs.extra``: the
        voxel layout differs from corabench's pillar convention, and silently
        reinterpreting one as the other would produce plausible-looking
        garbage.
        """
        self._assert_eval("encode")

        processed = inputs.extra.get("processed_lidar")
        if processed is None:
            raise KeyError(
                "the OpenCOOD backend needs `processed_lidar` in "
                "AgentInputs.extra (voxel_features / voxel_coords / "
                "voxel_num_points from OpenCOOD's SpVoxelPreprocessor). "
                "corabench's pillar tensors use a different layout and cannot "
                "be substituted."
            )

        batch = {
            "voxel_features": processed["voxel_features"],
            "voxel_coords": processed["voxel_coords"],
            "voxel_num_points": processed["voxel_num_points"],
            "record_len": torch.tensor(
                [inputs.n_agents], dtype=torch.long,
                device=processed["voxel_features"].device,
            ),
        }

        batch = self.model.pillar_vfe(batch)
        emit(taps, batch.get("pillar_features"), module="OpenCOODBackbone",
             location="lgcp/perception/pillar_features")

        batch = self.model.scatter(batch)
        emit(taps, batch.get("spatial_features"), module="OpenCOODBackbone",
             location="lgcp/perception/scatter_bev")

        batch = self.model.backbone(batch)
        features = batch["spatial_features_2d"]

        if self.shrink_flag:
            features = self.model.shrink_conv(features)

        emit(taps, features, module="OpenCOODBackbone",
             location="lgcp/perception/bev_features")

        if tuple(features.shape[-2:]) != self.feature_hw:
            raise RuntimeError(
                f"OpenCOOD encoder produced {tuple(features.shape[-2:])} but the "
                f"adapter is configured for feature_hw={self.feature_hw}; check "
                f"dataset.grid against the model's hypes_yaml cav_lidar_range"
            )
        return features

    def confidence(
        self, features: torch.Tensor, *, taps: Optional[TapProtocol] = None
    ) -> torch.Tensor:
        """Paper Eq. 1's ``f_gen`` -- design doc derivation D1.

        This is exactly what Where2comm does: the SHARED ``cls_head`` applied
        to the pre-fusion per-agent map, then sigmoid, then max over anchors.
        Verified in ``point_pillar_where2comm.py`` (``psm_single =
        self.cls_head(spatial_features_2d)``) and ``where2comm_fuse.py``
        (``.sigmoid().max(dim=1, keepdim=True)``).
        """
        self._assert_eval("confidence")

        logits = self.model.cls_head(features)
        emit(taps, logits, module="OpenCOODBackbone",
             location="lgcp/perception/psm_single")

        probs = torch.sigmoid(logits)
        emit(taps, probs, module="OpenCOODBackbone",
             location="lgcp/perception/psm_sigmoid")

        conf, _ = probs.max(dim=1, keepdim=True)
        emit(taps, conf, module="OpenCOODBackbone",
             location="lgcp/perception/confidence_map")
        return conf

    def fuse(
        self,
        ego: torch.Tensor,
        collab: Sequence[torch.Tensor],
        *,
        taps: Optional[TapProtocol] = None,
    ) -> torch.Tensor:
        """Leader-side fusion over one group, one area."""
        self._assert_eval("fuse")

        emit(taps, ego, module="OpenCOODBackbone",
             location="lgcp/perception/fuse_ego_in", n_collab=len(collab))

        if not collab:
            emit(taps, ego, module="OpenCOODBackbone",
                 location="lgcp/perception/fused_feature", n_collab=0)
            return ego

        for i, f in enumerate(collab):
            if f.shape != ego.shape:
                raise ValueError(
                    f"collab[{i}] has shape {tuple(f.shape)}, expected "
                    f"{tuple(ego.shape)}"
                )

        stack = torch.stack([ego, *collab], dim=0)
        emit(taps, stack, module="OpenCOODBackbone",
             location="lgcp/perception/fuse_stack")

        fused = self.fusion(stack)
        emit(taps, fused, module="OpenCOODBackbone",
             location="lgcp/perception/fused_feature",
             n_collab=len(collab), strategy=self.fusion.name)
        return fused

    def detect(
        self, fused: torch.Tensor, *, taps: Optional[TapProtocol] = None
    ) -> Dict[str, torch.Tensor]:
        """Detection maps for one area, from the pretrained heads."""
        self._assert_eval("detect")

        if fused.dim() != 3:
            raise ValueError(f"expected (C, h, w), got {tuple(fused.shape)}")
        batched = fused.unsqueeze(0)

        cls = self.model.cls_head(batched)
        emit(taps, cls, module="OpenCOODBackbone",
             location="lgcp/perception/cls_logits")

        reg = self.model.reg_head(batched)
        emit(taps, reg, module="OpenCOODBackbone",
             location="lgcp/perception/reg_map")

        return {"cls": cls[0], "reg": reg[0]}
