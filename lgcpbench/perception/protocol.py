"""
protocol.py
-----------
The seam between LGCP's control plane and whatever perception backbone it
orchestrates.

Paper mapping
    Section I: "The LGCP framework adopts existing collaborative perception
    models for the perception tasks of areas."
    Section VI-B: the paper evaluates CoBEVT, Where2comm and CoAlign.

Why a four-method protocol instead of one forward()
    OpenCOOD backbones expose a single ``forward(data_dict)`` that encodes all
    agents, fuses across ALL of them on the FULL BEV map, and detects -- in
    one call. LGCP cannot use that, because it needs to:

        1. get per-agent features and their confidence BEFORE any fusion,
           since confidence is what the RSU schedules on (Eq. 1);
        2. fuse at a LEADER, over a GROUP, restricted to ONE AREA (Eq. 2, 10);
        3. detect per area, so the RSU can aggregate area results (step 4).

    A monolithic forward cannot express any of those three. Splitting into
    encode / confidence / fuse / detect is therefore not a stylistic choice --
    it is the minimum decomposition in which the paper's method is
    expressible. The OpenCOOD adapter reuses the pretrained submodules
    (pillar_vfe, scatter, backbone, cls_head, reg_head, fusion_net) rather
    than calling forward(), so checkpoint fidelity is preserved.

Deviation from the design document
    The design doc listed a fourth method ``decode -> Detections``. That is
    split here into ``detect`` (feature map -> cls/reg maps, the backbone's
    job) and a separate anchor-aware decoder (maps -> boxes, owned by
    ``lgcpbench.perception.decode``). Decoding needs the area's anchor slice,
    which is grid geometry rather than model weights; keeping it out of the
    backbone means every backbone shares one decoder implementation instead
    of reimplementing NMS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

    def runtime_checkable(cls):  # type: ignore[misc]
        return cls


@dataclass(frozen=True)
class AgentInputs:
    """Collated per-agent sensor inputs for one frame.

    Purpose
        Carries every CAV of one frame as a single flat "agent batch", so the
        encoder runs once per frame rather than once per CAV -- and, more
        importantly, never once per AREA. Encoding is frame-level; only
        fusion is area-level.

    Attributes / shapes
    -------------------
    features    (P, max_points, 9) float  decorated pillar point features
    coords      (P, 3) int64              [agent_index, row(y), col(x)]
    num_points  (P,) int64                valid points per pillar
    n_agents    V, the number of CAVs in this frame
    agent_ids   (V,) tuple of str, index-aligned with ``coords[:, 0]``
    positions   (V, 2) float              CAV (x, y) in the ego frame; used by
                                          the network layer for path loss and
                                          interference, and by occupancy
    ego_index   row of the ego CAV within the agent batch

    Notes
    -----
    Faults have ALREADY been applied upstream by ``DataFaultBridge`` before
    this object exists (plane 1). Nothing downstream corrupts data.
    """

    features: torch.Tensor
    coords: torch.Tensor
    num_points: torch.Tensor
    n_agents: int
    agent_ids: Tuple[str, ...] = ()
    positions: Optional[np.ndarray] = None
    ego_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    """Backend-specific payload.

    The native backbone reads ``features``/``coords``/``num_points`` in
    corabench's pillar convention. OpenCOOD models expect their OWN
    preprocessor's output -- ``processed_lidar`` with ``voxel_features``,
    ``voxel_coords`` (4-column, spconv layout) and ``voxel_num_points`` --
    which is a different tensor layout, not a renaming.

    Rather than force one convention on both and silently mistranslate,
    each backend reads what it needs from here. A backend that finds its key
    missing raises with an actionable message instead of misinterpreting
    another backend's tensors.
    """

    def __post_init__(self) -> None:
        if self.agent_ids and len(self.agent_ids) != self.n_agents:
            raise ValueError(
                f"agent_ids has {len(self.agent_ids)} entries but n_agents={self.n_agents}"
            )
        if self.positions is not None:
            pos = np.asarray(self.positions)
            if pos.ndim != 2 or pos.shape[0] != self.n_agents or pos.shape[1] < 2:
                raise ValueError(
                    f"positions must be (n_agents, >=2) = ({self.n_agents}, >=2), "
                    f"got {pos.shape}"
                )
        if not 0 <= self.ego_index < max(self.n_agents, 1):
            raise ValueError(
                f"ego_index {self.ego_index} out of range for n_agents={self.n_agents}"
            )

    def agent_index(self, agent_id: str) -> int:
        """Row of ``agent_id`` within the agent batch."""
        try:
            return self.agent_ids.index(agent_id)
        except ValueError:
            raise KeyError(
                f"unknown agent {agent_id!r}; known: {list(self.agent_ids)}"
            ) from None


@dataclass(frozen=True)
class Detections:
    """Object detections for one area, or one aggregated global view.

    Shapes
    ------
    boxes   (M, 7) float  x, y, z, l, w, h, yaw[rad] in the ego frame
    scores  (M,) float    confidence in [0, 1], descending
    area_id originating area, or None for an aggregated global view

    Example
    -------
    >>> d = Detections(boxes=np.zeros((3, 7)), scores=np.array([.9, .5, .1]))
    >>> len(d)
    3
    >>> len(Detections.empty())
    0
    """

    boxes: np.ndarray
    scores: np.ndarray
    area_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 7:
            raise ValueError(f"boxes must be (M, 7), got {self.boxes.shape}")
        if self.scores.shape != (self.boxes.shape[0],):
            raise ValueError(
                f"scores must be (M,) matching boxes ({self.boxes.shape[0]},), "
                f"got {self.scores.shape}"
            )

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @classmethod
    def empty(cls, area_id: Optional[int] = None) -> "Detections":
        """An empty result -- what an orphaned area produces."""
        return cls(
            boxes=np.zeros((0, 7), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            area_id=area_id,
        )


@runtime_checkable
class CollabPerceptionModel(Protocol):
    """Any collaborative perception backbone LGCP can orchestrate.

    Implementations
    ---------------
    NativeReferenceBackbone  CPU, torch-only, no OpenCOOD. Backs every unit
                             test and all synthetic development.
    OpenCOODBackbone         Where2comm / CoBEVT / CoAlign with pretrained
                             weights. Runs in its own Python 3.7 environment;
                             reproduces the paper's Table II.
    StubBackbone             fixed tensors, for scheduler-only tests.

    Attributes
    ----------
    feature_channels  C of the BEV feature map (256 for OPV2V Where2comm).
    feature_hw        (H, W) of the BEV feature map (48, 176 for OPV2V).

    Every method takes ``taps=None`` (measurement plane) and must be
    bit-identical with and without taps.
    """

    feature_channels: int
    feature_hw: Tuple[int, int]

    def encode(
        self, inputs: AgentInputs, *, taps: Optional[Any] = None
    ) -> torch.Tensor:
        """Per-CAV BEV features. Inputs AgentInputs -> (V, C, H, W)."""
        ...

    def confidence(
        self, features: torch.Tensor, *, taps: Optional[Any] = None
    ) -> torch.Tensor:
        """Paper Eq. 1's ``f_gen``. (V, C, H, W) -> (V, 1, H, W) in [0, 1]."""
        ...

    def fuse(
        self,
        ego: torch.Tensor,
        collab: Sequence[torch.Tensor],
        *,
        taps: Optional[Any] = None,
    ) -> torch.Tensor:
        """Leader-side fusion over one group, one area.

        Inputs  ego (C, h, w); collab a sequence of (C, h, w) from members.
        Output  (C, h, w).
        """
        ...

    def detect(
        self, fused: torch.Tensor, *, taps: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """Fused feature -> detection maps.

        Inputs  (C, h, w).
        Output  {"cls": (A, h, w) logits, "reg": (A*7, h, w) box deltas}.
        """
        ...
