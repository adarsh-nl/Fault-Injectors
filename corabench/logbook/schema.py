"""
schema.py
---------
Typed records for everything an experiment logs.

Every record is a dataclass with an ``as_row()`` flattener, so the same
object feeds the CSV sink, the JSON sink and TensorBoard without duplicated
field lists. See docs/corabench_design.md section 6.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _flat(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


@dataclass
class ExperimentMeta:
    """Immutable description of one experiment run (written once, as JSON).

    Captures everything needed to reproduce the run: identity, code + config
    versions, environment, seeds and the paper assumption flags (A1-A9).
    """

    experiment_id: str
    experiment_name: str
    paper: str
    architecture: str
    dataset: str
    seed: int
    deterministic: bool
    fault_config: Dict[str, Any] = field(default_factory=dict)
    tap_config: Dict[str, Any] = field(default_factory=dict)
    assumptions: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    resolved_config: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainRecord:
    """One training step/epoch row (metrics.csv, phase='train')."""

    epoch: int
    batch: int
    loss_total: float
    loss_cls: float = 0.0
    loss_reg: float = 0.0
    loss_align: float = 0.0
    loss_pac: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    batch_time_s: float = 0.0
    gpu_mem_mb: float = 0.0

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["phase"] = "train"
        return row


@dataclass
class EvalRecord:
    """One evaluation under one condition (metrics.csv, phase='eval').

    `condition` carries the fault setting (type, sigma/delay/rate, scope);
    detection, robustness and system metrics are flat sub-dicts so the CSV
    row is self-describing (det_ap50, rob_flip_rate, sys_latency_ms, ...).
    """

    epoch: int
    dataset: str
    split: str
    condition: Dict[str, Any] = field(default_factory=dict)
    detection: Dict[str, float] = field(default_factory=dict)
    robustness: Dict[str, float] = field(default_factory=dict)
    system: Dict[str, float] = field(default_factory=dict)
    per_class: Dict[str, float] = field(default_factory=dict)
    n_frames: int = 0
    n_faults_injected: int = 0

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "phase": "eval", "epoch": self.epoch, "dataset": self.dataset,
            "split": self.split, "n_frames": self.n_frames,
            "n_faults_injected": self.n_faults_injected,
        }
        row.update(_flat("cond_", self.condition))
        row.update(_flat("det_", self.detection))
        row.update(_flat("rob_", self.robustness))
        row.update(_flat("sys_", self.system))
        row.update(_flat("cls_", self.per_class))
        return row


@dataclass
class PredictionRecord:
    """Per-frame outputs (predictions.jsonl; optional, can be large).

    boxes/scores are the detection analogue of softmax + top-k requested in
    the benchmark spec: every retained box with its confidence, plus the
    matched ground-truth assignment.
    """

    frame: int
    boxes: List[List[float]]            # (M, 7) x,y,z,l,w,h,yaw
    scores: List[float]                 # (M,) confidence in [0,1]
    labels: List[int]                   # (M,) class ids
    gt_boxes: List[List[float]]         # (G, 7)
    matched_gt: List[int]               # (M,) gt index or -1 (false positive)
    branch: List[str] = field(default_factory=list)   # 'lc' | 'pac' per box

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
