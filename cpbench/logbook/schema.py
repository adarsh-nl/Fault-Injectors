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
    # AMP diagnostics. The scale is the quantity that distinguishes fp16
    # gradient overflow (scale backs off, run recovers) from a non-finite
    # forward or loss (scale backs off forever and the run is dead), and
    # without it in the row that distinction is unrecoverable after the fact.
    # opt_state_amax is max|exp_avg_sq| across the optimizer: it goes
    # non-finite at the step Adam's moments are poisoned, which is otherwise
    # invisible until the checkpoint is inspected.
    scaler_scale: float = 0.0
    n_skipped_steps: float = 0.0
    opt_state_amax: float = 0.0

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["phase"] = "train"
        return row


@dataclass
class EvalRecord:
    """One evaluation under one condition (metrics.csv, phase='eval').

    `condition` carries the fault setting (type, sigma/delay/rate, scope);
    detection, segmentation, robustness and system metrics are flat sub-dicts
    so the CSV row is self-describing (det_ap50, seg_iou_vehicle,
    rob_flip_rate, sys_latency_ms, ...).

    `detection` and `segmentation` are separate fields rather than one
    `task_metrics` dict because a run may populate either, and the CSV is the
    union of columns across all rows: a shared key would silently merge
    `ap70` and `iou_vehicle` into one column when detection and segmentation
    experiments land in the same results directory.

    `comms` is transmitted-volume metrics (see `cpbench.metrics.comms`), and
    it is deliberately not folded into `system`. Latency and memory are
    properties of the machine the run happened on; communication volume is a
    property of the *model's decisions* -- for a paper such as Where2comm it
    is a headline result, varies with the input, and is what a fault can move
    without touching a single wall-clock number. Reporting it in `sys_*`
    columns would file the paper's contribution under profiling.
    """

    epoch: int
    dataset: str
    split: str
    condition: Dict[str, Any] = field(default_factory=dict)
    detection: Dict[str, float] = field(default_factory=dict)
    segmentation: Dict[str, float] = field(default_factory=dict)
    robustness: Dict[str, float] = field(default_factory=dict)
    system: Dict[str, float] = field(default_factory=dict)
    comms: Dict[str, float] = field(default_factory=dict)
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
        row.update(_flat("seg_", self.segmentation))
        row.update(_flat("rob_", self.robustness))
        row.update(_flat("sys_", self.system))
        row.update(_flat("comm_", self.comms))
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


@dataclass
class SegPredictionRecord:
    """Per-frame segmentation summary (predictions.jsonl; optional).

    Purpose
        The segmentation analogue of PredictionRecord, for the same opt-in
        sink.

    Why this stores counts and not pixels
        A BEV map is 256x256. Writing the predicted label map per frame, per
        fault condition, would produce gigabytes of JSONL that nobody reads,
        and JSON is the worst possible container for a dense integer array.
        The confusion counts are what any downstream analysis actually needs
        -- every per-class IoU, precision, recall and F1 is recoverable from
        them exactly, with no loss. For the pixels themselves, enable the
        qualitative PNG dump instead and put its path in ``sample_path``.

    Shapes
    ------
    confusion    (K, K) nested list, rows = ground truth, cols = prediction
    per_class_iou  (K,) float, NaN classes already zeroed
    class_names  (K,) str, so the record is readable without the config

    Example
    -------
    >>> rec = SegPredictionRecord(frame=0, class_names=["background", "vehicle"],
    ...                           confusion=[[1, 0], [1, 2]],
    ...                           per_class_iou=[0.5, 0.6667],
    ...                           mean_confidence=0.81)
    >>> rec.as_dict()["frame"], rec.as_dict()["class_names"][1]
    (0, 'vehicle')
    """

    frame: int
    class_names: List[str]
    confusion: List[List[int]]
    per_class_iou: List[float]
    mean_confidence: float = 0.0
    sample_path: Optional[str] = None      # qualitative PNG, if dumped

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
