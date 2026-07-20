"""
benchmark.py
------------
Benchmark runners.

CleanBenchmarkRunner   one clean evaluation; caches per-frame outputs as the
                       robustness reference.
FaultBenchmarkRunner   a sweep of physical fault conditions; each condition
                       rebuilds the dataset with a fresh DataFaultBridge,
                       evaluates, and scores robustness against the cached
                       clean reference (delta-AP, flip rate, SDC, fault
                       success). Everything lands in the ExperimentLogger:
                       metrics.csv, fault_statistics.csv,
                       injection_summary.csv, taps.csv.

The runner never mutates the model and never touches tensors: corruption is
entirely inside each condition's bridge (physical, upstream), measurement is
passive (taps + metrics).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from cpbench.faults.bridge import DataFaultBridge
from cpbench.logbook.experiment import ExperimentLogger
from cpbench.logbook.schema import EvalRecord
from cpbench.metrics.robustness import FramePair, RobustnessMetrics
from cpbench.observation.taps import TapProtocol
from .sweeps import expand_sweep
from .tester import Tester, TestResult

logger = logging.getLogger(__name__)

# a factory rebuilding the eval dataset around a given bridge
DatasetFactory = Callable[[Optional[DataFaultBridge]], Any]


class CleanBenchmarkRunner:
    """Baseline evaluation under no faults.

    Example
    -------
    >>> clean = CleanBenchmarkRunner(model, make_ds, device, explog).run()
    """

    def __init__(self, model, dataset_factory: DatasetFactory,
                 device: torch.device, explog: Optional[ExperimentLogger],
                 taps: Optional[TapProtocol] = None, batch_size: int = 2,
                 dataset_name: str = "dataset") -> None:
        self.model = model
        self.dataset_factory = dataset_factory
        self.device = device
        self.explog = explog
        self.taps = taps
        self.batch_size = batch_size
        self.dataset_name = dataset_name

    def run(self, max_batches: Optional[int] = None) -> TestResult:
        dataset = self.dataset_factory(None)      # clean bridge
        result = Tester(self.model, dataset, self.device,
                        batch_size=self.batch_size, taps=self.taps).run(
                            max_batches=max_batches)
        if self.explog:
            self.explog.log_eval(EvalRecord(
                epoch=-1, dataset=self.dataset_name, split="test",
                condition={"fault": "clean"}, detection=result.detection,
                system={**result.system, **result.comm},
                n_frames=int(result.detection.get("n_frames", 0)),
                n_faults_injected=0))
            self.explog.flush()
        return result


class FaultBenchmarkRunner:
    """Sweep physical fault conditions and score robustness vs clean.

    Parameters
    ----------
    clean_result  the CleanBenchmarkRunner output (reference frames).
    sweep         list of pipeline-config dicts (see sweeps.py).
    bridge_kwargs extra DataFaultBridge kwargs (agent_scope, seed, fps).

    Example
    -------
    >>> runner = FaultBenchmarkRunner(model, make_ds, device, explog, clean)
    >>> rows = runner.run([{"pose_error": {"sigma_xy": .4,
    ...                                    "sigma_heading": .4}}])
    """

    def __init__(self, model, dataset_factory: DatasetFactory,
                 device: torch.device, explog: Optional[ExperimentLogger],
                 clean_result: TestResult,
                 taps: Optional[TapProtocol] = None, batch_size: int = 2,
                 dataset_name: str = "dataset", fps: float = 10.0,
                 bridge_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.dataset_factory = dataset_factory
        self.device = device
        self.explog = explog
        self.clean = clean_result
        self.taps = taps
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.fps = fps
        self.bridge_kwargs = dict(bridge_kwargs or {})

    def _robustness(self, faulted: TestResult) -> Dict[str, float]:
        rm = RobustnessMetrics()
        per_frame_faults: Dict[int, int] = {}
        for rec in faulted.fault_records:
            per_frame_faults[rec.frame] = per_frame_faults.get(rec.frame, 0) + 1
        for frame_no, ref in self.clean.frames.items():
            cur = faulted.frames.get(frame_no)
            if cur is None:
                continue
            rm.add(FramePair(
                frame=frame_no,
                clean_boxes=ref["boxes"], clean_scores=ref["scores"],
                fault_boxes=cur["boxes"], fault_scores=cur["scores"],
                gt_boxes=ref["gt"],
                n_faults=per_frame_faults.get(frame_no, 0),
                had_numeric_error=frame_no in faulted.numeric_error_frames))
        rob = rm.compute()
        for key in ("ap50", "ap70"):
            rob[f"delta_{key}"] = (self.clean.detection.get(key, 0.0) -
                                   faulted.detection.get(key, 0.0))
        return rob

    def run(self, sweep: Sequence[Dict[str, Any]],
            max_batches: Optional[int] = None
            ) -> List[Tuple[str, TestResult, Dict[str, float]]]:
        results = []
        for name, pipeline_cfg in expand_sweep(sweep):
            logger.info("benchmark condition: %s", name)
            bridge = DataFaultBridge(
                {"name": name, "pipeline": pipeline_cfg,
                 **self.bridge_kwargs}, fps=self.fps) \
                if pipeline_cfg else None
            dataset = self.dataset_factory(bridge)
            faulted = Tester(self.model, dataset, self.device,
                             batch_size=self.batch_size,
                             taps=self.taps).run(max_batches=max_batches)
            robustness = self._robustness(faulted)
            if self.explog:
                self.explog.log_eval(EvalRecord(
                    epoch=-1, dataset=self.dataset_name, split="test",
                    condition={"fault": name, **_flatten(pipeline_cfg)},
                    detection=faulted.detection, robustness=robustness,
                    system={**faulted.system, **faulted.comm},
                    n_frames=int(faulted.detection.get("n_frames", 0)),
                    n_faults_injected=faulted.n_faults))
                self.explog.log_fault_records(faulted.fault_records)
                self.explog.log_fault_statistics({
                    "condition": name, **_flatten(pipeline_cfg),
                    "ap50": faulted.detection.get("ap50", 0.0),
                    "ap70": faulted.detection.get("ap70", 0.0),
                    **robustness,
                    "n_faults": faulted.n_faults})
                self.explog.flush()
            results.append((name, faulted, robustness))
        return results


def _flatten(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in cfg.items():
        full = f"{prefix}{key}"
        if isinstance(val, dict):
            out.update(_flatten(val, f"{full}."))
        else:
            out[full] = val
    return out
