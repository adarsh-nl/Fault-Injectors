"""
experiment.py
-------------
ExperimentLogger: one object owning every sink of one experiment run.

Output layout (created under ``<root>/<experiment_name>/``)::

    config.yaml             resolved configuration
    meta.json               ExperimentMeta (env, seeds, git, assumptions)
    metrics.csv             train + eval rows (union of columns)
    metrics.json            final summary (best/last metrics per condition)
    training.log            python-logging mirror of the console
    tensorboard/            scalars (only if tensorboard is installed)
    checkpoints/            trainer checkpoints
    fault_statistics.csv    aggregated robustness stats per condition
    injection_summary.csv   every physically injected fault (FaultRecord)
    taps.csv                observation-tap statistics (if taps active)
    predictions.jsonl       per-frame predictions (opt-in)

Example
-------
>>> logger = ExperimentLogger("results", "cora_opv2v_clean", meta)  # doctest: +SKIP
>>> logger.log_train(TrainRecord(epoch=0, batch=1, loss_total=1.2))  # doctest: +SKIP
>>> logger.close()                                                   # doctest: +SKIP
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .schema import EvalRecord, ExperimentMeta, PredictionRecord, TrainRecord

logger = logging.getLogger(__name__)

try:  # TensorBoard is optional (not installed on every machine/HPC image)
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _HAS_TB = False


class _CsvSink:
    """CSV file with a dynamic column union.

    Rows may carry different keys (eval conditions vary); the sink keeps all
    rows in memory and rewrites the file on flush so the header is always the
    union, in first-seen order. Row counts are small (one per epoch /
    condition / fault), so rewriting is cheap and crash-safe enough.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: List[Dict[str, Any]] = []
        self.columns: List[str] = []

    def add(self, row: Dict[str, Any]) -> None:
        for key in row:
            if key not in self.columns:
                self.columns.append(key)
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns, restval="")
            writer.writeheader()
            writer.writerows(self.rows)


class ExperimentLogger:
    """All logging for one experiment run: CSV + JSON + TensorBoard + console.

    Parameters
    ----------
    root      results root directory (e.g. ``results/``).
    name      experiment name -> subdirectory.
    meta      ExperimentMeta written to meta.json (and config.yaml from
              ``meta.resolved_config``).
    log_predictions  write predictions.jsonl (large; default False).
    """

    def __init__(self, root: "str | Path", name: str, meta: ExperimentMeta,
                 log_predictions: bool = False) -> None:
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        self.meta = meta
        self.log_predictions_enabled = log_predictions

        self._metrics = _CsvSink(self.dir / "metrics.csv")
        self._faults = _CsvSink(self.dir / "injection_summary.csv")
        self._fault_stats = _CsvSink(self.dir / "fault_statistics.csv")
        self._taps = _CsvSink(self.dir / "taps.csv")
        self._pred_fh = None
        self._summary: Dict[str, Any] = {"experiment_id": meta.experiment_id,
                                         "eval": []}

        # console + file logging (python logging, never print)
        self._file_handler = logging.FileHandler(self.dir / "training.log")
        self._file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger("corabench").addHandler(self._file_handler)
        logging.getLogger("corabench").setLevel(logging.INFO)

        self.tb = SummaryWriter(str(self.dir / "tensorboard")) if _HAS_TB else None
        if not _HAS_TB:
            logger.info("tensorboard not installed; skipping TB sink")

        with (self.dir / "meta.json").open("w") as fh:
            json.dump(meta.as_dict(), fh, indent=2, default=str)
        with (self.dir / "config.yaml").open("w") as fh:
            yaml.safe_dump(meta.resolved_config, fh, sort_keys=False)
        logger.info("experiment %s -> %s", meta.experiment_id, self.dir)

    # -- record sinks -------------------------------------------------------

    def log_train(self, rec: TrainRecord) -> None:
        self._metrics.add(rec.as_row())
        if self.tb:
            step = rec.epoch * 1_000_000 + rec.batch
            for key in ("loss_total", "loss_cls", "loss_reg", "loss_align",
                        "loss_pac", "lr", "grad_norm"):
                self.tb.add_scalar(f"train/{key}", getattr(rec, key), step)

    def log_eval(self, rec: EvalRecord) -> None:
        row = rec.as_row()
        self._metrics.add(row)
        self._summary["eval"].append(row)
        if self.tb:
            tag = "_".join(str(v) for v in rec.condition.values()) or "clean"
            for key, val in rec.detection.items():
                self.tb.add_scalar(f"eval/{tag}/{key}", val, rec.epoch)

    def log_fault_records(self, records: Iterable[Any]) -> None:
        """FaultRecords from DataFaultBridge -> injection_summary.csv."""
        for rec in records:
            self._faults.add(rec.as_row())

    def log_fault_statistics(self, row: Dict[str, Any]) -> None:
        """One aggregated robustness row per condition -> fault_statistics.csv."""
        self._fault_stats.add(row)

    def log_tap_records(self, records: Iterable[Any]) -> None:
        """TapRecords from observation recorders -> taps.csv."""
        for rec in records:
            self._taps.add(rec.as_row())

    def log_prediction(self, rec: PredictionRecord) -> None:
        if not self.log_predictions_enabled:
            return
        if self._pred_fh is None:
            self._pred_fh = (self.dir / "predictions.jsonl").open("w")
        self._pred_fh.write(json.dumps(rec.as_dict()) + "\n")

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Free-form TensorBoard scalar (system metrics, profiling)."""
        if self.tb:
            self.tb.add_scalar(tag, value, step)

    @property
    def checkpoints_dir(self) -> Path:
        return self.dir / "checkpoints"

    # -- lifecycle ----------------------------------------------------------

    def flush(self) -> None:
        for sink in (self._metrics, self._faults, self._fault_stats, self._taps):
            sink.flush()
        if self._pred_fh:
            self._pred_fh.flush()
        if self.tb:
            self.tb.flush()

    def close(self) -> None:
        """Flush all sinks and write the metrics.json summary."""
        self.flush()
        with (self.dir / "metrics.json").open("w") as fh:
            json.dump(self._summary, fh, indent=2, default=str)
        if self._pred_fh:
            self._pred_fh.close()
        if self.tb:
            self.tb.close()
        logging.getLogger("corabench").removeHandler(self._file_handler)
        self._file_handler.close()
        logger.info("experiment closed: %s", self.dir)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
