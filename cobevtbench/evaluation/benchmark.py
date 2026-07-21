"""
benchmark.py
------------
Run a whole fault sweep and write the results bundle.

The order is not incidental
---------------------------
The clean condition runs **first** and its per-frame outputs are cached,
because flip rate, SDC rate and fault-success rate are all defined against
it. A runner that evaluated fault conditions before establishing the clean
reference could not compute robustness at all, and one that recomputed the
clean run per condition would waste the majority of its time.

So: clean once (cache) -> each fault condition (compared to the cache) ->
one EvalRecord per condition -> the tables.

What lands on disk
------------------
Through ``cpbench.logbook.ExperimentLogger``:
``metrics.csv`` (one eval row per condition), ``fault_statistics.csv``
(robustness per condition), ``injection_summary.csv`` (every physical
fault), ``taps.csv`` and ``taps/`` (if taps are active), and a
``confusion_matrix.png`` per condition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from cpbench.logbook import EvalRecord, ExperimentLogger

from .sweeps import Condition, expand_sweep
from .tester import EvalResult

logger = logging.getLogger(__name__)

# A condition -> a tester bound to that condition's faulted dataset.
TesterFactory = Callable[[Condition, Optional[EvalResult]], Any]


@dataclass
class ConditionResult:
    """One condition's evaluation, kept for the summary table."""

    condition: Condition
    result: EvalResult


class FaultBenchmarkRunner:
    """Drive a fault sweep, clean-first, and persist the results.

    Purpose
        Turn a fault config plus a way to build per-condition testers into a
        full results bundle.

    Inputs
    ------
    tester_factory  ``(condition, reference) -> tester``. The runner does not
                    know how to build a dataset or a model; it is handed a
                    factory that wraps the condition's bridge and returns a
                    ready tester. This is what keeps the runner track-agnostic
                    and testable without a model.
    metric_kind     ``"segmentation"`` or ``"detection"`` -- only decides
                    which EvalRecord field the metrics land in.
    logbook         ExperimentLogger; None disables persistence (tests)
    dataset_name, split  logging labels
    taps_factory    optional ``() -> taps`` built fresh per condition, so each
                    condition's dumps are self-contained

    Outputs
    -------
    ``run(model)`` returns the list of :class:`ConditionResult`, clean first.

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, fault_config: Dict[str, Any],
                 tester_factory: TesterFactory,
                 metric_kind: str = "segmentation",
                 logbook: Optional[ExperimentLogger] = None,
                 dataset_name: str = "", split: str = "test",
                 taps_factory: Optional[Callable[[], Any]] = None) -> None:
        if metric_kind not in ("segmentation", "detection"):
            raise ValueError(
                f"unknown metric_kind {metric_kind!r}; expected "
                "'segmentation' or 'detection'")
        self.conditions = expand_sweep(fault_config)
        self.tester_factory = tester_factory
        self.metric_kind = metric_kind
        self.logbook = logbook
        self.dataset_name = dataset_name
        self.split = split
        self.taps_factory = taps_factory

    def _record(self, condition: Condition, result: EvalResult) -> EvalRecord:
        detection = result.metrics if self.metric_kind == "detection" else {}
        segmentation = result.metrics if self.metric_kind == "segmentation" else {}
        return EvalRecord(
            epoch=-1, dataset=self.dataset_name, split=self.split,
            condition={"name": condition.name, **_flatten(condition.config)},
            detection=detection, segmentation=segmentation,
            robustness=result.robustness, system=result.system,
            n_frames=result.n_frames, n_faults_injected=result.n_faults)

    def run(self, model) -> List[ConditionResult]:
        clean_condition = next(c for c in self.conditions if c.is_clean)
        logger.info("clean reference: %s", clean_condition.name)
        clean_tester = self.tester_factory(clean_condition, None)
        clean_tester.keep_predictions = True
        reference = clean_tester.run(model, taps=self._taps())

        results = [ConditionResult(clean_condition, reference)]
        self._persist(clean_condition, reference)

        for condition in self.conditions:
            if condition.is_clean:
                continue
            logger.info("condition: %s", condition.name)
            tester = self.tester_factory(condition, reference)
            result = tester.run(model, taps=self._taps())
            results.append(ConditionResult(condition, result))
            self._persist(condition, result)
        return results

    def _taps(self):
        return self.taps_factory() if self.taps_factory is not None else None

    def _persist(self, condition: Condition, result: EvalResult) -> None:
        if self.logbook is None:
            return
        self.logbook.log_eval(self._record(condition, result))
        if result.fault_records:
            self.logbook.log_fault_records(result.fault_records)
        if result.robustness:
            row = {"condition": condition.name, **result.robustness,
                   "n_faults": result.n_faults}
            self.logbook.log_fault_statistics(row)


def _flatten(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested condition config into scalar cells for the CSV.

    Kept shallow and string-valued: the condition column exists to be
    filtered and grouped, not to round-trip the config, which meta.json
    already stores in full.
    """
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, (str, int, float, bool)):
            flat[name] = value
        else:
            flat[name] = repr(value)
    return flat
