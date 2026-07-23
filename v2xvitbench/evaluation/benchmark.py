"""
benchmark.py
------------
Run a whole sweep and write the results bundle.

The order is not incidental: the clean condition runs **first** and its
per-frame outputs are cached, because flip rate, SDC rate and fault-success
rate are all defined against it. A runner that evaluated fault conditions
before establishing the reference could not compute robustness at all; one
that recomputed the clean run per condition would spend most of its time on
it. ``order_conditions`` enforces clean-first and refuses a sweep without a
reference, rather than leaving either to the caller.

What lands on disk
------------------
Through ``cpbench.logbook.ExperimentLogger``: ``metrics.csv`` (one eval row
per condition), ``fault_statistics.csv`` (robustness per fault condition),
``injection_summary.csv`` (every fault fired, both planes, one row each),
``taps.csv`` and ``taps/`` when taps are active, plus ``meta.json`` and the
resolved ``config.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from cpbench.logbook import EvalRecord, ExperimentLogger

from v2xvitbench.evaluation.sweeps import (Condition, expand_sweep,
                                           order_conditions)
from v2xvitbench.evaluation.tester import EvalResult

logger = logging.getLogger(__name__)

# A condition -> a tester bound to that condition's faulted dataset/bridges.
TesterFactory = Callable[[Condition, Optional[EvalResult]], Any]


@dataclass
class ConditionResult:
    """One condition's evaluation, kept for the summary table."""

    condition: Condition
    result: EvalResult


class FaultBenchmarkRunner:
    """Drive a fault sweep, clean-first, and persist it.

    Purpose
        Turn a fault config plus a way to build per-condition testers into
        a full results bundle.

    Inputs
    ------
    fault_config    the fault group's YAML, already loaded.
    tester_factory  ``(condition, reference) -> tester``. The runner does
                    not know how to build a dataset, a bridge or a model;
                    it is handed a factory that wraps the condition's
                    bridges. That is what keeps it testable without a model.
    logbook         ExperimentLogger; None disables persistence (tests).
    taps_factory    optional ``() -> taps``, built fresh per condition so
                    each condition's dumps are self-contained.

    Outputs
    -------
    ``run(model)`` returns the list of :class:`ConditionResult`, in
    evaluation order (clean first).

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, fault_config: Optional[Dict[str, Any]],
                 tester_factory: TesterFactory,
                 logbook: Optional[ExperimentLogger] = None,
                 dataset_name: str = "", split: str = "test",
                 taps_factory: Optional[Callable[[], Any]] = None) -> None:
        self.conditions = order_conditions(expand_sweep(fault_config))
        self.tester_factory = tester_factory
        self.logbook = logbook
        self.dataset_name = dataset_name
        self.split = split
        self.taps_factory = taps_factory

    def _taps(self):
        return self.taps_factory() if self.taps_factory is not None else None

    def _record(self, condition: Condition, result: EvalResult) -> EvalRecord:
        return EvalRecord(
            epoch=-1, dataset=self.dataset_name, split=self.split,
            condition={"name": condition.name, **_flatten(condition.config)},
            detection=result.metrics, robustness=result.robustness,
            comms={}, system=result.system,
            n_frames=result.n_frames, n_faults_injected=result.n_faults)

    def _persist(self, condition: Condition, result: EvalResult) -> None:
        if self.logbook is None:
            return
        self.logbook.log_eval(self._record(condition, result))
        if result.fault_records:
            self.logbook.log_fault_records(result.fault_records)
        if result.robustness:
            self.logbook.log_fault_statistics({
                "condition": condition.name,
                **result.robustness, "n_faults": result.n_faults,
                **{f"det_{k}": v for k, v in result.metrics.items()}})

    def run(self, model) -> List[ConditionResult]:
        """Evaluate every condition, establishing the reference first."""
        results: List[ConditionResult] = []
        clean = self.conditions[0]
        logger.info("clean reference: %s", clean.name)
        tester = self.tester_factory(clean, None)
        tester.keep_predictions = True
        reference = tester.run(model, taps=self._taps())
        results.append(ConditionResult(clean, reference))
        self._persist(clean, reference)

        for condition in self.conditions[1:]:
            logger.info("condition: %s", condition.name)
            tester = self.tester_factory(condition, reference)
            result = tester.run(model, taps=self._taps())
            results.append(ConditionResult(condition, result))
            self._persist(condition, result)
        return results


class CleanBenchmarkRunner(FaultBenchmarkRunner):
    """The reproduction path: the clean condition alone.

    Sharing the fault runner's machinery means the clean numbers come from
    exactly the same code path as the faulted ones -- a reproduction that
    used a separate loop would prove less than it appears to. Robustness
    columns are absent by construction: with no fault condition there is
    nothing to compare a reference against.

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, tester_factory: TesterFactory, **kwargs) -> None:
        super().__init__({"sweep": []}, tester_factory, **kwargs)


def _flatten(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested condition config into scalar CSV cells.

    Kept shallow and string-valued: the condition columns exist to be
    filtered and grouped, not to round-trip the config, which ``meta.json``
    already stores in full.

    >>> _flatten({"metadata_pipeline": {"type_flip": {"p_flip": 0.5}}})
    {'metadata_pipeline.type_flip.p_flip': 0.5}
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
