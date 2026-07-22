"""
benchmark.py
------------
Run a whole sweep and write the results bundle.

The order is not incidental
---------------------------
Within each bandwidth group the clean condition runs **first** and its
per-frame outputs are cached, because flip rate, SDC rate and fault-success
rate are all defined against it. A runner that evaluated fault conditions
before establishing the reference could not compute robustness at all; one that
recomputed the clean run per condition would spend most of its time on it.

And the reference is *per group*. Comparing a faulted run at one budget against
a clean run at another would attribute the bandwidth reduction to the fault,
inflating every robustness number by an amount that grows as the budget
shrinks. That failure is silent -- the numbers look plausible -- which is why
the grouping is enforced in :func:`~w2cbench.evaluation.sweeps.group_conditions`
rather than left to the caller.

Swapping the selector, and putting it back
------------------------------------------
A bandwidth condition is applied by exchanging the model's selector for the
run's duration. It is done with a context manager rather than by assignment
because an exception mid-condition would otherwise leave every later condition
running the wrong strategy -- and the results would still look like a complete
sweep.

What lands on disk
------------------
Through ``cpbench.logbook.ExperimentLogger``: ``metrics.csv`` (one eval row per
condition, now with ``comm_*`` columns beside ``det_*``),
``fault_statistics.csv``, ``injection_summary.csv``, ``taps.csv`` and ``taps/``
when taps are active, plus ``meta.json`` and the resolved ``config.yaml``.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from cpbench.logbook import EvalRecord, ExperimentLogger

from ..comm.selection import make_selector
from .sweeps import Condition, expand_sweep, group_conditions
from .tester import EvalResult

logger = logging.getLogger(__name__)

# A condition -> a tester bound to that condition's faulted dataset.
TesterFactory = Callable[[Condition, Optional[EvalResult]], Any]


@dataclass
class ConditionResult:
    """One condition's evaluation, kept for the summary table."""

    condition: Condition
    result: EvalResult


@contextlib.contextmanager
def selector_override(model, setting: Optional[Dict[str, Any]]):
    """Temporarily swap the model's selection strategy.

    Restores the original even on an exception. Without that, a failure in one
    condition would leave every later condition silently running a different
    strategy, and the sweep would still look complete.

    ``channels`` is read from the model's encoder, because a byte budget only
    becomes a cell count once the feature width is known.
    """
    if not setting:
        yield
        return
    original = model.selector
    channels = getattr(model.encoder, "out_channels", None)
    spec = dict(setting)
    kind = spec.pop("kind", "budget")
    model.selector = make_selector(kind, channels=channels, **spec)
    logger.info("selector override: %s -> %s", type(original).__name__,
                model.selector)
    try:
        yield
    finally:
        model.selector = original


class FaultBenchmarkRunner:
    """Drive a fault sweep, clean-first per bandwidth group, and persist it.

    Purpose
        Turn a fault config plus a way to build per-condition testers into a
        full results bundle.

    Inputs
    ------
    fault_config    the fault group's YAML, already loaded.
    tester_factory  ``(condition, reference) -> tester``. The runner does not
                    know how to build a dataset or a model; it is handed a
                    factory that wraps the condition's bridge. That is what
                    keeps it testable without a model.
    bandwidth_sweep optional list of selector settings to cross with (the
                    paper's curve, design doc section 9.3). None evaluates at
                    the model's configured strategy only.
    logbook         ExperimentLogger; None disables persistence (tests).
    taps_factory    optional ``() -> taps``, built fresh per condition so each
                    condition's dumps are self-contained.

    Outputs
    -------
    ``run(model)`` returns the list of :class:`ConditionResult`, in evaluation
    order (each group's clean first).

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, fault_config: Optional[Dict[str, Any]],
                 tester_factory: TesterFactory,
                 bandwidth_sweep: Optional[List[Dict[str, Any]]] = None,
                 logbook: Optional[ExperimentLogger] = None,
                 dataset_name: str = "", split: str = "test",
                 taps_factory: Optional[Callable[[], Any]] = None) -> None:
        self.conditions = expand_sweep(fault_config, bandwidth_sweep)
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
            condition={"name": condition.name, "bandwidth": condition.group,
                       **_flatten(condition.config)},
            detection=result.metrics, robustness=result.robustness,
            comms=result.comms, system=result.system,
            n_frames=result.n_frames, n_faults_injected=result.n_faults)

    def _persist(self, condition: Condition, result: EvalResult) -> None:
        if self.logbook is None:
            return
        self.logbook.log_eval(self._record(condition, result))
        if result.fault_records:
            self.logbook.log_fault_records(result.fault_records)
        if result.robustness:
            self.logbook.log_fault_statistics({
                "condition": condition.name, "bandwidth": condition.group,
                **result.robustness, "n_faults": result.n_faults,
                "comm_log2_bytes": result.comms.get("log2_bytes", float("nan")),
                **{f"det_{k}": v for k, v in result.metrics.items()}})

    def run(self, model) -> List[ConditionResult]:
        """Evaluate every condition, establishing a reference per group."""
        results: List[ConditionResult] = []
        for group, members in group_conditions(self.conditions):
            clean = members[0]
            logger.info("group %s: clean reference %s", group, clean.name)
            with selector_override(model, clean.selector):
                tester = self.tester_factory(clean, None)
                tester.keep_predictions = True
                reference = tester.run(model, taps=self._taps())
            results.append(ConditionResult(clean, reference))
            self._persist(clean, reference)

            for condition in members[1:]:
                logger.info("group %s: condition %s", group, condition.name)
                with selector_override(model, condition.selector):
                    tester = self.tester_factory(condition, reference)
                    result = tester.run(model, taps=self._taps())
                results.append(ConditionResult(condition, result))
                self._persist(condition, result)
        return results


class CleanBenchmarkRunner(FaultBenchmarkRunner):
    """The reproduction path: no faults, optionally the full bandwidth curve.

    Purpose
        Produce the paper's accuracy-versus-bandwidth figure, which is a
        sweep over budgets with no fault at all. Sharing the fault runner's
        machinery means the clean numbers come from exactly the same code
        path as the faulted ones -- a reproduction that used a separate loop
        would prove less than it appears to.

    Robustness columns are absent by construction: with no fault condition
    there is nothing to compare a reference against.

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, tester_factory: TesterFactory, **kwargs) -> None:
        super().__init__({"sweep": []}, tester_factory, **kwargs)


def _flatten(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested condition config into scalar CSV cells.

    Kept shallow and string-valued: the condition columns exist to be filtered
    and grouped, not to round-trip the config, which ``meta.json`` already
    stores in full.

    >>> _flatten({"pipeline": {"pose_error": {"sigma_xy": 0.4}}})
    {'pipeline.pose_error.sigma_xy': 0.4}
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
