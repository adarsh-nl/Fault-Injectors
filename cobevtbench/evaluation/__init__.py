"""
Tester, benchmark runners, sweep expansion, static+dynamic merge.

The clean condition is run first and cached per frame, because flip rate,
silent-data-corruption rate and fault success rate are all defined against a
clean reference. A clean run that silently injected something would make every
downstream comparison meaningless, so `bridge.is_clean` is asserted upstream.

Contents
--------
sweeps     expand a fault config's `sweep` into named conditions
tester     SegmentationTester / DetectionTester: one model, one condition
benchmark  FaultBenchmarkRunner: clean-first sweep + results bundle
merge      MergedSegmentationModel: overlay the dynamic and static models (A8)
"""

from .benchmark import ConditionResult, FaultBenchmarkRunner
from .merge import (MERGED_CLASSES, MergedSegmentationModel, merge_label_maps)
from .sweeps import Condition, expand_sweep, name_condition
from .tester import (DetectionTester, EvalResult, SegmentationTester)

__all__ = ["FaultBenchmarkRunner", "ConditionResult",
           "SegmentationTester", "DetectionTester", "EvalResult",
           "expand_sweep", "name_condition", "Condition",
           "MergedSegmentationModel", "merge_label_maps", "MERGED_CLASSES"]
