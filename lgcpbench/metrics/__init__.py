"""
lgcpbench.metrics
=================
Measurement for LGCP runs.

Detection metrics (AP / precision / recall / F1 at IoU 0.3, 0.5, 0.7) reuse
``corabench.metrics.DetectionEvaluator``. This package adds the system-level
metrics the paper reports (data transmission, end-to-end latency) plus two a
benchmark needs and the paper does not report: schedule health and area
coverage.

AUROC is recorded as inapplicable rather than fabricated -- see
``evaluator.INAPPLICABLE_METRICS``.

Example
-------
>>> from lgcpbench.metrics import LGCPEvaluator
>>> ev = LGCPEvaluator(pipeline)                       # doctest: +SKIP
>>> result = ev.run(dataset)                           # doctest: +SKIP
>>> result.as_dict()["ap50"]                           # doctest: +SKIP
"""

from .accumulators import (
    CommunicationMetrics,
    CoverageMetrics,
    LatencyMetrics,
    ScheduleMetrics,
)
from .evaluator import (
    INAPPLICABLE_METRICS,
    PAPER_IOU_THRESHOLDS,
    LGCPEvaluator,
    RunResult,
)

__all__ = [
    "LGCPEvaluator",
    "RunResult",
    "PAPER_IOU_THRESHOLDS",
    "INAPPLICABLE_METRICS",
    "CommunicationMetrics",
    "LatencyMetrics",
    "ScheduleMetrics",
    "CoverageMetrics",
]
