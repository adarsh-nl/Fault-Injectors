"""
accumulators.py
---------------
System-level metrics for LGCP, accumulated over frames.

Paper mapping -- section VI-B, the three reported metrics
    * Average precision at IoU 0.3 / 0.5 / 0.7 (detection; handled by
      ``corabench.metrics.DetectionEvaluator``, wrapped in ``evaluator.py``).
    * "Amount of data transmission. It is defined as the total amount of data
      transmitted by all CAVs to complete a single collaborative perception."
    * "End-to-end latency ... measured as the time interval from the
      initiation to the completion of a collaborative perception task."

Plus two the paper does not report but a benchmark needs
    * Schedule health -- subchannel utilisation, deadline misses, unscheduled
      packets. Without these, a latency regression is a number with no cause.
    * Coverage -- how many areas went unperceived. This is the control
      plane's headline robustness signal: a falsified-report or leader-failure
      fault destroys coverage long before it moves AP, because an orphaned
      area produces no detections at all rather than wrong ones.

Design
    Each accumulator takes ``FrameResult`` objects and returns a flat dict of
    floats. They are separate classes rather than one because they answer
    different questions and are reported in different places -- and because a
    baseline that has no schedule (no-collaboration) can use the comm and
    latency accumulators without carrying a meaningless schedule section.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class _Accumulator:
    """Shared plumbing: collect per-frame values, reduce at the end."""

    def __init__(self) -> None:
        self._rows: List[Dict[str, float]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def reset(self) -> None:
        self._rows.clear()

    def _reduce(self, prefix: str) -> Dict[str, float]:
        """mean/max over collected rows, prefixed for the logbook."""
        if not self._rows:
            return {}
        keys = self._rows[0].keys()
        out: Dict[str, float] = {}
        for key in keys:
            values = np.asarray([r[key] for r in self._rows], dtype=np.float64)
            out[f"{prefix}_{key}_mean"] = float(values.mean())
            out[f"{prefix}_{key}_max"] = float(values.max())
        out[f"{prefix}_n_frames"] = float(len(self._rows))
        return out


class CommunicationMetrics(_Accumulator):
    """Amount of data transmission, and reduction against the baselines.

    Reports totals rather than only means, because the paper's Fig. 4 metric
    is per-collaboration volume and a mean would hide a heavy-tailed frame.

    Example
    -------
    >>> m = CommunicationMetrics()
    >>> m.compute()
    {}
    """

    def add(self, result) -> None:
        comm = result.comm
        self._rows.append(
            {
                "bits_v2v": float(comm.v2v_bits),
                "bits_v2i": float(comm.v2i_bits),
                "bits_total": float(comm.total_bits),
                "n_packets": float(comm.n_packets),
                "reduction_vs_edge": float(comm.reduction_vs_edge_assisted),
                "reduction_vs_vehicle": float(comm.reduction_vs_vehicle_based),
            }
        )

    def compute(self) -> Dict[str, float]:
        out = self._reduce("comm")
        if self._rows:
            out["comm_bits_total_sum"] = float(
                sum(r["bits_total"] for r in self._rows)
            )
            # The headline ratios: computed on the SUMS, not as a mean of
            # per-frame ratios, which would be an average of ratios and not
            # the ratio of totals the paper reports.
            v2v = sum(r["bits_v2v"] for r in self._rows)
            total = sum(r["bits_total"] for r in self._rows)
            out["comm_v2v_fraction"] = float(v2v / total) if total else 0.0
        return out


class LatencyMetrics(_Accumulator):
    """End-to-end latency and its decomposition (Eq. 5).

    Keeps ``t_delta`` separate from ``t_schedule`` so a latency change can be
    attributed to the scheduler rather than to B7's message-size assumptions.
    """

    def add(self, result) -> None:
        lat = result.latency
        self._rows.append(
            {
                "t_delta_ms": lat.t_delta * 1e3,
                "t_aggregate_ms": lat.t_aggregate * 1e3,
                "t_fuse_ms": lat.t_fuse * 1e3,
                "t_schedule_ms": lat.t_schedule * 1e3,
                "t_total_ms": lat.total * 1e3,
                "overhead_fraction": lat.overhead_fraction,
                "deadline_met": float(lat.deadline_met),
            }
        )

    def compute(self) -> Dict[str, float]:
        out = self._reduce("latency")
        if self._rows:
            met = [r["deadline_met"] for r in self._rows]
            # Constraint (7a) is a hard bound, so the violation RATE matters
            # more than mean latency: a system that meets the deadline 90% of
            # the time is not 90% as good as one that always does.
            out["latency_deadline_violation_rate"] = float(1.0 - np.mean(met))
        return out


class ScheduleMetrics(_Accumulator):
    """Schedule health -- why a latency number came out the way it did.

    ``n_conflicts`` is always 0 under Algorithm 2 (conflict-freedom is
    guaranteed by construction and asserted by test). It becomes non-zero
    only under a schedule-corruption fault or a baseline scheduler, which is
    precisely what makes it worth reporting.
    """

    def __init__(self, interference=None) -> None:
        super().__init__()
        self.interference = interference

    def add(self, result) -> None:
        schedule = result.schedule
        conflicts = 0
        if self.interference is not None:
            by_slot: Dict[float, list] = {}
            for p in schedule.packets:
                by_slot.setdefault(p.t, []).append(p)
            conflicts = sum(
                len(self.interference.audit(slot)) for slot in by_slot.values()
            )
        self._rows.append(
            {
                "n_slots": float(schedule.n_slots),
                "n_packets": float(len(schedule.packets)),
                "subchannel_utilisation": schedule.subchannel_utilisation,
                "makespan_ms": schedule.makespan * 1e3,
                "n_unscheduled": float(len(schedule.unscheduled)),
                "n_conflicts": float(conflicts),
            }
        )

    def compute(self) -> Dict[str, float]:
        out = self._reduce("schedule")
        if self._rows:
            out["schedule_conflicts_total"] = float(
                sum(r["n_conflicts"] for r in self._rows)
            )
            out["schedule_unscheduled_total"] = float(
                sum(r["n_unscheduled"] for r in self._rows)
            )
        return out


class CoverageMetrics(_Accumulator):
    """Area coverage -- the control plane's headline robustness signal.

    An orphaned area produces NO detections rather than wrong ones, so
    coverage loss is invisible in precision and only weakly visible in
    recall, while being the most direct consequence of a control-plane fault.
    Reporting it separately is what makes leader-failure and falsified-report
    injections measurable at all.
    """

    def add(self, result) -> None:
        sel = result.selection
        n_areas = max(sel.n_areas, 1)
        self._rows.append(
            {
                "n_occupied_areas": float(result.occupied_area_ids.size),
                "n_areas": float(sel.n_areas),
                "n_orphaned": float(sel.n_orphaned),
                "n_leaderless": float(sel.n_leaderless),
                "n_unperceived": float(sel.n_unperceived),
                # The headline coverage number keys on UNPERCEIVED, not
                # orphaned: a leaderless area still has members, so a rate
                # built on membership reports perfect coverage while whole
                # areas vanish from the global view.
                "orphan_rate": float(sel.n_unperceived / n_areas),
                "mean_group_size": sel.mean_group_size,
                "max_group_size": float(sel.max_group_size),
                "mean_area_confidence": sel.accuracy_proxy,
                "leader_load_max": sel.makespan,
                "load_imbalance": sel.load_imbalance,
                "objective": result.objective,
            }
        )

    def compute(self) -> Dict[str, float]:
        return self._reduce("coverage")
