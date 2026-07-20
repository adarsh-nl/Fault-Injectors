"""
latency.py
----------
The paper's end-to-end latency model, Eq. 4 and Eq. 5.

Paper mapping
    Stage 1 (initiation broadcast + CAV reports):
        t_1 = D_init / R_t  +  ceil(|V| / Z) * D_info / R_t
    Stage 2 (task assignment broadcast):
        t_2 = D_ts / R_t
    Stage 3 (aggregation + fusion + upload), Eq. 4:
        t_3 = max_i ( t_a(a_i) + t_f(a_i) ) + D_rep / R_t
            = |S(V^)| + D_rep / R_t
    Stage 4 (global view broadcast):
        t_4 = D_G / R_t
    Total, Eq. 5:
        SUM t_i ~= t_delta + |S(V^)|
        t_delta = ( D_init + ceil(|V|/Z)*D_info + D_ts + D_rep + D_G ) / R_t

    Constraint (7a):  t_delta + |S(V^)| <= T,  with T = 100 ms (Table I).

Why t_delta is reported separately
    t_delta is a fixed overhead: it depends on |V| and the message sizes but
    NOT on the group set or the schedule. Only |S(V^)| responds to grouping
    and scheduling decisions. Logging them apart means a latency regression
    can be attributed to the scheduler rather than to a config change in
    message sizes -- and it makes the paper's own claim checkable, since the
    reductions reported in Fig. 5 must come from |S(V^)|.

Assumption B7 -- message sizes
    D_init, D_info, D_ts, D_rep and D_G are never given numerically in the
    paper. They are config-supplied with conventional defaults, and because
    t_delta is logged separately their contribution is always auditable
    rather than baked into a single number.

Assumption B4 -- the fusion cost model
    The paper gives per-model costs (2228 / 1400 / 2684 MFLOPs for CoBEVT /
    Where2comm / CoAlign) and a CAV capability of 0.1 TFLOPS, but does not
    state how cost scales with group size. We scale linearly in |V^_i|,
    matching the structure of Eq. 10's fusion load ``L_j = SUM y_i,j |V^_i| B``.
    Configurable, and recorded in the experiment metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Table I
DEFAULT_CAV_CAPACITY_TFLOPS: float = 0.1
DEFAULT_EDGE_CAPACITY_TFLOPS: float = 2.0      # section VI-C
DEFAULT_MAX_LATENCY_S: float = 100e-3
DEFAULT_SUBCHANNELS: int = 5

# Section VI-C, per-model fusion cost in MFLOPs.
MODEL_MFLOPS: Dict[str, float] = {
    "cobevt": 2228.0,
    "where2comm": 1400.0,
    "coalign": 2684.0,
}

# Assumption B7: message sizes in bits. Conventional defaults; overridable.
DEFAULT_MESSAGE_BITS: Dict[str, float] = {
    "D_init": 1024.0,
    "D_info": 512.0,
    "D_ts": 2048.0,
    "D_rep": 8192.0,
    "D_G": 16384.0,
}


class FusionLatencyModel:
    """t_f(a_i): how long a leader takes to fuse one area.

    Purpose
        Converts a group size into a compute time using the paper's own
        MFLOPs and TFLOPS numbers. Injected into the scheduler so that
        aggregation and fusion can be overlapped (section V-B) rather than
        summed naively.

    Inputs
    ------
    mflops_per_member  fusion cost per group member (section VI-C).
    capacity_tflops    the fusing device: 0.1 for a CAV, 2.0 for the edge
                       server -- which is exactly the asymmetry the
                       edge-assisted baseline exploits and LGCP trades away
                       in exchange for parallelism.

    Outputs
    -------
    ``fusion_time_s(n_members)`` -> seconds.

    Example
    -------
    >>> m = FusionLatencyModel.for_model("where2comm")
    >>> round(m.fusion_time_s(1) * 1e3, 1)      # ms
    14.0
    >>> round(m.fusion_time_s(3) * 1e3, 1)
    42.0
    >>> round(FusionLatencyModel.for_model("where2comm", edge=True)
    ...       .fusion_time_s(3) * 1e3, 2)
    2.1
    """

    def __init__(
        self,
        mflops_per_member: float = MODEL_MFLOPS["where2comm"],
        capacity_tflops: float = DEFAULT_CAV_CAPACITY_TFLOPS,
        cost_scale: float = 1.0,
    ) -> None:
        if mflops_per_member < 0:
            raise ValueError(f"mflops_per_member must be >= 0, got {mflops_per_member}")
        if capacity_tflops <= 0:
            raise ValueError(f"capacity_tflops must be > 0, got {capacity_tflops}")
        if cost_scale <= 0:
            raise ValueError(f"cost_scale must be > 0, got {cost_scale}")
        self.mflops_per_member = float(mflops_per_member)
        self.capacity_tflops = float(capacity_tflops)
        self.cost_scale = float(cost_scale)

    @classmethod
    def for_model(cls, name: str, edge: bool = False) -> "FusionLatencyModel":
        """Build from a paper model name (section VI-C costs).

        ``edge=True`` uses the 2 TFLOPS edge server instead of a 0.1 TFLOPS
        CAV, for the edge-assisted baseline.
        """
        key = name.lower()
        if key not in MODEL_MFLOPS:
            raise KeyError(
                f"unknown model {name!r}; expected one of {sorted(MODEL_MFLOPS)}"
            )
        return cls(
            mflops_per_member=MODEL_MFLOPS[key],
            capacity_tflops=(
                DEFAULT_EDGE_CAPACITY_TFLOPS if edge else DEFAULT_CAV_CAPACITY_TFLOPS
            ),
        )

    def fusion_time_s(self, n_members: int) -> float:
        """Seconds to fuse an area for a group of ``n_members`` (B4)."""
        if n_members <= 0:
            return 0.0
        flops = n_members * self.mflops_per_member * 1e6 * self.cost_scale
        return flops / (self.capacity_tflops * 1e12)

    @classmethod
    def area_scaled(
        cls,
        model: str,
        area_cells: float,
        total_cells: float,
        edge: bool = False,
    ) -> "FusionLatencyModel":
        """Scale the published per-model cost by the area's share of the map.

        Assumption B4, refined. Section VI-C's MFLOPs figures (2228 / 1400 /
        2684) are whole-model inference costs on a FULL BEV map. LGCP fuses an
        area-restricted patch -- roughly 28 of 8448 cells on OPV2V, about
        0.3% -- so charging the full cost per area overstates fusion time by
        two to three orders of magnitude.

        The difference is not academic. With the unscaled cost a 30-CAV
        deployment comes out fusion-bound at ~165 ms, blowing the 100 ms
        deadline; the paper reports LGCP completing well inside it (Fig. 5,
        Fig. 7). Scaling by area share reconciles the two, and identifies
        which reading of the paper's cost model the numbers depend on.

        Both are available: the unscaled constructor reproduces the literal
        reading, this one the proportional reading. Which is intended is not
        stated in the paper, so the choice is recorded rather than hidden.

        Example
        -------
        >>> m = FusionLatencyModel.area_scaled("where2comm", 28, 8448)
        >>> round(m.fusion_time_s(3) * 1e3, 3)         # ms
        0.139
        """
        base = cls.for_model(model, edge=edge)
        if total_cells <= 0:
            raise ValueError(f"total_cells must be > 0, got {total_cells}")
        return cls(
            mflops_per_member=base.mflops_per_member,
            capacity_tflops=base.capacity_tflops,
            cost_scale=float(area_cells) / float(total_cells),
        )


@dataclass(frozen=True)
class LatencyBreakdown:
    """Eq. 5, decomposed so each term is separately attributable.

    Attributes / units (all seconds)
    --------------------------------
    t_delta      fixed protocol overhead; independent of grouping.
    t_aggregate  max over areas of the moment its last packet arrives (t_a).
    t_fuse       max over areas of fusion duration (t_f), for reporting.
    t_schedule   |S(V^)| = max_i(t_a + t_f); what Eq. 4 actually charges.
    total        t_delta + t_schedule.
    deadline     T from Table I.

    Example
    -------
    >>> b = LatencyBreakdown(t_delta=0.002, t_aggregate=0.01, t_fuse=0.028,
    ...                       t_schedule=0.038, total=0.040, deadline=0.1)
    >>> b.deadline_met, round(b.total_ms, 1)
    (True, 40.0)
    """

    t_delta: float
    t_aggregate: float
    t_fuse: float
    t_schedule: float
    total: float
    deadline: float = DEFAULT_MAX_LATENCY_S

    @property
    def deadline_met(self) -> bool:
        """Constraint (7a): t_delta + |S(V^)| <= T."""
        return self.total <= self.deadline

    @property
    def total_ms(self) -> float:
        return self.total * 1e3

    @property
    def overhead_fraction(self) -> float:
        """Share of total latency that grouping/scheduling cannot influence.

        A high value means the measured system is dominated by fixed protocol
        overhead (assumption B7's message sizes) rather than by the
        contribution the paper is actually claiming -- worth knowing before
        reading a reduction ratio.
        """
        return self.t_delta / self.total if self.total > 0 else 0.0

    def as_record(self) -> Dict[str, Any]:
        """Flat dict for the logbook's LatencyRecord."""
        return {
            "t_delta_ms": self.t_delta * 1e3,
            "t_aggregate_ms": self.t_aggregate * 1e3,
            "t_fuse_ms": self.t_fuse * 1e3,
            "t_schedule_ms": self.t_schedule * 1e3,
            "t_total_ms": self.total * 1e3,
            "deadline_T_ms": self.deadline * 1e3,
            "deadline_met": self.deadline_met,
            "overhead_fraction": self.overhead_fraction,
        }


class LatencyModel:
    """Eq. 5: assemble t_delta and combine it with the schedule makespan.

    Purpose
        Owns the protocol overhead arithmetic. The schedule makespan comes
        from ``TransmissionScheduler``; this class never simulates anything
        itself, which keeps Eq. 5 auditable against the paper line by line.

    Inputs
    ------
    rate_bps       R_t, the data transmission rate.
    n_subchannels  Z, used by the ceil(|V|/Z) term.
    message_bits   B7's D_init / D_info / D_ts / D_rep / D_G.
    deadline_s     T (Table I: 100 ms).

    Example
    -------
    >>> lm = LatencyModel(rate_bps=27e6)
    >>> round(lm.t_delta(n_cavs=5) * 1e6, 1)      # microseconds
    1043.0
    >>> lm.breakdown(n_cavs=5, t_aggregate=0.01, t_fuse=0.028,
    ...              t_schedule=0.038).deadline_met
    True
    """

    def __init__(
        self,
        rate_bps: float = 27e6,
        n_subchannels: int = DEFAULT_SUBCHANNELS,
        message_bits: Optional[Dict[str, float]] = None,
        deadline_s: float = DEFAULT_MAX_LATENCY_S,
    ) -> None:
        if rate_bps <= 0:
            raise ValueError(f"rate_bps must be > 0, got {rate_bps}")
        if n_subchannels < 1:
            raise ValueError(f"n_subchannels must be >= 1, got {n_subchannels}")
        self.rate_bps = float(rate_bps)
        self.n_subchannels = int(n_subchannels)
        self.message_bits = dict(DEFAULT_MESSAGE_BITS)
        if message_bits:
            self.message_bits.update(message_bits)
        self.deadline_s = float(deadline_s)

    def t_delta(self, n_cavs: int) -> float:
        """Eq. 5's fixed overhead term.

        ``( D_init + ceil(|V|/Z)*D_info + D_ts + D_rep + D_G ) / R_t``

        The ceil term is the only part that grows with fleet size: |V| CAVs
        report over Z shared subchannels, so reporting takes ceil(|V|/Z)
        rounds.
        """
        m = self.message_bits
        rounds = math.ceil(max(int(n_cavs), 0) / self.n_subchannels)
        total_bits = (
            m["D_init"]
            + rounds * m["D_info"]
            + m["D_ts"]
            + m["D_rep"]
            + m["D_G"]
        )
        return total_bits / self.rate_bps

    def breakdown(
        self,
        n_cavs: int,
        t_aggregate: float,
        t_fuse: float,
        t_schedule: float,
    ) -> LatencyBreakdown:
        """Assemble Eq. 5 from a schedule's measured components."""
        delta = self.t_delta(n_cavs)
        return LatencyBreakdown(
            t_delta=delta,
            t_aggregate=float(t_aggregate),
            t_fuse=float(t_fuse),
            t_schedule=float(t_schedule),
            total=delta + float(t_schedule),
            deadline=self.deadline_s,
        )

    def objective(self, accuracy_proxy: float, breakdown: LatencyBreakdown) -> float:
        """Eq. 7's objective: mean area confidence divided by total latency.

        Inputs   accuracy_proxy : Eq. 3's (1/N) SUM_i F_i(V^_i).
                 breakdown      : Eq. 5's decomposition.
        Outputs  the ratio being maximised in P0. Returns 0.0 when the
                 deadline constraint (7a) is violated, since such a solution
                 is infeasible rather than merely poor -- collapsing it to a
                 finite score would let an infeasible schedule win a sweep.
        """
        if not breakdown.deadline_met:
            return 0.0
        if breakdown.total <= 0:
            return 0.0
        return float(accuracy_proxy) / breakdown.total
