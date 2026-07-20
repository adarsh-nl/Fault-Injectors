"""
simulate.py
-----------
Network-only latency sweep -- paper Fig. 7's dense multi-CAV regime.

Usage
    python -m lgcpbench.scripts.simulate --n-cavs 30
    python -m lgcpbench.scripts.simulate --n-cavs 5 10 15 20 25 30 --repeats 5

Why this exists and what it is not
    The paper obtains its 5-30 CAV latency curve from a CARLA + OpenCDA + NS3
    co-simulation. That integration is deliberately out of scope (design doc
    section 15, item 3). But LGCP's control plane -- partitioning, grouping,
    leader election, Algorithm 2 and the Eq. 5 latency model -- consumes and
    produces plain data. It needs no backbone, no dataset and no GPU.

    So the TREND can be reproduced analytically, on CPU, in seconds, which is
    what this script does. It is not the co-simulator, and the absolute
    numbers depend on assumption B7's message sizes. What it does establish is
    how latency scales with fleet size under the paper's own algorithms and
    Table I parameters.

Why confidences are synthetic here
    With no backbone there is no f_gen, so per-area confidences are drawn from
    a Beta distribution rather than computed. That is honest for a scaling
    study -- the question is how Algorithm 2's makespan grows with |V| and N,
    which depends on the GROUP STRUCTURE, not on which particular CAV sees
    which area. The draw is seeded, so the sweep is reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..confidence.estimator import AreaConfidenceMatrix
from ..network import (
    FusionLatencyModel,
    InterferenceModel,
    LatencyModel,
    RateModel,
    ShadowingModel,
    TransmissionScheduler,
    build_packets,
)
from ..roi import AreaGrid, BoxOccupancy
from ..selection import GreedyGroupSelector, SelectionAlgorithm

logger = logging.getLogger("lgcpbench.simulate")

# OPV2V RoI, the paper's 280 m x 80 m detection range.
DEFAULT_POINT_RANGE = (-140.8, -38.4, -3.0, 140.8, 38.4, 1.0)


def simulate_frame(
    n_cavs: int,
    grid: AreaGrid,
    rng: np.random.Generator,
    delta_g: float = 0.075,
    max_group_size: int = 5,
    model_mflops: float = 1400.0,
    n_objects: int = 40,
) -> Dict[str, Any]:
    """One synthetic collaboration cycle, control plane only.

    Inputs
    ------
    n_cavs        fleet size.
    grid          the area partition.
    rng           seeded generator.
    delta_g       Eq. 8 threshold.
    model_mflops  section VI-C fusion cost (Where2comm: 1400).
    n_objects     traffic density, which sets how many areas are occupied.

    Outputs
    -------
    flat dict of latency, schedule and grouping metrics for one frame.
    """
    x_min, y_min, _, x_max, y_max, _ = grid.point_range
    cav_xy = np.stack(
        [rng.uniform(x_min, x_max, n_cavs), rng.uniform(y_min, y_max, n_cavs)], axis=1
    )
    object_xy = np.stack(
        [rng.uniform(x_min, x_max, n_objects), rng.uniform(y_min, y_max, n_objects)],
        axis=1,
    )

    occupied = np.flatnonzero(BoxOccupancy()(grid, boxes=object_xy, cav_positions=cav_xy))
    agent_ids = tuple(f"cav{i}" for i in range(n_cavs))

    # Beta(2, 3) is a plausible confidence spread: mostly moderate, few near
    # 1.0. The scaling result depends on group structure, not on this shape.
    values = rng.beta(2.0, 3.0, size=(n_cavs, occupied.size))
    matrix = AreaConfidenceMatrix(
        values=values, area_ids=occupied, agent_ids=agent_ids
    )

    selection = SelectionAlgorithm(
        GreedyGroupSelector(delta_g=delta_g, max_group_size=max_group_size)
    )(matrix)

    positions = {aid: tuple(p) for aid, p in zip(agent_ids, cav_xy)}
    scheduler = TransmissionScheduler(
        InterferenceModel(positions, rate_model=RateModel(
            shadowing=ShadowingModel(seed=int(rng.integers(1 << 30)))
        )),
        fusion_model=FusionLatencyModel(mflops_per_member=model_mflops),
    )
    packets = build_packets(selection.groups)
    schedule = scheduler.schedule(
        packets,
        group_sizes={g.area_id: g.size for g in selection.groups},
        leaders={g.area_id: g.leader for g in selection.groups if g.leader},
    )

    latency = LatencyModel().breakdown(
        n_cavs=n_cavs,
        t_aggregate=schedule.t_aggregate,
        t_fuse=schedule.t_fuse,
        t_schedule=schedule.makespan,
    )

    return {
        "n_cavs": n_cavs,
        "n_areas": selection.n_areas,
        "n_orphaned": selection.n_orphaned,
        "mean_group_size": selection.mean_group_size,
        "leader_load_max": selection.makespan,
        "load_imbalance": selection.load_imbalance,
        "n_packets": len(packets),
        "n_slots": schedule.n_slots,
        "subchannel_utilisation": schedule.subchannel_utilisation,
        **latency.as_record(),
    }


def run(
    n_cavs: Sequence[int],
    repeats: int = 5,
    seed: int = 2026,
    delta_g: float = 0.075,
    objects_per_cav: float = 4.0,
    results_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Sweep fleet size, averaging over ``repeats`` synthetic frames.

    ``objects_per_cav`` scales traffic density with the fleet, which matters
    more than it looks. Holding the object count FIXED while CAVs grow makes
    latency fall with fleet size -- more CAVs means more leaders to spread
    fusion across (Eq. 10) while the work stays constant. That is a real
    property of LGCP, but it is not the regime the paper measures: in a denser
    deployment there are more vehicles AND more CAVs, so the number of
    occupied areas grows too. Scaling the two together is the honest default;
    set ``objects_per_cav=0`` to hold traffic fixed and see the other effect
    in isolation.
    """
    grid = AreaGrid(DEFAULT_POINT_RANGE)
    rows: List[Dict[str, Any]] = []

    for n in n_cavs:
        rng = np.random.default_rng(seed + n)
        n_objects = int(round(objects_per_cav * n)) if objects_per_cav else 40
        frames = [
            simulate_frame(n, grid, rng, delta_g=delta_g, n_objects=n_objects)
            for _ in range(repeats)
        ]
        row: Dict[str, Any] = {"n_cavs": n, "repeats": repeats,
                               "delta_g": delta_g, "n_objects": n_objects}
        for key in frames[0]:
            if key == "n_cavs":
                continue
            values = [float(f[key]) for f in frames]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_max"] = float(np.max(values))
        rows.append(row)

        logger.info(
            "n_cavs=%-3d  areas=%-5.1f  packets=%-6.1f  slots=%-6.1f  "
            "latency=%6.1fms  deadline_met=%.2f",
            n, row["n_areas_mean"], row["n_packets_mean"], row["n_slots_mean"],
            row["t_total_ms_mean"], row["deadline_met_mean"],
        )

    if results_dir is not None:
        _write(Path(results_dir), rows)
    return rows


def _write(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with (out_dir / "simulate.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "simulate.json").write_text(json.dumps(rows, indent=2))
    logger.info("wrote %d rows to %s", len(rows), out_dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LGCP network-only latency sweep (paper Fig. 7 trend).",
    )
    parser.add_argument("--n-cavs", type=int, nargs="+",
                        default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--delta-g", type=float, default=0.075)
    parser.add_argument("--objects-per-cav", type=float, default=4.0,
                        help="traffic density scaling; 0 holds it fixed")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    run(
        n_cavs=args.n_cavs,
        repeats=args.repeats,
        seed=args.seed,
        delta_g=args.delta_g,
        objects_per_cav=args.objects_per_cav,
        results_dir=args.results_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
