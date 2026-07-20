"""
benchmark.py
------------
Clean run + fault sweep, into a complete results tree.

Usage
    python -m lgcpbench.scripts.benchmark
    python -m lgcpbench.scripts.benchmark faults=pose_error
    python -m lgcpbench.scripts.benchmark faults=comm_stress --max-frames 20
    python -m lgcpbench.scripts.benchmark lgcp.confidence.delta_g=0.05 seed=7

What it produces
    results/<experiment_name>/
        config.yaml            resolved config + environment + assumptions
        metrics.csv            one row per condition
        metrics.json
        fault_statistics.csv   what was injected, per condition
        injection_summary.csv  per-record audit trail
        control_plane.csv      RSU decisions (when taps=stats)
        benchmark.log

Why every condition rebuilds the dataset
    A fault condition is a property of the DATA, not of the model. Rebuilding
    the dataset around a fresh bridge -- and never touching the model --
    means the only difference between conditions is the corruption, which is
    what makes the comparison attributable.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import common

logger = logging.getLogger("lgcpbench.benchmark")


def expand_sweep(sweep: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand a fault sweep spec into named conditions.

    Input   [{"pose_error": {"sigma_xy": [0.2, 0.4]}}]
    Output  [("pose_error_sxy0.2", {"pose_error": {"sigma_xy": 0.2}}), ...]

    Example
    -------
    >>> expand_sweep([{"agent_drop": {"p_drop": [0.1, 0.5]}}])
    [('agent_drop_p0.1', {'agent_drop': {'p_drop': 0.1}}), \
('agent_drop_p0.5', {'agent_drop': {'p_drop': 0.5}})]
    """
    abbrev = {
        "sigma_xy": "sxy", "sigma_heading": "sh", "mu_delay": "mu",
        "sigma_jitter": "jit", "p_drop": "p", "keep_fraction": "keep",
    }
    out: List[Tuple[str, Dict[str, Any]]] = []
    for entry in sweep or []:
        for injector, params in entry.items():
            for param, values in params.items():
                for value in (values if isinstance(values, list) else [values]):
                    tag = f"{injector}_{abbrev.get(param, param)}{value}"
                    out.append((tag, {injector: {param: value}}))
    return out


def run(cfg: Dict[str, Any], max_frames: Optional[int] = None) -> Dict[str, Any]:
    """Run the clean condition plus every sweep condition."""
    common.apply_seed(cfg)
    adapter = common.build_adapter(cfg)
    taps, control_tap, _ = common.build_taps(cfg)

    # Conditions carry BOTH planes: (name, physical_pipeline, control_pipeline).
    # The clean reference passes EXPLICIT empty dicts, not None. `None` means
    # "fall back to the configured faults", which for the reference condition
    # would silently corrupt it -- and a contaminated reference invalidates
    # every comparison drawn against it.
    conditions: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = [("clean", {}, {})]

    physical = cfg["faults"].get("pipeline") or {}
    control = cfg["faults"].get("control_pipeline") or {}
    if physical or control:
        conditions.append((str(cfg["faults"]["name"]), physical, control))
    for name, spec_ in expand_sweep(cfg["faults"].get("sweep") or []):
        conditions.append((name, spec_, {}))
    for name, spec_ in expand_sweep(cfg["faults"].get("control_sweep") or []):
        conditions.append((f"ctl_{name}", {}, spec_))

    rows: List[Dict[str, Any]] = []
    with common.build_logger(cfg) as explog:
        for name, physical_cfg, control_cfg in conditions:
            # Rebuild BOTH the dataset and the pipeline per condition: the
            # control bridge holds per-run RNG state, so reusing one across
            # conditions would make them depend on evaluation order.
            control_bridge = common.build_control_bridge(cfg, overrides=control_cfg)
            pipeline, spec = common.build_pipeline(cfg, control_faults=control_bridge)
            evaluator = common.build_evaluator(cfg, pipeline)

            bridge = common.build_bridge(cfg, overrides=physical_cfg)
            dataset = common.build_dataset(cfg, spec, adapter, bridge)

            logger.info(
                "condition %-28s physical=%s control=%s",
                name, not dataset.is_clean,
                control_bridge is not None and not control_bridge.is_clean,
            )
            result = evaluator.run(dataset, max_frames=max_frames, taps=taps)

            row = {"condition": name, **result.as_dict()}
            rows.append(row)
            explog.log_fault_records(result.fault_records)
            explog.log_fault_records(result.control_fault_records)
            explog.log_fault_statistics(
                {"condition": name, "n_faults": result.n_faults,
                 "n_control_faults": result.n_control_faults,
                 "n_frames": result.n_frames}
            )
            for tag, value in result.as_dict().items():
                if isinstance(value, (int, float)):
                    explog.scalar(f"{name}/{tag}", float(value), step=0)

            logger.info(
                "  ap50=%.4f  bits=%.0f  latency=%.1fms  orphan=%.3f  "
                "conflicts=%.0f  faults=%d/%d",
                row.get("ap50", float("nan")),
                row.get("comm_bits_total_sum", float("nan")),
                row.get("latency_t_total_ms_mean", float("nan")),
                row.get("coverage_orphan_rate_mean", float("nan")),
                row.get("schedule_conflicts_total", 0.0),
                result.n_faults, result.n_control_faults,
            )

        _write_tables(explog.dir, cfg, rows, control_tap)

    return {"conditions": rows}


def _write_tables(out_dir: Path, cfg: Dict[str, Any], rows: List[Dict[str, Any]],
                  control_tap) -> None:
    """metrics.csv / metrics.json / control_plane.csv, with a column union."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with (out_dir / "metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2, default=str))

    if control_tap is not None and control_tap.records:
        control_tap.to_csv(out_dir / "control_plane.csv")

    logger.info("wrote %d conditions to %s", len(rows), out_dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LGCP-Bench: clean run plus fault sweep.",
        epilog="Config overrides are positional, e.g. faults=pose_error seed=7",
    )
    parser.add_argument("overrides", nargs="*", default=[])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    # parse_known_args so `key=value` overrides may appear anywhere on the
    # command line, not only before the first flag. argparse cannot interleave
    # positionals with optionals, and requiring a fixed order is a trap.
    args, extra = parser.parse_known_args(argv)

    overrides = list(args.overrides)
    for token in extra:
        if "=" in token and not token.startswith("-"):
            overrides.append(token)
        else:
            parser.error(f"unrecognized argument: {token}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    cfg = common.load(overrides, args.config)
    run(cfg, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
