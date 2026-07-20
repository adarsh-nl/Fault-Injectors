"""
Tests for the config tree and CLI entry points.

The requirement is that nothing needs a source edit to change an experiment.
These tests exercise that literally: every shipped config group is composed,
and a full benchmark runs from the command-line interface into a results tree.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lgcpbench.scripts import common
from lgcpbench.scripts.benchmark import expand_sweep
from lgcpbench.scripts.benchmark import run as run_benchmark

CONFIG_DIR = Path(common.CONFIG_ROOT).parent


def _cfg(*overrides: str, tmp_path: Path = None):
    cfg = common.load(list(overrides))
    if tmp_path is not None:
        cfg["results_dir"] = str(tmp_path)
    return cfg


# --------------------------------------------------------------------- #
# config composition
# --------------------------------------------------------------------- #


def test_default_config_composes() -> None:
    cfg = common.load()
    assert cfg["paper"] == "arXiv:2601.12749v1"
    assert cfg["model"]["name"] == "native"
    assert cfg["dataset"]["name"] == "synthetic"
    assert cfg["faults"]["name"] == "none"


def test_experiment_name_interpolates() -> None:
    cfg = common.load(["faults=pose_error"])
    assert cfg["experiment_name"] == "native_synthetic_pose_error"


@pytest.mark.parametrize(
    "group,name",
    [(p.parent.name, p.stem) for p in sorted(CONFIG_DIR.glob("*/*.yaml"))],
)
def test_every_shipped_config_group_composes(group: str, name: str) -> None:
    """A config file that cannot be selected is dead weight, and a broken one
    fails at job-submission time on the cluster rather than here."""
    cfg = common.load([f"{group}={name}"])
    assert cfg[group]["name"] == name


def test_leaf_override_applies() -> None:
    cfg = common.load(["lgcp.confidence.delta_g=0.25", "seed=7"])
    assert cfg["lgcp"]["confidence"]["delta_g"] == 0.25
    assert cfg["seed"] == 7


def test_assumptions_are_recorded_in_config() -> None:
    """Every ambiguity resolution travels with the run (design doc 1.4)."""
    cfg = common.load()
    assumptions = cfg["assumptions"]
    for key in ("B1", "B2", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "AUROC"):
        assert key in assumptions
    assert "inapplicable" in assumptions["AUROC"]


def test_paper_constants_are_in_config_not_source() -> None:
    """Table I and section VI-C values must be overridable without a code edit."""
    cfg = common.load()
    n = cfg["network"]
    assert n["subchannels_Z"] == 5
    assert n["tx_power_dbm"] == 23.0
    assert n["noise_power_dbm"] == -114.0
    assert n["path_loss_intercept_db"] == 128.1
    assert n["shadowing_std_db"] == 8.0
    assert n["time_slot_s"] == pytest.approx(2.5e-4)
    assert cfg["lgcp"]["confidence"]["delta_g"] == 0.075
    assert cfg["lgcp"]["latency"]["deadline_T_ms"] == 100.0


def test_paper_model_costs_match_section_vi_c() -> None:
    for name, mflops in (("where2comm", 1400), ("cobevt", 2228), ("coalign", 2684)):
        assert common.load([f"model={name}"])["model"]["mflops"] == mflops


# --------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------- #


def test_build_pipeline_from_default_config() -> None:
    cfg = common.load()
    pipeline, spec = common.build_pipeline(cfg)
    assert pipeline.backbone.feature_hw == spec.feature_hw
    assert len(pipeline.rsu.grid) > 0


def test_grid_divisibility_is_checked_with_an_actionable_message() -> None:
    """A bad point_range must fail at build time, not as an opaque shape
    error inside the backbone twenty minutes into a cluster job."""
    cfg = common.load(["dataset.grid.point_range=[-40.0,-12.0,-3.0,40.0,12.0,1.0]"])
    with pytest.raises(ValueError, match="must divide by"):
        common.build_grid_spec(cfg)


def test_unknown_backend_and_policy_are_rejected() -> None:
    cfg = common.load()
    spec = common.build_grid_spec(cfg)
    cfg["model"]["backend"] = "nope"
    with pytest.raises(KeyError):
        common.build_backbone(cfg, spec, common.resolve_device(cfg))

    cfg = common.load()
    cfg["lgcp"]["selection"]["leader_policy"] = "nope"
    with pytest.raises(KeyError):
        common.build_rsu(cfg, spec, common.build_area_grid(cfg, spec), 0)


def test_opencood_backend_fails_with_actionable_guidance() -> None:
    """The core deliberately does not depend on OpenCOOD's py3.7 environment,
    so selecting it without that environment must say so plainly."""
    cfg = common.load(["model=where2comm"])
    spec = common.build_grid_spec(cfg)
    with pytest.raises(ImportError, match="Python 3.7"):
        common.build_backbone(cfg, spec, common.resolve_device(cfg))


def test_clean_faults_config_builds_no_bridge() -> None:
    assert common.build_bridge(common.load()) is None


def test_fault_config_builds_a_bridge() -> None:
    bridge = common.build_bridge(common.load(["faults=pose_error"]))
    assert bridge is not None and not bridge.is_clean


def test_explicit_empty_override_forces_a_clean_bridge() -> None:
    """The reference condition must be clean even when faults are configured.
    A contaminated reference invalidates every comparison drawn against it."""
    cfg = common.load(["faults=pose_error"])
    assert common.build_bridge(cfg, overrides={}) is None
    assert common.build_bridge(cfg, overrides=None) is not None


# --------------------------------------------------------------------- #
# sweep expansion
# --------------------------------------------------------------------- #


def test_expand_sweep_names_conditions() -> None:
    got = expand_sweep([{"pose_error": {"sigma_xy": [0.2, 0.4]}}])
    assert got == [
        ("pose_error_sxy0.2", {"pose_error": {"sigma_xy": 0.2}}),
        ("pose_error_sxy0.4", {"pose_error": {"sigma_xy": 0.4}}),
    ]


def test_expand_sweep_handles_empty_and_scalar() -> None:
    assert expand_sweep([]) == []
    assert expand_sweep(None) == []
    assert expand_sweep([{"agent_drop": {"p_drop": 0.5}}]) == [
        ("agent_drop_p0.5", {"agent_drop": {"p_drop": 0.5}})
    ]


def test_shipped_sweeps_expand() -> None:
    for name, expected in (("pose_error", 8), ("agent_drop", 4), ("comm_stress", 6)):
        cfg = common.load([f"faults={name}"])
        assert len(expand_sweep(cfg["faults"]["sweep"])) == expected


# --------------------------------------------------------------------- #
# end-to-end CLI
# --------------------------------------------------------------------- #


def test_benchmark_writes_a_complete_results_tree(tmp_path: Path) -> None:
    cfg = _cfg("faults=agent_drop", "taps=stats",
               "lgcp.confidence.delta_g=0.005", "dataset.n_frames=4",
               tmp_path=tmp_path)
    run_benchmark(cfg, max_frames=3)

    out = tmp_path / cfg["experiment_name"]
    for filename in (
        "config.yaml", "meta.json", "metrics.csv", "metrics.json",
        "fault_statistics.csv", "control_plane.csv", "training.log",
    ):
        assert (out / filename).exists(), filename

    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert rows[0]["condition"] == "clean"
    assert {r["condition"] for r in rows} >= {"clean", "agent_drop_p0.5"}
    assert json.loads((out / "metrics.json").read_text())


def test_clean_condition_is_provably_clean(tmp_path: Path) -> None:
    cfg = _cfg("faults=agent_drop", "lgcp.confidence.delta_g=0.005",
               "dataset.n_frames=3", tmp_path=tmp_path)
    result = run_benchmark(cfg, max_frames=3)
    clean = next(r for r in result["conditions"] if r["condition"] == "clean")
    assert clean["n_injected_faults"] == 0


def test_stronger_agent_drop_reduces_transmitted_volume(tmp_path: Path) -> None:
    """The sweep must produce a monotonic trend, not just distinct numbers.

    Dropping more collaborators shrinks groups, so fewer packets carry fewer
    bits. If this inverted, the fault would not be reaching the control plane.
    """
    cfg = _cfg("faults=agent_drop", "lgcp.confidence.delta_g=0.005",
               "dataset.n_frames=6", tmp_path=tmp_path)
    rows = {r["condition"]: r for r in run_benchmark(cfg, max_frames=6)["conditions"]}

    volumes = [
        rows[f"agent_drop_p{p}"]["comm_bits_total_sum"]
        for p in (0.1, 0.25, 0.5, 0.75)
    ]
    assert volumes == sorted(volumes, reverse=True), volumes
    assert rows["clean"]["comm_bits_total_sum"] > volumes[-1]


def test_every_sweep_condition_actually_injected(tmp_path: Path) -> None:
    """A fault condition that injected nothing would report 'no degradation'
    from an experiment that never ran."""
    cfg = _cfg("faults=pose_error", "lgcp.confidence.delta_g=0.005",
               "dataset.n_frames=3", tmp_path=tmp_path)
    for row in run_benchmark(cfg, max_frames=3)["conditions"]:
        if row["condition"] == "clean":
            continue
        assert row["n_injected_faults"] > 0, row["condition"]


def test_delta_g_sweep_reproduces_the_fig3_trend(tmp_path: Path) -> None:
    """Larger dg -> smaller groups -> less transmitted data (Fig. 3)."""
    volumes = []
    for delta_g in (0.005, 0.05, 0.5):
        cfg = _cfg(f"lgcp.confidence.delta_g={delta_g}", "dataset.n_frames=4",
                   tmp_path=tmp_path / str(delta_g))
        rows = run_benchmark(cfg, max_frames=4)["conditions"]
        volumes.append(rows[0]["comm_bits_total_sum"])
    assert volumes == sorted(volumes, reverse=True), volumes
