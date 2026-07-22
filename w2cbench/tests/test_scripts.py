"""
Tests for the config groups and the three CLI entry points.

Every shipped YAML is loaded and composed here, because a config that does not
parse is indistinguishable from one that does until a cluster job starts -- and
a config that parses but names a tap location that does not exist is worse,
since that job finishes cleanly and writes an empty ``taps.csv``.

The CLIs are exercised in-process rather than by subprocess: the point is that
the wiring is right, and a subprocess would mostly be testing the interpreter's
startup time.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from w2cbench.comm import BudgetSelector, ThresholdSelector, TopKSelector
from w2cbench.fusion import AttenFusion, TransformerFusion
from w2cbench.scripts import benchmark as benchmark_cli
from w2cbench.scripts import common
from w2cbench.scripts import evaluate as evaluate_cli
from w2cbench.scripts import train as train_cli

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _cfg(overrides=None, tmp_path=None):
    cfg = common.load(list(overrides or []) + ["dataset.n_frames=4"])
    if tmp_path is not None:
        cfg["results_dir"] = str(tmp_path)
    return cfg


# ------------------------------------------------------------------ configs --

@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")),
                         ids=lambda p: str(p.name))
def test_every_shipped_yaml_parses(path: Path) -> None:
    with path.open() as handle:
        assert yaml.safe_load(handle) is not None, f"{path} is empty"


@pytest.mark.parametrize("group,name", [
    ("model", p.stem) for p in sorted((CONFIGS / "model").glob("*.yaml"))
] + [
    ("faults", p.stem) for p in sorted((CONFIGS / "faults").glob("*.yaml"))
] + [
    ("taps", p.stem) for p in sorted((CONFIGS / "taps").glob("*.yaml"))
] + [
    ("trainer", p.stem) for p in sorted((CONFIGS / "trainer").glob("*.yaml"))
])
def test_every_group_composes_and_validates(group: str, name: str) -> None:
    """Including the eager validation, so a typo in a taps include list fails
    here rather than after a six-hour job.

    A model group is paired with a dataset of its own track: the camera model
    against the LiDAR default is a *correct* validation failure, not a broken
    config, and pairing them here keeps this test about composition rather
    than re-testing the cross-check.
    """
    overrides = [f"{group}={name}"]
    if group == "model" and "camera" in name:
        overrides.append("dataset=synthetic_camera")
    cfg = common.load(overrides)
    assert cfg[group]["name"]


def test_the_camera_model_against_a_lidar_dataset_is_rejected() -> None:
    """The pairing the test above sidesteps -- asserted directly, because
    composing without error and then failing with an opaque shape mismatch
    deep in the encoder is exactly what the cross-check exists to prevent."""
    with pytest.raises(ValueError, match="disagree"):
        common.load(["model=where2comm_camera", "dataset=synthetic_lidar"])


def test_dataset_groups_compose() -> None:
    """Real-data groups are composed but not opened: the paths point at
    /deepstore and only exist on the cluster."""
    for name in ("synthetic_lidar", "opv2v_lidar", "v2xset", "dair_v2x"):
        cfg = common.load([f"dataset={name}"])
        assert cfg["dataset"]["track"] == "lidar"
        assert common.build_grid_spec(cfg).feature_hw[0] > 0


def test_the_experiment_name_interpolates_from_its_parts() -> None:
    cfg = common.load(["faults=pose_error"])
    assert cfg["experiment_name"] == "where2comm_lidar_synthetic_lidar_pose_error"


def test_grid_downsample_and_block_stride_agree_in_every_dataset() -> None:
    """The trap from step 3: the backbone produces grid_hw // block_strides[0]
    while anchors are sized from grid.downsample, and a mismatch lowers AP
    without ever failing. Checked across every shipped pairing."""
    for name in ("synthetic_lidar", "opv2v_lidar", "v2xset", "dair_v2x"):
        cfg = common.load([f"dataset={name}"])
        assert (int(cfg["dataset"]["grid"]["downsample"])
                == int(cfg["model"]["encoder"]["block_strides"][0]))
        common.build_encoder(cfg)          # raises if they disagree


# --------------------------------------------------------------- validation --

def test_a_bad_tap_location_fails_at_load_time() -> None:
    with pytest.raises(KeyError, match="unknown observation location"):
        common.load(["taps=stats", "taps.stats.include=['fusion/r0/attn_weights']"])


def test_a_selector_missing_its_argument_fails_at_load_time() -> None:
    with pytest.raises(ValueError, match="needs model.communication.topk"):
        common.load(["model.communication.selector=topk"])
    with pytest.raises(ValueError, match="budget_bytes"):
        common.load(["model.communication.selector=budget"])


def test_a_track_mismatch_names_both_sides() -> None:
    with pytest.raises(ValueError, match="disagree"):
        common.load(["model.track=camera"])


# ------------------------------------------------------------------ builders --

def test_overrides_reach_the_built_model() -> None:
    """The brief's requirement that nothing needs a source edit, checked at
    the only place it can be: the objects a config actually produces."""
    model = common.build_model(common.load(["model.communication.rounds=3"]))
    assert model.rounds == 3

    model = common.build_model(common.load(
        ["model.communication.selector=budget",
         "model.communication.budget_bytes=16384"]))
    assert isinstance(model.selector, BudgetSelector)
    assert model.selector.bytes_per_cell == 256 * 4 + 4

    model = common.build_model(common.load(
        ["model.communication.selector=topk", "model.communication.topk=128"]))
    assert isinstance(model.selector, TopKSelector) and model.selector.k == 128


def test_the_transformer_model_group_swaps_the_fusion_module() -> None:
    assert isinstance(
        common.build_model(common.load([])).aggregator, AttenFusion)
    assert isinstance(
        common.build_model(
            common.load(["model=where2comm_lidar_transformer"])).aggregator,
        TransformerFusion)


def test_smoothing_can_be_disabled_from_config() -> None:
    off = common.build_model(
        common.load(["model.confidence.gaussian_smooth.enabled=false"]))
    assert off.confidence.smoother is None
    assert common.build_model(common.load([])).confidence.smoother is not None


def test_the_curriculum_range_is_configurable() -> None:
    model = common.build_model(
        common.load(["model.communication.train_bandwidth=[0.5, 0.5]"]))
    assert model.selector.train_bandwidth == (0.5, 0.5)


def test_taps_groups_build_the_right_sinks(tmp_path) -> None:
    assert common.build_taps(common.load(["taps=none"]), tmp_path) == (None, None)
    taps, stats = common.build_taps(common.load(["taps=stats"]), tmp_path)
    assert taps is not None and stats is not None
    taps, _ = common.build_taps(common.load(["taps=attention"]), tmp_path)
    assert len(taps.taps) == 2                 # stats + dump


def test_a_clean_fault_group_builds_no_injector() -> None:
    cfg = common.load(["faults=none"])
    assert common.build_bridge_for(cfg).is_clean
    assert common.build_protocol_for(cfg).is_clean


def test_the_protocol_group_arms_only_the_protocol_plane() -> None:
    cfg = common.load(["faults=protocol"])
    assert common.build_bridge_for(cfg).is_clean
    assert not common.build_protocol_for(cfg).is_clean


# ---------------------------------------------------------------- the CLIs --

def test_train_runs_end_to_end(tmp_path) -> None:
    history = train_cli.run(_cfg(["trainer=smoke"], tmp_path), max_batches=2)
    assert history and all(torch.isfinite(torch.tensor(r.loss_total))
                           for r in history)
    run_dir = tmp_path / "where2comm_lidar_synthetic_lidar_clean_train"
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "config.yaml").exists()


def test_the_meta_records_every_assumption(tmp_path) -> None:
    """A result must never be separated from the reading of the paper that
    produced it."""
    train_cli.run(_cfg(["trainer=smoke"], tmp_path), max_batches=1)
    meta = json.loads(
        (tmp_path / "where2comm_lidar_synthetic_lidar_clean_train"
         / "meta.json").read_text())
    assert meta["paper"].startswith("Where2comm")
    assert {"A1", "A2", "A11", "A17", "A18"} <= set(meta["assumptions"])
    assert meta["environment"]["python"]


def test_evaluate_runs_one_condition(tmp_path) -> None:
    results = evaluate_cli.run(_cfg(["faults=pose_error"], tmp_path))
    assert [r.condition.name for r in results] == ["clean", "pose0.4"]


def test_evaluate_can_select_one_sweep_entry(tmp_path) -> None:
    results = evaluate_cli.run(_cfg(["faults=pose_error"], tmp_path),
                               condition="pose0.2")
    assert [r.condition.name for r in results] == ["clean", "pose0.2"]


def test_evaluate_rejects_an_unknown_condition(tmp_path) -> None:
    with pytest.raises(SystemExit, match="no condition named"):
        evaluate_cli.run(_cfg(["faults=pose_error"], tmp_path),
                         condition="pose9.9")


def test_benchmark_writes_ap_and_bytes_in_one_row(tmp_path) -> None:
    benchmark_cli.run(_cfg(["faults=agent_drop", "taps=comm"], tmp_path))
    run_dir = tmp_path / "where2comm_lidar_synthetic_lidar_agent_drop_bench"
    with (run_dir / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5                      # clean + four drop rates
    for row in rows:
        assert row["det_ap50"] != "" and row["comm_log2_bytes"] != ""
    assert (run_dir / "taps.csv").exists()
    assert (run_dir / "injection_summary.csv").exists()


def test_benchmark_crosses_bandwidth_from_config(tmp_path) -> None:
    cfg = _cfg(["faults=pose_error"], tmp_path)
    cfg["faults"]["sweep"] = [{}, {"pipeline": {"pose_error": {"sigma_xy": 0.4}}}]
    cfg["bandwidth_sweep"] = [{"kind": "budget", "budget_bytes": 4096},
                              {"kind": "budget", "budget_bytes": 262144}]
    results = benchmark_cli.run(cfg)
    assert len(results) == 4
    volumes = {r.condition.group: r.result.comms["bytes_per_frame"]
               for r in results}
    assert volumes["bw4096"] < volumes["bw262144"]


def test_the_protocol_family_reaches_the_model(tmp_path) -> None:
    """With rounds=3 the request-loss entries must actually do something --
    the K=1 no-op is a property of the round count, not of the wiring."""
    cfg = _cfg(["faults=protocol", "model.communication.rounds=3"], tmp_path)
    cfg["faults"]["sweep"] = [{}, {"protocol_pipeline":
                                   {"request_loss": {"p_loss": 1.0}}}]
    results = benchmark_cli.run(cfg)
    assert results[1].result.n_faults > 0


def test_main_parses_positional_overrides(tmp_path, monkeypatch) -> None:
    """The CLI shape a README command relies on."""
    captured = {}
    monkeypatch.setattr(train_cli, "run",
                        lambda cfg, max_batches=None: captured.update(cfg=cfg))
    assert train_cli.main(["trainer=smoke", "seed=99",
                           "model.communication.rounds=2"]) == 0
    assert captured["cfg"]["seed"] == 99
    assert captured["cfg"]["trainer"]["name"] == "smoke"
    assert captured["cfg"]["model"]["communication"]["rounds"] == 2


def test_an_unparseable_argument_is_rejected(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        train_cli.main(["--nonsense"])


# ------------------------------------------- the multi-round / selector trap --

def test_multi_round_with_a_threshold_selector_warns(caplog) -> None:
    """Found by measuring, not by reading. Round k>0 selects on C_i (x) R_j and
    R_j <= 1, so its priority is ALWAYS <= round 0's: against a fixed threshold
    a later round can only select a subset, and near the bar, nothing.

    A warning rather than an error because whether it bites depends on the
    checkpoint. A trained model (C ~ 0.9, R ~ 0.9 -> 0.81) is fine; an
    undertrained one at the focal prior is not.
    """
    import logging
    with caplog.at_level(logging.WARNING, logger="w2cbench.scripts.common"):
        common.load(["model.communication.rounds=3"])
    assert any("self" in r.message or "subset of round" in r.message
               for r in caplog.records), caplog.text


def test_multi_round_with_a_budget_selector_is_silent(caplog) -> None:
    """The configuration multi-round is actually well posed under: k cells are
    kept regardless of magnitude, so a later round is not bounded by an
    earlier one."""
    import logging
    with caplog.at_level(logging.WARNING, logger="w2cbench.scripts.common"):
        common.load(["model.communication.rounds=3",
                     "model.communication.selector=budget",
                     "model.communication.budget_bytes=16384"])
    assert not [r for r in caplog.records if "subset of round" in r.message]


def test_the_default_configuration_is_single_round(caplog) -> None:
    """Q5, resolved: K=1 matches the released config AND is the only setting
    for which the released threshold is a coherent choice."""
    import logging
    with caplog.at_level(logging.WARNING, logger="w2cbench.scripts.common"):
        cfg = common.load([])
    assert cfg["model"]["communication"]["rounds"] == 1
    assert cfg["model"]["communication"]["selector"] == "threshold"
    assert not caplog.records
