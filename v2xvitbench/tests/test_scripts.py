"""
Tests for config loading, the builders and eager validation.

Every builder is exercised from the shipped tiny config, because the configs
are part of the deliverable: a YAML that drifts from the builders fails a
cluster job at hour zero of a submission window, or worse, silently runs
with a default the experiment did not ask for.
"""

from __future__ import annotations

import pytest
import torch

from v2xvitbench.scripts import common


def _cfg(*overrides: str):
    return common.load(["model=v2xvit_tiny", "dataset=synthetic_lidar",
                        "trainer=smoke", *overrides])


# ------------------------------------------------------------------ config --

def test_default_config_loads_and_validates() -> None:
    cfg = common.load([])
    assert cfg["model"]["name"] == "v2xvit"
    assert cfg["experiment_name"].startswith("v2xvit_")


def test_group_swap_and_leaf_override() -> None:
    cfg = _cfg("faults=type_flip", "seed=7")
    assert cfg["faults"]["name"] == "type_flip"
    assert cfg["seed"] == 7


def test_every_fault_group_loads(tmp_path) -> None:
    for group in ("none", "pose_error", "latency", "agent_drop",
                  "lidar_weather", "delay_encoding", "type_flip",
                  "correction_matrix", "v2x_noise"):
        cfg = _cfg(f"faults={group}")
        assert cfg["faults"]["name"] == group


def test_every_taps_group_loads() -> None:
    for group in ("none", "stats", "attention"):
        _cfg(f"taps={group}")


def test_paper_model_group_loads_with_v2xset_geometry() -> None:
    cfg = common.load(["model=v2xvit", "dataset=v2xset"])
    grid = common.build_grid_spec(cfg)
    assert grid.grid_hw == (192, 704)
    assert grid.feature_hw == (48, 176)


# -------------------------------------------------------------- validation --

def test_wrong_downsample_fails_by_config_key() -> None:
    with pytest.raises(ValueError, match="dataset.grid.downsample"):
        _cfg("dataset.grid.downsample=2")


def test_unknown_tap_location_fails_with_suggestions() -> None:
    with pytest.raises(KeyError, match="unknown observation location"):
        _cfg("taps=attention",
             "taps.dump.include=[rte/embeddings]")


def test_mismatched_mswin_lists_fail() -> None:
    with pytest.raises(ValueError, match="same length"):
        _cfg("model.fusion.mswin.heads=[2]")


def test_indivisible_window_fails_at_load() -> None:
    with pytest.raises(ValueError, match="MSwin"):
        _cfg("model.fusion.mswin.window_sizes=[2, 5]",
             "model.fusion.mswin.heads=[2, 2]",
             "model.fusion.mswin.dim_heads=[16, 32]")


def test_wrong_relation_count_fails() -> None:
    with pytest.raises(ValueError, match="num_types"):
        _cfg("model.fusion.hmsa.num_relations=3")


# ---------------------------------------------------------------- builders --

def test_build_model_from_tiny_config() -> None:
    cfg = _cfg()
    model = common.build_model(cfg)
    assert model.max_cav == 3
    assert model.grid.feature_hw == (16, 16)


def test_model_and_dataset_drive_each_other() -> None:
    cfg = _cfg()
    model = common.build_model(cfg).eval()
    dataset = common.build_dataset(cfg, split="test")
    batch = common.build_collator(cfg)([dataset[0]])
    out = model(batch)
    assert out["cls"].shape == (1, 2, 16, 16)


def test_build_bridges_clean_override() -> None:
    cfg = _cfg("faults=type_flip")
    assert not common.build_metadata_for(cfg).is_clean
    assert common.build_metadata_for(cfg, {}).is_clean     # {} = explicit clean
    assert common.build_bridge_for(cfg, {}).is_clean


def test_build_loss_decoder_optimizer_scheduler() -> None:
    cfg = _cfg()
    model = common.build_model(cfg)
    loss = common.build_loss(cfg)
    decoder = common.build_decoder(cfg)
    optimizer = common.build_optimizer(cfg, model)
    scheduler = common.build_scheduler(cfg, optimizer)
    assert loss is not None and decoder is not None
    assert scheduler is None                               # smoke: none
    assert common.build_trainer_config(cfg).epochs == 2


def test_build_taps_none_and_stats(tmp_path) -> None:
    assert common.build_taps(_cfg(), tmp_path) == (None, None)
    taps, stats = common.build_taps(_cfg("taps=stats"), tmp_path)
    assert taps is not None and stats is not None


def test_tester_factory_produces_a_runnable_tester() -> None:
    from v2xvitbench.evaluation import Condition
    cfg = _cfg("max_frames=2")
    factory = common.build_tester_factory(cfg, torch.device("cpu"))
    tester = factory(Condition(name="clean", is_clean=True), None)
    result = tester.run(common.build_model(cfg).eval())
    assert result.n_frames == 2
