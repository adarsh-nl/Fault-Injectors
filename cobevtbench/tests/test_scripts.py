"""
Tests for the CLI entry points and config composition.

The load-bearing guarantee here is the brief's: nothing needs a source edit.
So these tests drive the real ``main(argv)`` functions with the same
``key=value`` overrides a user would type, on the synthetic datasets, and
assert a results bundle appears. If a config key drifts from what
``common.py`` reads, this is where it surfaces -- rather than twenty minutes
into a cluster job.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cobevtbench.scripts import benchmark, common, evaluate, train

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

SMOKE_CAMERA = ["dataset=synthetic_camera", "model=cobevt_camera_dynamic",
                "trainer=smoke", "dataset.n_frames=4", "dataset.image_size=[32,32]",
                "dataset.bev.height=32", "dataset.bev.width=32",
                "dataset.max_cav=2",
                "model.sinbevt.dims=[16,16]", "model.sinbevt.bev_size=16",
                "model.sinbevt.bev_meters=40.0",
                "model.sinbevt.q_win_sizes=[8,8]",
                "model.sinbevt.feat_win_sizes=[2,2]",
                "model.sinbevt.heads=[2,2]", "model.sinbevt.dim_head=[8,8]",
                "model.sinbevt.middle=[1,1]",
                "model.sinbevt.bev_embedding_flags=[true,false]",
                "model.sinbevt.self_attn_dim_head=8",
                "model.backbone.arch=resnet18",
                "model.backbone.pretrained=false",
                "model.backbone.id_pick=[1,2]",
                "model.fusebevt.window=4", "model.fusebevt.dim_head=8",
                "model.fusebevt.depth=1", "model.decoder.channels=[4,8]"]

SMOKE_LIDAR = ["dataset=synthetic_lidar", "model=cobevt_lidar",
               "trainer=smoke", "dataset.n_frames=4", "dataset.max_cav=2",
               "model.encoder.out_channels=32", "model.fusebevt.depth=1",
               "model.fusebevt.window=8", "model.fusebevt.dim_head=8"]


# ------------------------------------------------------ config composition --

def test_default_config_composes() -> None:
    cfg = common.load([])
    assert cfg["model"]["track"] == "camera"
    assert cfg["dataset"]["name"] == "synthetic_camera"
    assert cfg["faults"]["name"] == "clean"


@pytest.mark.parametrize("path", sorted((CONFIG_DIR / "model").glob("*.yaml")),
                         ids=lambda p: p.stem)
def test_every_model_config_names_itself(path: Path) -> None:
    """The house rule: a group file's first job is to name itself, so
    composition can assert it landed in the right slot."""
    cfg = yaml.safe_load(path.read_text())
    assert cfg["name"] == path.stem


@pytest.mark.parametrize("group", ["dataset", "faults", "taps", "trainer"])
def test_every_group_file_names_itself(group: str) -> None:
    for path in (CONFIG_DIR / group).glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        assert "name" in cfg, f"{group}/{path.name} has no name"


def test_interpolation_resolves_the_experiment_name() -> None:
    cfg = common.load(["faults=pose_error"])
    assert cfg["experiment_name"] == "cobevt_camera_dynamic_synthetic_camera_pose_error"


def test_assumptions_interpolate_live_config() -> None:
    """meta.json records which side of each ambiguity a run took, so the
    assumption strings must resolve against the actual config, not restate a
    literal."""
    cfg = common.load([])
    assert "layer2/3/4" in cfg["model"]["assumptions"]["A2"]
    assert "[1, 2, 3]" in cfg["model"]["assumptions"]["A2"]


def test_track_mismatch_is_caught_early() -> None:
    """A camera dataset with a lidar model composes fine and then fails deep
    in a forward pass. common.track() names both sides instead."""
    cfg = common.load(["dataset=synthetic_camera", "model=cobevt_lidar"])
    with pytest.raises(ValueError, match="disagree"):
        common.track(cfg)


# --------------------------------------------------------- object builders --

def test_build_model_matches_the_track() -> None:
    from cobevtbench.models.cobevt_camera import CoBEVTCamera
    from cobevtbench.models.cobevt_lidar import CoBEVTLidar
    assert isinstance(common.build_model(common.load(SMOKE_CAMERA)),
                      CoBEVTCamera)
    assert isinstance(common.build_model(common.load(SMOKE_LIDAR)),
                      CoBEVTLidar)


def test_build_bridge_empty_override_is_clean() -> None:
    """The clean reference must go through the explicit-empty path, distinct
    from None which falls back to config."""
    cfg = common.load(["faults=camera_dropout"])
    assert common.build_bridge_for(cfg, overrides={}).is_clean
    assert not common.build_bridge_for(cfg, overrides=None).is_clean


def test_taps_config_locations_are_validated() -> None:
    """A typo in a taps include should fail at build, not produce an empty
    taps.csv after a long run."""
    cfg = common.load(["taps=attention"])
    common.build_taps(cfg, Path("/tmp"))            # every location valid
    cfg["taps"]["dump"]["include"] = ["fusebevt/d0/local/not_a_tensor"]
    with pytest.raises(KeyError, match="unknown observation location"):
        common.build_taps(cfg, Path("/tmp"))


def test_unknown_optimizer_raises() -> None:
    cfg = common.load(["trainer=smoke"])
    cfg["trainer"]["optimizer"] = "lamb"
    with pytest.raises(ValueError, match="unknown optimizer"):
        common.build_optimizer(cfg, common.build_model(common.load(SMOKE_CAMERA)))


# ---------------------------------------------------------- entry points ----

def test_train_camera_end_to_end(tmp_path) -> None:
    code = train.main(SMOKE_CAMERA + [f"results_dir={tmp_path}", "seed=0"])
    assert code == 0
    runs = list(tmp_path.glob("*_train"))
    assert runs, "no results directory was created"
    assert (runs[0] / "metrics.csv").exists()
    assert (runs[0] / "meta.json").exists()
    assert (runs[0] / "training.log").read_text().strip(), "training.log is empty"


def test_train_lidar_end_to_end(tmp_path) -> None:
    assert train.main(SMOKE_LIDAR + [f"results_dir={tmp_path}"]) == 0
    assert list(tmp_path.glob("*_train"))


def test_benchmark_produces_a_condition_table(tmp_path) -> None:
    argv = SMOKE_CAMERA + ["faults=camera_dropout", f"results_dir={tmp_path}",
                           "--max-frames", "2"]
    assert benchmark.main(argv) == 0
    bench = next(iter(tmp_path.glob("*_bench")))
    metrics = (bench / "metrics.csv").read_text()
    # clean plus the four dropout conditions, each a row.
    assert "camdrop4" in metrics and "cond_name" in metrics
    assert (bench / "fault_statistics.csv").exists()


def test_benchmark_records_injected_faults(tmp_path) -> None:
    argv = SMOKE_CAMERA + ["faults=calibration_error", f"results_dir={tmp_path}",
                           "--max-frames", "2"]
    benchmark.main(argv)
    bench = next(iter(tmp_path.glob("*_bench")))
    assert (bench / "injection_summary.csv").exists()


def test_evaluate_single_condition(tmp_path) -> None:
    argv = SMOKE_CAMERA + ["faults=camera_dropout", "--condition", "camdrop4",
                           f"results_dir={tmp_path}", "--max-frames", "2"]
    assert evaluate.main(argv) == 0
    bench = next(iter(tmp_path.glob("*_bench")))
    metrics = (bench / "metrics.csv").read_text()
    assert "camdrop4" in metrics


def test_evaluate_unknown_condition_lists_the_options(tmp_path) -> None:
    argv = SMOKE_CAMERA + ["faults=camera_dropout", "--condition", "camdrop9",
                           f"results_dir={tmp_path}"]
    with pytest.raises(SystemExit, match="camdrop4"):
        evaluate.main(argv)


def test_benchmark_with_taps_writes_tap_stats(tmp_path) -> None:
    argv = SMOKE_CAMERA + ["faults=camera_dropout", "taps=stats",
                           f"results_dir={tmp_path}", "--max-frames", "2"]
    benchmark.main(argv)
    bench = next(iter(tmp_path.glob("*_bench")))
    assert (bench / "taps.csv").exists()
