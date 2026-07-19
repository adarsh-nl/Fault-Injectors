"""ExperimentLogger + config loader round trips."""

import csv
import json
from pathlib import Path

from corabench.logbook.experiment import ExperimentLogger
from corabench.logbook.schema import EvalRecord, ExperimentMeta, TrainRecord
from corabench.utils.config import load_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _meta():
    return ExperimentMeta(
        experiment_id="t1", experiment_name="t", paper="p", architecture="a",
        dataset="d", seed=1, deterministic=True,
        resolved_config={"seed": 1})


def test_logger_round_trip(tmp_path):
    with ExperimentLogger(tmp_path, "exp", _meta()) as log:
        log.log_train(TrainRecord(epoch=0, batch=0, loss_total=1.5))
        log.log_eval(EvalRecord(epoch=0, dataset="d", split="val",
                                condition={"fault": "clean"},
                                detection={"ap50": 0.5, "ap70": 0.3}))
        log.log_fault_statistics({"condition": "clean", "ap70": 0.3})
    d = tmp_path / "exp"
    for name in ("meta.json", "config.yaml", "metrics.csv", "metrics.json",
                 "training.log", "fault_statistics.csv"):
        assert (d / name).exists(), name
    rows = list(csv.DictReader((d / "metrics.csv").open()))
    assert {"train", "eval"} == {r["phase"] for r in rows}
    eval_row = next(r for r in rows if r["phase"] == "eval")
    assert float(eval_row["det_ap50"]) == 0.5
    summary = json.loads((d / "metrics.json").read_text())
    assert summary["eval"][0]["det_ap70"] == 0.3


def test_config_compose_override_interpolate():
    cfg = load_config(CONFIG_ROOT, overrides=[
        "faults=pose_error", "model.cit.strategy=maxout", "seed=7"])
    assert cfg["seed"] == 7
    assert cfg["model"]["cit"]["strategy"] == "maxout"
    assert cfg["faults"]["pipeline"]["pose_error"]["sigma_xy"] == 0.4
    assert cfg["experiment_name"] == "cora_synthetic_pose_error"
    assert len(cfg["faults"]["sweep"]) == 4
