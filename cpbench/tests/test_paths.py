"""
Tests for the dataset path resolver.

Two things are worth pinning here, and they fail in different ways.

The precedence rules are ordinary logic and are tested as such. The second
test is the one that matters: it asserts that no config file anywhere in the
repository contains an absolute dataset path again. That is the regression
this module exists to prevent -- a literal re-introduced into one of sixteen
config files is invisible in review, correct on the machine of whoever added
it, and surfaces as a FileNotFoundError on a cluster hours after submission.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from cpbench.utils.config import load_config
from cpbench.utils.paths import (DEFAULT_DATA_ROOT, ENV_VAR, data_root,
                                 dataset_root, describe_source,
                                 require_dataset_root)

REPO = pathlib.Path(__file__).resolve().parents[2]
PACKAGES = ("corabench", "lgcpbench", "cobevtbench", "w2cbench", "v2xvitbench")


# ------------------------------------------------------------ precedence --

def test_environment_outranks_the_config_key() -> None:
    """The config key is checked into git and shared across machines; the
    environment is the machine describing itself. The machine wins."""
    assert data_root("/from/config", env={ENV_VAR: "/from/env"}) == \
        pathlib.Path("/from/env")


def test_config_key_is_used_when_the_environment_is_unset() -> None:
    assert data_root("/from/config", env={}) == pathlib.Path("/from/config")


def test_falls_back_to_the_builtin_default() -> None:
    assert data_root(None, env={}) == pathlib.Path(DEFAULT_DATA_ROOT)


def test_an_empty_environment_variable_does_not_win() -> None:
    """`export CPBENCH_DATA_ROOT=` in a job script must not resolve every
    dataset to the filesystem root."""
    assert data_root("/from/config", env={ENV_VAR: ""}) == \
        pathlib.Path("/from/config")


def test_a_cli_override_beats_the_environment(monkeypatch, tmp_path) -> None:
    """`data_root=` typed at the point of use is never silently ignored --
    otherwise a stale variable in a login shell would quietly redirect a run
    whose command line says otherwise."""
    monkeypatch.setenv(ENV_VAR, "/from/env")
    cfg = load_config(REPO / "corabench" / "configs" / "config.yaml",
                      [f"data_root={tmp_path}"])
    assert cfg["data_root"] == str(tmp_path)
    assert cfg["dataset"]["root"] in (None, "None") or \
        str(tmp_path) in str(cfg["dataset"]["root"])


def test_the_resolved_root_is_recorded_in_the_config(monkeypatch) -> None:
    """The results bundle writes the resolved config, so a finished run must
    carry the absolute path it actually read -- not an unexpanded reference."""
    monkeypatch.setenv(ENV_VAR, "/mnt/staged")
    cfg = load_config(REPO / "corabench" / "configs" / "config.yaml",
                      ["dataset=opv2v"])
    assert cfg["data_root"] == "/mnt/staged"
    assert cfg["dataset"]["root"] == "/mnt/staged/opencood/opv2v"
    assert "${" not in str(cfg["dataset"]["root"])


def test_dataset_root_composes_the_known_layout() -> None:
    assert dataset_root("v2xset", env={ENV_VAR: "/d"}) == \
        pathlib.Path("/d/opencood/v2xset")


def test_an_unknown_adapter_names_the_registry() -> None:
    with pytest.raises(KeyError, match="RELATIVE"):
        dataset_root("nuscenes", env={})


# -------------------------------------------------------- error messages --

def test_the_missing_root_error_names_the_environment_variable(tmp_path) -> None:
    """The reader is on a machine whose layout this repo cannot know, so the
    message must say what to set -- not quote a path from someone else's
    filesystem."""
    with pytest.raises(FileNotFoundError) as excinfo:
        require_dataset_root(tmp_path / "absent")
    message = str(excinfo.value)
    assert ENV_VAR in message
    assert "dataset.root=" in message


def test_the_error_says_which_rule_produced_the_path() -> None:
    assert describe_source(env={ENV_VAR: "/e"}) == f"${ENV_VAR}=/e"
    assert "config key" in describe_source("/c", env={})
    assert "default" in describe_source(None, env={})


# ------------------------------------------------- no literals, anywhere --

# Matches an absolute path that looks like a dataset location. Deliberately
# narrow: `/local`, `/home` and `/tmp` are job scratch and results, not data.
_LITERAL = re.compile(r"^\s*root:\s*(/\S+)", re.MULTILINE)


def _dataset_configs():
    for package in PACKAGES:
        yield from sorted((REPO / package / "configs" / "dataset").glob("*.yaml"))


def test_no_dataset_config_contains_an_absolute_path() -> None:
    """Every `root:` must go through ${data_root}. This is the invariant the
    resolver buys; without the test it lasts until the next config added in a
    hurry on a machine where the literal happened to be right."""
    offenders = []
    for path in _dataset_configs():
        for match in _LITERAL.finditer(path.read_text()):
            offenders.append(f"{path.relative_to(REPO)}: root: {match.group(1)}")
    assert not offenders, (
        "dataset configs must derive root from ${data_root}, not hardcode it:\n  "
        + "\n  ".join(offenders))


def test_every_real_dataset_config_references_the_resolver() -> None:
    """The complement of the test above: a config with no `root` at all is
    fine (synthetic), but one with a root must reference ${data_root}, so a
    future literal cannot hide behind a relative path."""
    offenders = []
    for path in _dataset_configs():
        root = (yaml.safe_load(path.read_text()) or {}).get("root")
        if root is None:
            continue
        if "${data_root}" not in str(root):
            offenders.append(f"{path.relative_to(REPO)}: root: {root}")
    assert not offenders, (
        "these declare a root that does not derive from the resolver:\n  "
        + "\n  ".join(offenders))


def test_every_package_declares_the_data_root_key() -> None:
    """`${data_root}` in a group file resolves against the root config. A
    package missing the key fails at interpolation with a bare KeyError."""
    missing = []
    for package in PACKAGES:
        cfg = yaml.safe_load((REPO / package / "configs" / "config.yaml").read_text())
        if "data_root" not in cfg:
            missing.append(package)
    assert not missing, f"packages without a data_root key: {missing}"
