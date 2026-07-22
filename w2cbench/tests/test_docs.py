"""
Tests that the shipped documentation and job scripts stay true.

A README that names a config group which no longer exists, or an sbatch array
whose ``--array`` range has drifted from its ``FAULTS`` list, fails in the one
place nobody is watching: on a cluster, hours after submission, for a user who
did not write either file. Both are cheap to check and neither is checked by
anything else.

The numeric claims are pinned too. The README warns that an untrained model
shows no compression because the focal prior sits just above the released
threshold; if either constant ever moved, that warning would become a
confusing falsehood rather than a useful one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

PACKAGE = Path(__file__).resolve().parent.parent
CONFIGS = PACKAGE / "configs"
README = PACKAGE / "README.md"
SLURM = PACKAGE / "slurm"


def _groups(group: str) -> set:
    return {p.stem for p in (CONFIGS / group).glob("*.yaml")}


# ------------------------------------------------------------------ README --

def test_every_config_group_the_readme_lists_exists() -> None:
    """The README's configuration table is the first thing anyone copies from."""
    text = README.read_text()
    listed = {}
    for group in ("model", "dataset", "faults", "taps", "trainer"):
        match = re.search(rf"^\s*{group}/\s+(.+?)(?=\n\s*\w+/|\n```)", text,
                          re.MULTILINE | re.DOTALL)
        assert match, f"the README no longer lists the {group} group"
        listed[group] = {name.strip().rstrip(",")
                         for name in re.split(r"[,\s]+", match.group(1))
                         if name.strip().rstrip(",")}

    for group, names in listed.items():
        missing = sorted(names - _groups(group))
        assert not missing, f"README lists {group} groups that do not exist: {missing}"


def test_every_shipped_group_is_mentioned_in_the_readme() -> None:
    """The other direction: an undocumented config group is one nobody runs."""
    text = README.read_text()
    for group in ("model", "dataset", "faults", "taps", "trainer"):
        for name in _groups(group):
            assert name in text, f"{group}={name} is shipped but undocumented"


def test_the_readme_quotes_the_focal_prior_correctly() -> None:
    """The saturation warning rests on two constants being adjacent. If either
    moved, the warning would become a confusing falsehood."""
    from cpbench.models import DetectionHead

    prior = float(torch.sigmoid(DetectionHead(in_channels=8).cls_head.bias[0]))
    assert f"{prior:.6f}" == "0.010051"

    text = README.read_text()
    assert "0.010051" in text
    assert "sigmoid(−4.59)" in text or "sigmoid(-4.59)" in text

    import yaml
    model = yaml.safe_load(
        (CONFIGS / "model" / "where2comm_lidar.yaml").read_text())
    threshold = float(model["communication"]["threshold"])
    assert threshold == 0.01
    assert prior > threshold, (
        "the focal prior no longer sits above the released threshold; the "
        "README's saturation warning needs rewriting")


def test_the_readme_states_the_protocol_family_needs_multiple_rounds() -> None:
    """Running it at K=1 reports perfect robustness by construction, which is
    the kind of result that gets believed."""
    text = README.read_text()
    assert "no-op at K=1" in text
    assert "rounds=3" in text


def test_the_readme_states_the_camera_caveat() -> None:
    """A14: there is no released camera model, so camera numbers must never be
    read as a reproduction."""
    text = README.read_text()
    assert "no released Where2comm camera model" in text
    assert "internal comparisons" in text


# ------------------------------------------------------------------- slurm --

@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")),
                         ids=lambda p: p.name)
def test_the_array_range_matches_the_fault_list(script: Path) -> None:
    """The classic desync: adding a fault family to FAULTS without widening
    --array silently drops it, and removing one makes the last task index an
    unbound variable."""
    text = script.read_text()
    faults = re.search(r"^FAULTS=\((.+?)\)", text, re.MULTILINE)
    array = re.search(r"^#SBATCH --array=(\d+)-(\d+)", text, re.MULTILINE)
    if faults is None:
        pytest.skip(f"{script.name} is not an array job")
    assert array, f"{script.name} defines FAULTS but no --array"

    names = faults.group(1).split()
    low, high = int(array.group(1)), int(array.group(2))
    assert low == 0
    assert high == len(names) - 1, (
        f"{script.name}: --array=0-{high} but FAULTS has {len(names)} entries")


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")),
                         ids=lambda p: p.name)
def test_every_fault_family_a_job_script_names_exists(script: Path) -> None:
    text = script.read_text()
    faults = re.search(r"^FAULTS=\((.+?)\)", text, re.MULTILINE)
    if faults is None:
        pytest.skip(f"{script.name} is not an array job")
    missing = sorted(set(faults.group(1).split()) - _groups("faults"))
    assert not missing, f"{script.name} names missing fault groups: {missing}"


def test_the_benchmark_array_forces_multiple_rounds_for_the_protocol_family() -> None:
    """Without it that array task reports perfect robustness by construction,
    and a job script is exactly where nobody re-reads the caveat."""
    text = (SLURM / "benchmark_array.sbatch").read_text()
    assert 'FAULT" = "protocol"' in text
    assert "model.communication.rounds=3" in text


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")),
                         ids=lambda p: p.name)
def test_job_scripts_export_the_deterministic_cublas_setting(script: Path) -> None:
    """torch.use_deterministic_algorithms raises at the first matmul without
    it, so a deterministic run dies immediately on the cluster."""
    assert "CUBLAS_WORKSPACE_CONFIG" in script.read_text()


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")),
                         ids=lambda p: p.name)
def test_job_scripts_use_the_permitted_partitions(script: Path) -> None:
    """UT EEMCS: ps,main-gpu / ps,main-cpu. A wrong partition queues forever."""
    match = re.search(r"^#SBATCH --partition=(\S+)", script.read_text(),
                      re.MULTILINE)
    assert match, f"{script.name} names no partition"
    assert match.group(1) in ("ps,main-gpu", "ps,main-cpu", "ps,main")


@pytest.mark.parametrize("script", sorted(SLURM.glob("*.sbatch")),
                         ids=lambda p: p.name)
def test_job_scripts_pass_overrides_through(script: Path) -> None:
    """Every script must forward "$@", or a command copied from the README
    would silently run the defaults instead."""
    assert '"$@"' in script.read_text(), f'{script.name} drops "$@"'


def test_the_slurm_readme_repeats_both_hazards() -> None:
    """Somebody reading only the job-script README must still learn that an
    untrained model shows no compression and that the protocol family needs
    K>1."""
    text = (SLURM / "README.md").read_text()
    assert "0.010051" in text
    assert "rounds=3" in text
