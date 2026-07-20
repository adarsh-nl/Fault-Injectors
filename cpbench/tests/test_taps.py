"""Tap mechanism: the read-only guarantee, routing, and recorders.

These test the measurement plane itself, which is paper-agnostic. Each paper
package tests its own location registry and its own model's tap-invariance.
"""

import numpy as np
import torch

from cpbench.observation import NullTap, StatsTap, TapSet, TensorDumpTap, emit


def test_emit_detaches():
    """Observation must not create autograd edges, or a tap could reroute
    gradients through the measurement path."""
    class Grab:
        def observe(self, tensor, *, module, location, **ctx):
            self.t = tensor

    grab = Grab()
    x = torch.rand(3, requires_grad=True) * 2
    emit(grab, x, module="M", location="lc/gate")
    assert grab.t.requires_grad is False


def test_null_tap_and_none_are_noops():
    emit(None, torch.ones(2), module="M", location="lc/gate")
    NullTap().observe(torch.ones(2), module="M", location="lc/gate")


def test_tapset_include_filter():
    stats = StatsTap()
    ts = TapSet([stats], include=["lc/*"])
    emit(ts, torch.ones(2), module="M", location="lc/gate")
    emit(ts, torch.ones(2), module="M", location="pac/attention_map")
    assert [r.location for r in stats.records] == ["lc/gate"]


def test_stats_tap_records_and_csv(tmp_path):
    stats = StatsTap()
    emit(TapSet([stats]), torch.zeros(4, 4), module="M",
         location="encoder/bev_features", frame=3)
    rec = stats.records[0]
    assert rec.stats["sparsity"] == 1.0 and rec.stats["n_nan"] == 0
    path = stats.to_csv(tmp_path / "taps.csv")
    assert path.exists() and "location" in path.read_text()


def test_dump_tap_writes_npz(tmp_path):
    dump = TensorDumpTap(tmp_path, every_n=1)
    emit(TapSet([dump]), torch.ones(2, 3), module="M",
         location="lc/output", frame=0, agent_id="cav1")
    files = list(tmp_path.rglob("*.npz"))
    assert len(files) == 1
    assert np.load(files[0])["tensor"].shape == (2, 3)


def test_stats_tap_ignores_non_tensors():
    """Documented behaviour, and the reason paper packages that observe
    DECISIONS (groups, schedules) need their own recorder."""
    stats = StatsTap()
    emit(TapSet([stats]), {"a": 1.0}, module="M", location="lgcp/selection/loads")
    assert stats.records == []
