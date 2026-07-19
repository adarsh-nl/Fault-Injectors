"""Tap protocol: read-only guarantee, routing, recorders."""

import numpy as np
import torch

from corabench.observation import (NullTap, StatsTap, TapSet, TensorDumpTap,
                                   all_locations, emit, validate_location)


def test_emit_detaches():
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


def test_forward_identical_with_and_without_taps(tiny_model, batch):
    tiny_model.eval()
    with torch.no_grad():
        ref = tiny_model(batch)
        tapped = tiny_model(batch, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(ref["f_out"], tapped["f_out"])
    assert torch.equal(ref["probs"]["prob_lc"], tapped["probs"]["prob_lc"])


def test_location_registry():
    locs = all_locations()
    assert "encoder/bev_features" in locs and "fusion/final_boxes" in locs
    assert validate_location("lc/ssm_out").module == "CSSM"
    try:
        validate_location("lc/nope")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
