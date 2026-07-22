"""
Tests for communication-volume metrics and the EvalRecord column they fill.

The point of these metrics is that a bandwidth number and an accuracy number
appear in the same row, so a fault that lowers both is not read as an
efficiency win. What has to be right for that to work is unglamorous: the
aggregation must not double-count rounds, must not average ratios wrongly,
and must not silently invent zeros for models that have no selection step.
"""

from __future__ import annotations

import math

from cpbench.comms.channel import MessageChannel
from cpbench.logbook.schema import EvalRecord
from cpbench.metrics import CommVolumeMetrics, FrameComms


# ----------------------------------------------------------- aggregation --

def test_empty_run_reports_no_frames_rather_than_zero_bytes() -> None:
    """A run that evaluated nothing and a run that transmitted nothing are
    different facts; only the second deserves a byte column."""
    assert CommVolumeMetrics().compute() == {"n_frames": 0.0}


def test_bytes_are_summed_and_averaged_over_frames() -> None:
    m = CommVolumeMetrics()
    m.add_frame(FrameComms(0, {"comm/r0/sent": 1000}, messages=2))
    m.add_frame(FrameComms(1, {"comm/r0/sent": 3000}, messages=6))
    out = m.compute()
    assert out["bytes_total"] == 4000.0
    assert out["bytes_per_frame"] == 2000.0
    assert out["n_messages"] == 8.0
    assert out["messages_per_frame"] == 4.0
    assert out["n_frames"] == 2.0


def test_rounds_collapse_into_one_column_per_message_type() -> None:
    """Per-round columns would encode a config value in a column NAME, so two
    runs at different K could not be compared with a single CSV read. Per-round
    detail lives in taps.csv."""
    m = CommVolumeMetrics()
    m.add_frame(FrameComms(0, {"comm/r0/sent": 1024, "comm/r1/sent": 1024,
                               "comm/r0/request_sent": 512}, rounds=2))
    out = m.compute()
    assert out["mb_sent"] == 2048 / 2 ** 20        # both rounds, one column
    assert out["mb_request_sent"] == 512 / 2 ** 20
    assert "mb_r0" not in out and "mb_r1" not in out
    assert out["rounds"] == 2.0


def test_message_types_stay_separate() -> None:
    """Feature payload and control payload must be distinguishable: the whole
    point of a request map is that it is small, and a column that merged it
    into the feature bytes would hide that."""
    m = CommVolumeMetrics()
    m.add_frame(FrameComms(0, {"comm/r0/sent": 8192, "comm/r0/request_sent": 32}))
    out = m.compute()
    assert out["mb_sent"] > out["mb_request_sent"] * 100


# ------------------------------------------------------------- the log2 axis --

def test_log2_is_taken_of_the_mean_not_the_mean_of_the_logs() -> None:
    """Jensen: log is concave, so mean(log) <= log(mean), and they diverge as
    the per-frame volume gets erratic -- exactly what a fault does. The
    published figures plot one point against the AVERAGE volume, so the
    average comes first."""
    m = CommVolumeMetrics()
    m.add_frame(FrameComms(0, {"c/sent": 2 ** 10}))
    m.add_frame(FrameComms(1, {"c/sent": 2 ** 20}))
    out = m.compute()
    assert out["log2_bytes"] == math.log2((2 ** 10 + 2 ** 20) / 2)
    assert out["mean_log2_bytes"] == (10.0 + 20.0) / 2
    assert out["mean_log2_bytes"] < out["log2_bytes"]


def test_constant_volume_makes_the_two_log_readings_agree() -> None:
    """The converse of the test above: with no variance Jensen's gap is zero,
    which is what makes the gap itself a usable signal."""
    m = CommVolumeMetrics()
    for i in range(3):
        m.add_frame(FrameComms(i, {"c/sent": 4096}))
    out = m.compute()
    assert out["log2_bytes"] == out["mean_log2_bytes"] == 12.0


def test_zero_bytes_is_nan_not_zero_or_negative_infinity() -> None:
    """0.0 would be indistinguishable from a one-byte message and -inf would
    poison every downstream mean. The raw count survives alongside it."""
    m = CommVolumeMetrics()
    m.add_frame(FrameComms(0, {}))
    out = m.compute()
    assert math.isnan(out["log2_bytes"])
    assert out["bytes_per_frame"] == 0.0


# ------------------------------------------------------- optional protocol --

def test_optional_fields_are_absent_for_models_without_selection() -> None:
    """A model with no selection step should produce no rate column, not a
    column of zeros that reads as 'selected nothing'."""
    m = CommVolumeMetrics()
    m.add_frame(FrameComms(0, {"c/sent": 128}))
    out = m.compute()
    for key in ("rate", "selected_cells_mean", "graph_density"):
        assert key not in out


def test_ratios_are_sum_over_sum_not_mean_of_ratios() -> None:
    """A frame with no possible links contributes 0/0. Guarding that to zero
    and averaging would report a model as failing to communicate when it had
    nobody to communicate with."""
    m = CommVolumeMetrics()
    m.add_frame(FrameComms(0, {"c/sent": 1}, selected_cells=50,
                           cells_per_map=100, graph_links=4, graph_possible=4))
    m.add_frame(FrameComms(1, {"c/sent": 1}, selected_cells=0,
                           cells_per_map=0, graph_links=0, graph_possible=0))
    m.add_frame(FrameComms(2, {"c/sent": 1}, selected_cells=10,
                           cells_per_map=100, graph_links=1, graph_possible=4))
    out = m.compute()
    assert out["rate"] == 60 / 200            # not (0.5 + 0.0 + 0.1) / 3
    assert out["graph_density"] == 5 / 8
    assert out["selected_cells_mean"] == 20.0


# ---------------------------------------------------- channel integration --

def test_from_comm_log_reads_a_real_message_channel() -> None:
    """The intended wiring: the channel counts bytes at transmission time and
    this module only aggregates, so precision assumptions live in one place."""
    import torch

    channel = MessageChannel(bytes_per_element=4)
    channel.new_frame()
    features = torch.zeros(8, 4, 4)
    features[:, 0, 0] = 1.0                    # one non-zero cell of 8 channels
    channel.send(features, sender="cav1", receiver="ego",
                 location="comm/r0/sent", sparse=True)
    channel.send(torch.ones(1, 4, 4), sender="ego", receiver="cav1",
                 location="comm/r0/request_sent", binary=True)

    frame = FrameComms.from_comm_log(channel.log, frame=0, rounds=1)
    assert frame.messages == 2
    # sparse: 1 cell * 8 channels * 4 bytes + 1 cell * 4 bytes of index
    assert frame.bytes_by_location["comm/r0/sent"] == 36
    # binary: 16 cells -> ceil(16/8) = 2 bytes
    assert frame.bytes_by_location["comm/r0/request_sent"] == 2

    m = CommVolumeMetrics()
    m.add_frame(frame)
    assert m.compute()["bytes_per_frame"] == 38.0


def test_sparsity_makes_a_quieter_agent_measurably_cheaper() -> None:
    """The property the whole benchmark rests on: transmitting fewer selected
    cells costs strictly fewer bytes. If this ever stopped holding, the
    fault-lowers-bandwidth finding would be an artefact."""
    import torch

    def bytes_for(n_cells: int) -> int:
        channel = MessageChannel(bytes_per_element=4)
        features = torch.zeros(8, 4, 4)
        features.view(8, -1)[:, :n_cells] = 1.0
        channel.send(features, sender="cav1", receiver="ego",
                     location="comm/r0/sent", sparse=True)
        return channel.log.total_bytes

    assert bytes_for(2) < bytes_for(8) < bytes_for(16)


# ------------------------------------------------------------ the record --

def test_eval_record_flattens_comms_into_comm_columns() -> None:
    row = EvalRecord(epoch=-1, dataset="opv2v", split="test",
                     detection={"ap50": 0.8},
                     comms={"log2_bytes": 14.2, "rate": 0.03}).as_row()
    assert row["comm_log2_bytes"] == 14.2
    assert row["comm_rate"] == 0.03
    assert row["det_ap50"] == 0.8


def test_comms_column_is_additive_for_existing_packages() -> None:
    """Three packages already write this schema. A record that sets no comms
    must produce exactly the columns it did before -- no empty comm_* cells,
    no reordering."""
    row = EvalRecord(epoch=0, dataset="d", split="test",
                     detection={"ap50": 0.5}).as_row()
    assert not [key for key in row if key.startswith("comm_")]


def test_comms_is_not_folded_into_system() -> None:
    """Latency is a property of the machine; transmitted volume is a property
    of the model's decisions. Sharing a prefix would file a headline result
    under profiling -- and would collide on any shared key name."""
    row = EvalRecord(epoch=0, dataset="d", split="test",
                     system={"latency_ms": 12.0},
                     comms={"latency_ms": 999.0}).as_row()
    assert row["sys_latency_ms"] == 12.0
    assert row["comm_latency_ms"] == 999.0
