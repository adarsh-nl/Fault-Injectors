"""
Tests for the dataset and collator: pillar layout, metadata extraction, and
the plane-1 latency fault flowing into ``time_delay``.

The latency-to-delay wiring is the load-bearing one: it is what makes the
plane-1 latency condition the paper's OWN asynchronous setting (delay known,
DPE compensates), and what the delay_encoding fault later corrupts. If it
silently broke, every latency number in the benchmark would describe a model
running blind while claiming to be delay-aware.
"""

from __future__ import annotations

import pytest
import torch

from cpbench.data import GridSpec, SyntheticCooperativeDataset
from cpbench.faults import DataFaultBridge

from v2xvitbench.data import (V2XVitLidarDataset, collate_v2xvit,
                              v2xvit_collator)
from v2xvitbench.models import V2XViT


def _spec() -> GridSpec:
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
                    downsample=4)


def _dataset(**kwargs) -> V2XVitLidarDataset:
    adapter = SyntheticCooperativeDataset(n_frames=3, n_agents=3)
    return V2XVitLidarDataset(adapter, _spec(), max_cav=3, **kwargs)


# ---------------------------------------------------------------- dataset --

def test_item_has_every_documented_key() -> None:
    item = _dataset()[0]
    for key in ("features", "coords", "num_points", "T_agent_to_ego",
                "time_delay", "infra", "velocity", "gt_boxes", "n_agents",
                "frame", "fault_records"):
        assert key in item, key
    assert item["features"].shape[1:] == (32, 9)
    assert item["coords"].shape[1] == 3
    assert item["time_delay"].dtype == torch.long
    assert item["velocity"].dtype == torch.float32


def test_clean_dataset_reports_zero_delay_and_no_faults() -> None:
    ds = _dataset()
    item = ds[0]
    assert ds.is_clean
    assert item["time_delay"].tolist() == [0, 0, 0]
    assert item["fault_records"] == []


def test_synthetic_agents_are_vehicles_unless_forced() -> None:
    assert _dataset()[0]["infra"].tolist() == [0, 0, 0]
    assert _dataset(force_infra=[2])[0]["infra"].tolist() == [0, 0, 1]


def test_forcing_the_ego_slot_to_infra_is_refused() -> None:
    with pytest.raises(ValueError, match="slot 0"):
        _dataset(force_infra=[0])


def test_latency_fault_reaches_time_delay() -> None:
    """THE wiring test: a plane-1 latency injector serves non-ego agents a
    stale frame and logs the staleness; the dataset must report exactly that
    number to the model."""
    bridge = DataFaultBridge(
        {"pipeline": {"latency": {"mu_delay": 0.3, "sigma_jitter": 0.0}},
         "seed": 0}, fps=10.0)
    ds = _dataset(bridge=bridge)
    item = ds[1]                       # frame 0 has nothing older to serve
    delays = item["time_delay"].tolist()
    assert delays[0] == 0, "the ego is never delayed"
    assert any(d > 0 for d in delays[1:]), (
        "latency configured but no non-ego delay recorded")
    assert any(r.fault_type == "comm_latency" for r in item["fault_records"])


def test_velocity_defaults_to_zero_on_synthetic() -> None:
    assert _dataset()[0]["velocity"].tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------- collate --

def test_collate_makes_agent_indices_global() -> None:
    ds = _dataset()
    batch = collate_v2xvit([ds[0], ds[1]], max_cav=3)
    assert batch["record_len"] == [3, 3]
    first = batch["coords"][:, 0]
    assert int(first.max()) == 5       # 6 agents across the batch, 0-indexed
    assert batch["T_agent_to_ego"].shape == (2, 3, 4, 4)
    assert batch["time_delay"].shape == (2, 3)
    assert batch["infra"].shape == (2, 3)
    assert batch["velocity"].shape == (2, 3)


def test_collate_pads_metadata_with_benign_values() -> None:
    ds = _dataset(force_infra=[1, 2])
    batch = collate_v2xvit([ds[0]], max_cav=5)
    assert batch["infra"].tolist() == [[0, 1, 1, 0, 0]]
    assert batch["time_delay"][0, 3:].tolist() == [0, 0]
    identity = torch.eye(4)
    assert torch.equal(batch["T_agent_to_ego"][0, 4], identity)


def test_collated_batch_drives_the_model() -> None:
    """The dataset/collator/model contract, end to end on synthetic data."""
    ds = _dataset(force_infra=[2])
    model = V2XViT(_spec(), max_cav=3, encoder_out_channels=48,
                   shrink_channels=32, depth=1, hmsa_heads=2,
                   hmsa_dim_head=16, window_sizes=(2, 4), mswin_heads=(2, 2),
                   mswin_dim_heads=(16, 16), mlp_dim=32, dropout=0.0).eval()
    out = model(v2xvit_collator(3)([ds[0], ds[1]]))
    assert out["cls"].shape == (2, 2, 16, 16)
    assert torch.isfinite(out["cls"]).all()
