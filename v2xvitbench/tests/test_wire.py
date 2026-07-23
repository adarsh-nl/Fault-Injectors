"""
The registry-versus-reality cross-check, in both directions.

``observation/locations.py`` documents what can be observed; these tests run
a real (tiny) model and compare. Direction one: everything emitted must be
registered, under the module the registry names -- otherwise the registry
lies about what a name means. Direction two: everything registered must be
emitted -- otherwise a taps config naming it validates cleanly and records
nothing, which on a cluster looks exactly like a broken tap.

Config-gated locations (``mswin/weights`` under naive fusion, ``rte/*`` with
the DPE off) are exercised explicitly rather than excluded, so the gating
itself is pinned.
"""

from __future__ import annotations

import torch

from cpbench.data import GridSpec
from cpbench.observation import StatsTap, TapSet

from v2xvitbench.models import V2XViT
from v2xvitbench.observation import all_locations, validate_location

DEPTH = 2
BRANCHES = 2


def _model(**kwargs) -> V2XViT:
    spec = GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
                    downsample=4)
    defaults = dict(max_cav=3, encoder_out_channels=48, shrink_channels=32,
                    depth=DEPTH, hmsa_heads=2, hmsa_dim_head=16,
                    window_sizes=(2, 4), mswin_heads=(2, 2),
                    mswin_dim_heads=(16, 16), mlp_dim=32, dropout=0.0)
    defaults.update(kwargs)
    return V2XViT(spec, **defaults).eval()


def _batch(n_agents: int = 3) -> dict:
    generator = torch.Generator().manual_seed(0)
    coords = torch.stack([
        torch.arange(12) % n_agents,
        torch.arange(12) % 16,
        torch.arange(12) % 16], dim=1)
    return {
        "features": torch.randn(12, 4, 9, generator=generator),
        "coords": coords,
        "num_points": torch.full((12,), 4),
        "record_len": [n_agents],
        "T_agent_to_ego": torch.eye(4).expand(1, n_agents, 4, 4).contiguous(),
        "time_delay": torch.tensor([[0, 2, 1]]),
        "infra": torch.tensor([[0, 1, 0]]),      # both HMSA type paths run
        "velocity": torch.tensor([[10.0, 0.0, 5.0]]),
    }


def _run(model: V2XViT) -> list:
    tap = StatsTap()
    model(_batch(), taps=TapSet([tap], strict=True))
    return tap.records


def test_every_emitted_location_is_registered() -> None:
    for record in _run(_model()):
        declared = validate_location(record.location)
        assert record.module in declared.emitters(), (
            f"{record.location}: registry says {declared.module}, "
            f"emitted by {record.module}")


def test_every_registered_location_is_emitted() -> None:
    emitted = {r.location for r in _run(_model())}
    expected = set(all_locations(depth=DEPTH, branches=BRANCHES))
    missing = sorted(expected - emitted)
    assert not missing, "registered but never emitted:\n  " + "\n  ".join(missing)


def test_no_location_outside_the_registry_dimensions_is_emitted() -> None:
    """The template expansion must match the model's real depth and branch
    count: a model emitting fusion/l2/* while the registry expands to l1
    means the taps config and the model disagree about the architecture."""
    emitted = {r.location for r in _run(_model())}
    expected = set(all_locations(depth=DEPTH, branches=BRANCHES))
    extra = sorted(emitted - expected)
    assert not extra, "emitted but not registered:\n  " + "\n  ".join(extra)


def test_naive_fusion_gates_only_the_weights_location() -> None:
    emitted = {r.location for r in _run(_model(fusion_method="naive"))}
    expected = set(all_locations(depth=DEPTH, branches=BRANCHES))
    gated = {name for name in expected if name.endswith("mswin/weights")}
    assert not (gated & emitted)
    assert expected - gated <= emitted


def test_disabling_rte_gates_only_the_rte_locations() -> None:
    emitted = {r.location for r in _run(_model(use_rte=False))}
    expected = set(all_locations(depth=DEPTH, branches=BRANCHES))
    gated = {name for name in expected if name.startswith("rte/")}
    assert not (gated & emitted)
    assert expected - gated <= emitted


def test_the_ego_head_emits_once_not_once_per_agent() -> None:
    records = _run(_model())
    cls_records = [r for r in records if r.location == "head/cls_logits"]
    assert len(cls_records) == 1
