"""
Tests for message packing, the communication graph and volume accounting --
the stretch where a selection mask becomes actual bytes.

This is where the package's central claim becomes falsifiable. Everything
upstream produces tensors; these three modules produce the *number* that gets
reported next to AP, and a mistake here would not fail anywhere. It would
publish a wrong bandwidth.

So the accounting invariants are asserted directly against a real
``MessageChannel``: the self-link is free, a broadcast request is charged once,
fewer selected cells cost strictly fewer bytes, and a training-mode
measurement is refused rather than returned.
"""

from __future__ import annotations

from collections import Counter

import math
import pytest
import torch

from cpbench.comms.channel import MessageChannel
from cpbench.observation import StatsTap, TapSet
from w2cbench.comm import (CommunicationGraph, CommVolumeAccountant,
                           MessagePacker, graph_density, incoming_links,
                           message_statistics)
from w2cbench.observation import validate_location


def _features(n_agents: int = 3, channels: int = 8, hw: int = 4) -> torch.Tensor:
    return torch.ones(n_agents, channels, hw, hw)


def _mask(n_agents: int = 3, hw: int = 4, cells: int = 1) -> torch.Tensor:
    """Every agent sends `cells` cells to agent 0; self-links unmasked (A6)."""
    mask = torch.zeros(n_agents, n_agents, hw, hw)
    flat = mask[:, 0].reshape(n_agents, -1)
    flat[:, :cells] = 1.0
    mask[:, 0] = flat.reshape(n_agents, hw, hw)
    for i in range(n_agents):
        mask[i, i] = 1.0
    return mask


# ------------------------------------------------------------------ packing --

def test_packing_is_the_mask_times_the_features() -> None:
    packer = MessagePacker()
    features, mask = _features(), _mask(cells=2)
    messages = packer(mask, features, receiver=0)
    assert messages.shape == (3, 8, 4, 4)
    assert torch.equal(messages, mask[:, 0].unsqueeze(1) * features)


def test_unselected_cells_are_zero_and_selected_cells_survive() -> None:
    messages = MessagePacker()(_mask(cells=1), _features(), receiver=0)
    assert float(messages[1, :, 0, 0].sum()) == 8.0     # one cell, 8 channels
    assert float(messages[1, :, 1, 1].sum()) == 0.0


def test_packing_only_materialises_the_receiver_column() -> None:
    """The paper writes the message set pairwise, but materialising it is 615
    MB per round at OPV2V scale against 123 MB for what one receiver actually
    consumes. The mask stays pairwise and observable; the messages do not."""
    messages = MessagePacker()(_mask(), _features(), receiver=0)
    assert messages.dim() == 4                          # (L, D, H, W), not (L, L, ...)


def test_shape_mismatches_are_named() -> None:
    packer = MessagePacker()
    with pytest.raises(ValueError, match=r"mask \(L, L, H, W\)"):
        packer(torch.zeros(3, 4, 4), _features())
    with pytest.raises(ValueError, match="covers 2 agents"):
        packer(_mask(n_agents=2), _features(n_agents=3))


# ------------------------------------------------------- byte accounting --

def test_the_self_link_is_never_charged() -> None:
    """The receiver's own features are already local. Charging them would
    inflate every reported volume by one agent's worth of features and make
    the paper's compression ratio look worse than it is -- and A6 forces the
    self-link mask to ones precisely because those cells are free."""
    channel = MessageChannel(bytes_per_element=4)
    MessagePacker()(_mask(n_agents=3), _features(), receiver=0, channel=channel)
    assert channel.log.messages == 2                    # 3 agents, 2 senders


def test_absent_links_are_not_charged() -> None:
    """A link the graph marks absent transmitted nothing, so it costs
    nothing -- otherwise an agent that had nothing to say would still show up
    in the bandwidth column."""
    graph = torch.ones(3, 3)
    graph[2, 0] = 0.0
    channel = MessageChannel(bytes_per_element=4)
    MessagePacker()(_mask(), _features(), receiver=0, channel=channel,
                    graph=graph)
    assert channel.log.messages == 1


def test_bytes_match_a_hand_computation() -> None:
    """Pinned arithmetic: sparse cost is cells x channels x precision plus 4
    index bytes per cell, and the whole benchmark's x-axis rests on it."""
    channel = MessageChannel(bytes_per_element=4)
    MessagePacker()(_mask(n_agents=2, cells=3), _features(n_agents=2, channels=8),
                    receiver=0, channel=channel)
    # one sender, 3 cells: 3 * 8 * 4 payload + 3 * 4 index
    assert channel.log.total_bytes == 3 * 8 * 4 + 3 * 4


def test_fewer_selected_cells_cost_strictly_fewer_bytes() -> None:
    """The property the 'a fault lowered bandwidth' finding depends on. If it
    ever stopped holding, that result would be an accounting artefact."""
    def cost(cells: int) -> int:
        channel = MessageChannel(bytes_per_element=4)
        MessagePacker()(_mask(cells=cells), _features(), receiver=0,
                        channel=channel)
        return channel.log.total_bytes

    assert cost(1) < cost(4) < cost(16)


def test_request_maps_are_charged_once_per_sender_not_once_per_link() -> None:
    """R_i does not depend on the receiver, so a real radio broadcasts it
    once. Charging it L-1 times would scale the control-plane bytes with the
    agent count and make 'request maps are cheap' look false when it is not.
    """
    channel = MessageChannel(bytes_per_element=4)
    request = torch.rand(4, 1, 4, 4)
    MessagePacker()(_mask(n_agents=4), _features(n_agents=4), receiver=0,
                    channel=channel, request=request)
    by_location = channel.log.bytes_by_location
    assert by_location["comm/r0/request_sent"] == 4 * 16 * 4     # 4 senders
    assert channel.log.messages == 3 + 4                        # 3 links + 4 broadcasts


def test_request_is_not_charged_when_no_round_will_consume_it() -> None:
    """An unconsumed message is not a message. At K=1 nobody answers a
    request, so transmitting it would be charging for a packet the protocol
    never sends."""
    channel = MessageChannel(bytes_per_element=4)
    MessagePacker()(_mask(), _features(), receiver=0, channel=channel,
                    request=None)
    assert "comm/r0/request_sent" not in channel.log.bytes_by_location


# -------------------------------------------------------------- statistics --

def test_message_statistics_excludes_the_free_self_link() -> None:
    """Including it would report a selection ratio inflated by a full map and
    would hide a collapse in what collaborators actually sent."""
    mask = torch.zeros(3, 3, 4, 4)
    mask[1, 0, :2, :2] = 1.0
    mask[2, 0, 0, 0] = 1.0
    mask[0, 0] = 1.0
    stats = message_statistics(mask, receiver=0)
    assert stats == {"selected_cells": 2.5, "cells_per_map": 16.0,
                     "n_links": 2.0}


def test_message_statistics_survives_a_lone_agent() -> None:
    stats = message_statistics(torch.ones(1, 1, 4, 4), receiver=0)
    assert stats["selected_cells"] == 0.0 and stats["n_links"] == 0.0


# ------------------------------------------------------------------- graph --

def test_link_exists_when_any_cell_was_selected() -> None:
    graph = CommunicationGraph()
    mask = torch.zeros(3, 3, 4, 4)
    mask[1, 0, 0, 0] = 1.0
    out = graph(mask, round_index=1)
    assert float(out[1, 0]) == 1.0
    assert float(out[2, 0]) == 0.0


def test_round_zero_is_complete_regardless_of_the_mask() -> None:
    """At round 0 nobody has requested anything, so every agent broadcasts on
    its own confidence. The link exists because the broadcast happened;
    whether that sender had anything worth putting in it is a property of the
    message, not the link. Conflating them would make round-0 density fall
    whenever the scene is empty, which has nothing to do with connectivity."""
    graph = CommunicationGraph()
    empty = torch.zeros(3, 3, 4, 4)
    assert float(graph(empty, round_index=0).sum()) == 9.0
    assert float(graph(empty, round_index=1).sum()) == 0.0


def test_broadcast_round_zero_can_be_disabled_for_the_ablation() -> None:
    strict = CommunicationGraph(broadcast_round_zero=False)
    assert float(strict(torch.zeros(3, 3, 4, 4), round_index=0).sum()) == 0.0


def test_a_dropped_agent_leaves_the_graph_even_at_round_zero() -> None:
    """A broadcast to an absent agent is not a link. This is the seam where an
    agent-drop fault becomes visible in the topology rather than only in the
    features."""
    graph = CommunicationGraph()
    present = torch.tensor([True, True, False])
    out = graph(torch.zeros(3, 3, 4, 4), agent_mask=present, round_index=0)
    assert float(out.sum()) == 4.0                  # the 2x2 present block
    assert float(out[2].sum()) == 0.0 and float(out[:, 2].sum()) == 0.0


def test_non_square_agent_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="square agent block"):
        CommunicationGraph()(torch.zeros(3, 2, 4, 4))


def test_density_excludes_self_links() -> None:
    """Counting L guaranteed self-links would put a floor under the density
    that RISES as the agent count falls, so a scene losing collaborators would
    report higher connectivity."""
    assert graph_density(torch.eye(3)) == 0.0
    assert graph_density(torch.ones(3, 3)) == 1.0
    assert graph_density(torch.ones(4, 4), include_self=True) == 1.0


def test_reported_density_counts_incoming_links_not_the_whole_matrix() -> None:
    """Where2comm fuses for one receiver at a time, so only that receiver's
    column is ever used. Density over all L*(L-1) ordered pairs is
    structurally capped at 1/(L-1): at L=5 a graph in which every collaborator
    reached the ego would report 0.2, reading as '80% of the topology unused'
    when all of the USED topology was active."""
    graph = torch.zeros(5, 5)
    graph[1:, 0] = 1.0                          # every collaborator reached the ego
    assert incoming_links(graph, receiver=0) == (4.0, 4.0)
    assert graph_density(graph) == pytest.approx(4 / 20)     # the misleading one


# --------------------------------------------------------- the accountant --

def _run_frame(accountant: CommVolumeAccountant, cells: int = 1,
               frame: int = 0) -> None:
    mask, features = _mask(cells=cells), _features()
    graph = CommunicationGraph()(mask, round_index=1)
    accountant.start_frame()
    MessagePacker()(mask, features, receiver=0, channel=accountant.channel,
                    graph=graph)
    accountant.record_round(mask, graph, receiver=0)
    accountant.end_frame(frame)


def test_accountant_produces_the_comm_columns() -> None:
    accountant = CommVolumeAccountant(bytes_per_element=4)
    _run_frame(accountant)
    out = accountant.compute()
    assert out["n_frames"] == 1.0
    assert out["rounds"] == 1.0
    assert out["bytes_per_frame"] == 2 * (1 * 8 * 4 + 4)
    assert out["log2_bytes"] == math.log2(out["bytes_per_frame"])
    assert 0.0 < out["rate"] <= 1.0
    assert out["graph_density"] == 1.0


def test_frames_do_not_accumulate_each_others_bytes() -> None:
    """Without the per-frame reset the mean would grow linearly with the frame
    index -- a bug that looks exactly like a memory leak in the protocol."""
    accountant = CommVolumeAccountant(bytes_per_element=4)
    for frame in range(4):
        _run_frame(accountant, frame=frame)
    out = accountant.compute()
    assert out["n_frames"] == 4.0
    assert out["bytes_per_frame"] == 2 * (1 * 8 * 4 + 4)     # not 4x that


def test_precision_default_matches_the_papers_log2_formula() -> None:
    """A8: the paper's log2(|M| * D * 32/8) is float32. MessageChannel
    defaults to fp16, and using that default would put every point exactly 1.0
    below every published one."""
    fp32, fp16 = (CommVolumeAccountant(bytes_per_element=4),
                  CommVolumeAccountant(bytes_per_element=2))
    _run_frame(fp32)
    _run_frame(fp16)
    payload_ratio = fp32.compute()["bytes_per_frame"] / fp16.compute()["bytes_per_frame"]
    assert payload_ratio > 1.0
    assert CommVolumeAccountant().channel.bytes_per_element == 4


def test_training_mode_measurement_is_refused() -> None:
    """A17. The selector keeps a random fraction of the map during training,
    so a measured volume would be a draw from the curriculum -- wrong by a
    random factor between 0.1 and 1.0, with nothing to indicate it."""
    accountant = CommVolumeAccountant()
    accountant.start_frame()
    with pytest.raises(RuntimeError, match="training mode"):
        accountant.end_frame(0, training=True)


def test_lifecycle_errors_are_explicit() -> None:
    accountant = CommVolumeAccountant()
    with pytest.raises(RuntimeError, match="outside a frame"):
        accountant.record_round(_mask(), torch.ones(3, 3))
    with pytest.raises(RuntimeError, match="without a matching start_frame"):
        accountant.end_frame(0)


def test_reset_clears_every_condition_boundary() -> None:
    """The benchmark runner reuses one model across conditions; without a
    reset each condition's volume would carry the previous one's."""
    accountant = CommVolumeAccountant()
    _run_frame(accountant)
    accountant.reset()
    assert accountant.compute() == {"n_frames": 0.0}


def test_accountant_is_not_a_module_so_state_cannot_live_on_the_model() -> None:
    """Run-scoped mutable state on an nn.Module would have each fault
    condition's volume contaminated by the last, silently and always in the
    direction that looks like more traffic."""
    assert not isinstance(CommVolumeAccountant(), torch.nn.Module)


# ------------------------------------------------- registry vs. reality --

def test_the_wire_emits_exactly_the_registered_locations() -> None:
    tap = TapSet([StatsTap()], strict=True)
    stats = tap.taps[0]
    accountant = CommVolumeAccountant(bytes_per_element=4, taps=tap)
    mask, features = _mask(), _features()
    graph = CommunicationGraph()(mask, taps=tap, round_index=1)
    accountant.start_frame()
    MessagePacker()(mask, features, receiver=0, channel=accountant.channel,
                    graph=graph, request=torch.rand(3, 1, 4, 4), taps=tap,
                    round_index=1)
    accountant.record_round(mask, graph, receiver=0, round_index=1)
    accountant.end_frame(0)

    counts = Counter(r.location for r in stats.records)
    assert set(counts) == {"comm/r1/comm_graph", "comm/r1/message_sparse",
                           "comm/r1/sent", "comm/r1/request_sent",
                           "comm/r1/comm_rate", "comm/r1/bytes"}
    for record in stats.records:
        assert record.module in validate_location(record.location).emitters(), (
            f"{record.location}: emitted by {record.module}")


def test_sent_is_emitted_once_per_message() -> None:
    """One record per transmission, not one per round: the layer-wise analysis
    attributes bandwidth to the agent that caused it."""
    tap = TapSet([StatsTap()], strict=True)
    channel = MessageChannel(bytes_per_element=4, taps=tap)
    MessagePacker()(_mask(n_agents=4), _features(n_agents=4), receiver=0,
                    channel=channel)
    counts = Counter(r.location for r in tap.taps[0].records)
    assert counts["comm/r0/sent"] == 3


def test_taps_none_does_not_change_the_result() -> None:
    packer, mask, features = MessagePacker(), _mask(), _features()
    assert torch.equal(
        packer(mask, features, receiver=0),
        packer(mask, features, receiver=0, taps=TapSet([StatsTap()], strict=True)))
