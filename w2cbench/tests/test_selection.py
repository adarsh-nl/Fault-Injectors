"""
Tests for the request map and the three selection strategies.

Selection is where assumption A1 becomes code, and A1 is the choice that
decides what the fault benchmark can observe at all: under a threshold a
sensor fault moves the bandwidth column, under a budget it cannot. Both
behaviours are therefore asserted directly, against the same degraded input,
so the divergence is a pinned property of the suite rather than a claim in a
design document.

The other thing pinned here is the training branch. The released module keeps
a random *fraction* of the map during training -- the paper's curriculum --
which means a communication measurement taken in train mode is a sample from
that curriculum and not a model decision. Getting this wrong would not fail;
it would silently make every reported bandwidth number wrong by a random
factor.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from cpbench.comms.channel import MessageChannel
from cpbench.observation import StatsTap, TapSet
from w2cbench.comm import (BudgetSelector, RequestMapGenerator,
                           ThresholdSelector, TopKSelector, top_k_mask)
from w2cbench.observation import validate_location


def _priority(n_agents: int = 3, hw: int = 8,
              seed: int = 0) -> torch.Tensor:
    """A (L, L, H, W) priority block with distinct per-link content."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n_agents, n_agents, hw, hw, generator=generator)


# --------------------------------------------------------- the request map --

def test_request_is_the_complement_of_confidence() -> None:
    gen = RequestMapGenerator()
    confidence = torch.rand(3, 1, 8, 8)
    assert torch.allclose(gen(confidence), 1.0 - confidence)


def test_a_blind_agent_asks_for_everything() -> None:
    """The signal a protocol fault destroys: an agent that cannot see reports
    maximal need, and a partner that can see fills it in."""
    gen = RequestMapGenerator()
    assert float(gen(torch.zeros(1, 1, 4, 4)).mean()) == 1.0
    assert float(gen(torch.ones(1, 1, 4, 4)).mean()) == 0.0


def test_request_map_emits_its_registered_location() -> None:
    tap = StatsTap()
    RequestMapGenerator()(torch.rand(2, 1, 4, 4),
                          taps=TapSet([tap], strict=True), round_index=1)
    records = {r.location: r.module for r in tap.records}
    assert records == {"comm/r1/request_map": "RequestMapGenerator"}
    assert "RequestMapGenerator" in validate_location(
        "comm/r1/request_map").emitters()


# ------------------------------------------------------------ the top-k core --

def test_top_k_mask_keeps_exactly_the_k_largest() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.2]])
    assert torch.equal(top_k_mask(scores, 2), torch.tensor([[0.0, 1.0, 1.0, 0.0]]))


def test_top_k_mask_clamps_both_ends() -> None:
    scores = torch.tensor([[0.1, 0.9]])
    assert float(top_k_mask(scores, 0).sum()) == 0.0
    assert float(top_k_mask(scores, 99).sum()) == 2.0


# ------------------------------------------------------------ A6, self-links --

@pytest.mark.parametrize("selector", [
    ThresholdSelector(threshold=0.99),      # would select almost nothing
    TopKSelector(k=1),
    BudgetSelector(budget_bytes=0, channels=32),   # budget of zero cells
])
def test_self_link_is_never_masked(selector) -> None:
    """A6. An agent's own features are already local, so masking them would
    discard information for no bandwidth saving. Tested against strategies
    that would otherwise select almost nothing, which is where a missing
    self-mask would actually hurt."""
    mask = selector.eval()(_priority(n_agents=4))
    for i in range(4):
        assert float(mask[i, i].mean()) == 1.0


def test_self_mask_can_be_disabled_for_the_ablation() -> None:
    strict = TopKSelector(k=1, self_mask="none").eval()
    mask = strict(_priority(n_agents=3))
    assert float(mask[0, 0].sum()) == 1.0


def test_invalid_self_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="self_mask must be"):
        TopKSelector(k=1, self_mask="diagonal")


# ------------------------------------------------------ the A1 divergence --

def _degrade(priority: torch.Tensor, factor: float = 0.2) -> torch.Tensor:
    """A sensor fault flattens the confidence map, so priorities shrink."""
    return priority * factor


def test_threshold_lets_a_fault_move_the_bandwidth_column() -> None:
    """The pathological case the benchmark exists to expose: a degraded sensor
    lowers confidence, fewer cells clear the bar, and measured bandwidth FALLS
    while perception degrades. Reported alone that column says the system got
    more efficient."""
    selector = ThresholdSelector(threshold=0.5).eval()
    healthy = selector(_priority())
    degraded = selector(_degrade(_priority()))
    assert float(degraded.sum()) < float(healthy.sum())


def test_topk_holds_bandwidth_fixed_under_the_same_fault() -> None:
    """The opposite behaviour from identical input. The budget is spent on
    cells the agent is no longer confident about, so the damage lands entirely
    in accuracy -- which is why shipping only one selector would make the
    benchmark answer half the question."""
    selector = TopKSelector(k=10).eval()
    healthy = selector(_priority())
    degraded = selector(_degrade(_priority()))
    assert float(degraded.sum()) == float(healthy.sum())


def test_a_uniform_rescale_does_not_change_which_cells_topk_picks() -> None:
    """Multiplying every priority by a constant is a pure bandwidth signal
    under a threshold and a pure no-op under top-k. Pinning it separates 'the
    fault changed the ranking' from 'the fault changed the magnitude'."""
    selector = TopKSelector(k=10).eval()
    assert torch.equal(selector(_priority()), selector(_degrade(_priority())))


# ------------------------------------------------------------- the strategies --

def test_threshold_selects_exactly_the_cells_above_the_bar() -> None:
    priority = _priority(n_agents=2)
    selector = ThresholdSelector(threshold=0.5).eval()
    mask = selector(priority)
    expected = (priority > 0.5).float()
    assert torch.equal(mask[0, 1], expected[0, 1])


def test_non_square_agent_block_is_rejected() -> None:
    """The two agent axes are sender and receiver; A6 needs them to index the
    same agents. A transposed or half-built priority block would otherwise
    self-mask the wrong links and produce a plausible wrong answer."""
    with pytest.raises(ValueError, match="square agent block"):
        TopKSelector(k=2).eval()(torch.rand(3, 2, 8, 8))


def test_topk_selects_the_k_largest_per_link() -> None:
    priority = _priority(n_agents=3, hw=8)
    mask = TopKSelector(k=5).eval()(priority)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            assert float(mask[i, j].sum()) == 5
            chosen = priority[i, j][mask[i, j].bool()]
            rejected = priority[i, j][~mask[i, j].bool()]
            assert float(chosen.min()) >= float(rejected.max())


def test_budget_converts_bytes_to_cells_the_way_the_channel_charges() -> None:
    selector = BudgetSelector(budget_bytes=1024, channels=32,
                              bytes_per_element=4)
    assert selector.bytes_per_cell == 32 * 4 + 4
    assert selector.k == 1024 // 132


def test_budget_is_never_exceeded_by_the_real_channel() -> None:
    """The one failure a bandwidth-constrained benchmark cannot tolerate: a
    budget the accountant then over-runs. Checked against a real
    MessageChannel rather than against the selector's own arithmetic, so the
    two cannot drift into agreement with each other and out of agreement with
    reality."""
    channels, budget = 16, 2048
    selector = BudgetSelector(budget_bytes=budget, channels=channels,
                              bytes_per_element=4).eval()
    mask = selector(_priority(n_agents=3, hw=16))

    features = torch.randn(channels, 16, 16)
    for sender in range(3):
        for receiver in range(3):
            if sender == receiver:
                continue                      # A6 exempts the self-link
            channel = MessageChannel(bytes_per_element=4)
            channel.send(features * mask[sender, receiver], sender="a",
                         receiver="b", location="comm/r0/sent", sparse=True)
            assert channel.log.total_bytes <= budget


def test_zero_budget_transmits_nothing_across_links() -> None:
    selector = BudgetSelector(budget_bytes=0, channels=32).eval()
    mask = selector(_priority(n_agents=2))
    assert float(mask[0, 1].sum()) == 0.0


def test_invalid_budget_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="budget_bytes"):
        BudgetSelector(budget_bytes=-1, channels=32)
    with pytest.raises(ValueError, match="channels must be positive"):
        BudgetSelector(budget_bytes=100, channels=0)
    with pytest.raises(ValueError, match="k must be non-negative"):
        TopKSelector(k=-1)


# ------------------------------------------------------- the training branch --

def test_training_ignores_the_configured_threshold() -> None:
    """The released curriculum. A threshold of 1.0 selects nothing in eval and
    still selects most of the map in training, because training samples a
    bandwidth rather than applying the rule."""
    selector = ThresholdSelector(threshold=1.0)
    priority = _priority(n_agents=2)

    selector.eval()
    assert float(selector(priority)[0, 1].sum()) == 0.0

    selector.train()
    assert float(selector(priority)[0, 1].sum()) > 0.0


def test_training_keeps_the_highest_priority_cells_not_random_ones() -> None:
    """'Random top-k', not 'random mask'. Only HOW MANY cells is random; WHICH
    cells is still the confidence ranking. Training on genuinely random cells
    would teach fusion to trust noise."""
    torch.manual_seed(0)
    selector = ThresholdSelector(threshold=0.5).train()
    priority = _priority(n_agents=2, hw=8)
    link = selector(priority)[0, 1].bool()
    chosen = priority[0, 1][link]
    rejected = priority[0, 1][~link]
    if rejected.numel():
        assert float(chosen.min()) >= float(rejected.max())


def test_training_bandwidth_varies_between_batches() -> None:
    """What makes one checkpoint serve the whole accuracy-versus-bandwidth
    curve: the model sees every operating point during training."""
    torch.manual_seed(0)
    selector = TopKSelector(k=1).train()
    priority = _priority(n_agents=2, hw=16)
    volumes = {float(selector(priority)[0, 1].sum()) for _ in range(8)}
    assert len(volumes) > 1


def test_training_bandwidth_respects_the_configured_range() -> None:
    torch.manual_seed(0)
    selector = TopKSelector(k=1, train_bandwidth=(0.5, 0.5)).train()
    n_cells = 16 * 16
    mask = selector(_priority(n_agents=2, hw=16))
    assert float(mask[0, 1].sum()) == pytest.approx(n_cells * 0.5, abs=1)


def test_training_is_reproducible_under_a_fixed_seed() -> None:
    """Randomness goes through torch, so seed_everything covers it. Without
    this, two runs of the same config would train on different bandwidth
    schedules and be incomparable."""
    priority = _priority(n_agents=2, hw=16)
    selector = TopKSelector(k=1).train()
    torch.manual_seed(7)
    first = selector(priority).clone()
    torch.manual_seed(7)
    assert torch.equal(first, selector(priority))


def test_invalid_train_bandwidth_is_rejected() -> None:
    with pytest.raises(ValueError, match="train_bandwidth"):
        TopKSelector(k=1, train_bandwidth=(0.8, 0.2))
    with pytest.raises(ValueError, match="train_bandwidth"):
        TopKSelector(k=1, train_bandwidth=(0.0, 1.5))


# ------------------------------------------------------------------ contract --

@pytest.mark.parametrize("selector", [
    ThresholdSelector(threshold=0.5), TopKSelector(k=4),
    BudgetSelector(budget_bytes=512, channels=8),
])
def test_every_strategy_shares_one_shape_contract(selector) -> None:
    """Interchangeability is the point of the protocol: a sweep swaps the
    strategy and nothing downstream notices."""
    priority = _priority(n_agents=3, hw=8)
    mask = selector.eval()(priority)
    assert mask.shape == priority.shape
    assert mask.dtype == priority.dtype
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_leading_batch_dimensions_are_preserved() -> None:
    """The orchestrator loops over samples, but nothing here requires it to --
    a batched caller works unchanged."""
    mask = TopKSelector(k=3).eval()(torch.rand(2, 3, 3, 8, 8))
    assert mask.shape == (2, 3, 3, 8, 8)
    assert float(mask[1, 0, 1].sum()) == 3


def test_masks_are_hard_so_no_gradient_reaches_the_confidence_head() -> None:
    """Which is exactly why A11 matters: the head is supervised directly by
    the round-0 detection loss. Were that removed, the tensor deciding what
    gets transmitted would be trained only through the mask -- and the mask is
    where its gradient was supposed to come from."""
    priority = torch.rand(2, 2, 4, 4, requires_grad=True)
    mask = ThresholdSelector(threshold=0.5).eval()(priority)
    assert not mask.requires_grad


# ------------------------------------------------- registry vs. reality --

def test_selectors_emit_exactly_the_registered_locations() -> None:
    tap = StatsTap()
    TopKSelector(k=4).eval()(_priority(), taps=TapSet([tap], strict=True),
                             round_index=2)
    counts = Counter(r.location for r in tap.records)
    assert set(counts) == {"comm/r2/selection_scores",
                           "comm/r2/selection_mask",
                           "comm/r2/selected_count"}
    assert set(counts.values()) == {1}
    for record in tap.records:
        assert record.module in validate_location(record.location).emitters(), (
            f"{record.location}: emitted by {record.module}")


def test_each_strategy_registers_itself_as_the_emitter() -> None:
    """The registry lists all three as alternatives, so 'which layer failed'
    answers name the strategy that was actually configured."""
    for selector, name in ((ThresholdSelector(0.5), "ThresholdSelector"),
                           (TopKSelector(k=2), "TopKSelector"),
                           (BudgetSelector(512, channels=8), "BudgetSelector")):
        tap = StatsTap()
        selector.eval()(_priority(), taps=TapSet([tap], strict=True))
        assert {r.module for r in tap.records} == {name}
        assert name in validate_location("comm/r0/selection_mask").emitters()


def test_selected_count_is_the_quantity_a_fault_moves() -> None:
    """comm/r{k}/selected_count is the third link of the causal chain, and its
    shape has to be per-link for the layer-wise analysis to attribute a
    bandwidth change to the agent that caused it."""
    tap = StatsTap()
    ThresholdSelector(threshold=0.5).eval()(
        _priority(n_agents=3), taps=TapSet([tap], strict=True))
    record = next(r for r in tap.records
                  if r.location == "comm/r0/selected_count")
    assert record.shape == (3, 3)


def test_taps_none_does_not_change_the_result() -> None:
    selector = TopKSelector(k=4).eval()
    priority = _priority()
    assert torch.equal(selector(priority),
                       selector(priority, taps=TapSet([StatsTap()], strict=True)))
