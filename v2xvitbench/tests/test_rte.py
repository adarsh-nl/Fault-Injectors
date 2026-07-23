"""
Tests for the prior encoder and the delay-aware positional encoding (DPE).

The DPE is one of the two mechanisms this benchmark exists to stress: the
``delay_encoding`` fault feeds it a delay that disagrees with the features'
actual staleness. These tests pin the properties that make that experiment
interpretable -- the encoding must actually depend on the delay (or the fault
would be a no-op), be spatially uniform (or it would double as a spatial
perturbation), and clamp rather than crash on out-of-range delays.
"""

from __future__ import annotations

import pytest
import torch

from v2xvitbench.fusion.prior import DelayPositionalEncoding, PriorEncoder


class _Recorder:
    def __init__(self) -> None:
        self.seen = {}

    def observe(self, tensor, *, module: str, location: str,
                **context) -> None:
        self.seen[location] = tensor


# ----------------------------------------------------------- prior encoder --

def test_prior_stacks_and_normalises() -> None:
    prior = PriorEncoder()(velocity=torch.tensor([[30.0, 0.0]]),
                           time_delay=torch.tensor([[2, 0]]),
                           infra=torch.tensor([[1, 0]]))
    assert prior.shape == (1, 2, 3)
    assert prior[0, 0].tolist() == [1.0, 2.0, 1.0]
    assert prior[0, 1].tolist() == [0.0, 0.0, 0.0]


def test_prior_emits_input_location() -> None:
    recorder = _Recorder()
    PriorEncoder()(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2),
                   taps=recorder)
    assert set(recorder.seen) == {"input/prior_encoding"}


# -------------------------------------------------------------------- DPE --

@pytest.fixture
def rte() -> DelayPositionalEncoding:
    return DelayPositionalEncoding(dim=16, max_delay=10, ratio=2)


def test_zero_delay_still_adds_an_embedding(rte) -> None:
    """dt=0 reads table row 0, which is a defined encoding ("fresh"), not an
    identity: the model must be able to tell "fresh" from "no information"."""
    x = torch.zeros(1, 1, 16, 4, 4)
    out = rte(x, torch.tensor([[0]]))
    assert not torch.equal(out, x)


def test_different_delays_give_different_features(rte) -> None:
    """The fault plane's premise: changing the REPORTED delay must change the
    model's input, or delay_encoding faults would be no-ops."""
    x = torch.zeros(1, 2, 16, 4, 4)
    out = rte(x, torch.tensor([[0, 3]]))
    assert not torch.allclose(out[0, 0], out[0, 1])

    out_a = rte(x, torch.tensor([[3, 3]]))
    assert torch.allclose(out_a[0, 0], out_a[0, 1])  # same dt, same encoding


def test_embedding_is_spatially_uniform(rte) -> None:
    """Delay is a property of the agent, not of any cell; a non-uniform
    embedding would smuggle a spatial perturbation into a temporal signal."""
    x = torch.zeros(1, 1, 16, 4, 4)
    out = rte(x, torch.tensor([[5]]))
    flat = out[0, 0].reshape(16, -1)
    assert torch.allclose(flat, flat[:, :1].expand_as(flat))


def test_delay_clamps_at_max(rte) -> None:
    """A latency fault can push dt past the table; clamping to the last row
    (maximally stale) is the conservative reading, crashing is not an option
    mid-benchmark."""
    x = torch.zeros(1, 2, 16, 4, 4)
    out = rte(x, torch.tensor([[10, 99]]))
    assert torch.allclose(out[0, 0], out[0, 1])


def test_ratio_spreads_the_table(rte) -> None:
    """RTE_ratio=2 means dt=1 reads row 2: adjacent delays sit further apart
    in the sinusoid, which is the reference's discriminability trick."""
    x = torch.zeros(1, 1, 16, 1, 1)
    via_ratio = rte(x, torch.tensor([[1]]))
    direct = rte.linear(rte.table[2]).reshape(1, 1, 16, 1, 1)
    assert torch.allclose(via_ratio, direct, atol=1e-6)


def test_shape_mismatch_raises(rte) -> None:
    with pytest.raises(ValueError, match="on \\(batch, agent\\)"):
        rte(torch.zeros(1, 3, 16, 4, 4), torch.zeros(1, 2))


def test_rte_emits_embedding_and_output(rte) -> None:
    recorder = _Recorder()
    rte(torch.zeros(1, 1, 16, 2, 2), torch.tensor([[1]]), taps=recorder)
    assert set(recorder.seen) == {"rte/embedding", "rte/output"}
    assert recorder.seen["rte/embedding"].shape == (1, 1, 16)


def test_only_the_linear_readout_is_trainable(rte) -> None:
    """The sinusoid table is a buffer (reference behaviour); training must
    not be able to erase the delay structure itself."""
    trainable = {name for name, p in rte.named_parameters() if p.requires_grad}
    assert trainable == {"linear.weight", "linear.bias"}
