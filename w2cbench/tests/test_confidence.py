"""
Tests for the spatial confidence generator and the Gaussian smoother.

This is the first stage that is actually Where2comm, and the tensor it
produces is load-bearing three times over: it decides what an agent transmits,
how strongly its message is weighted in fusion, and -- as ``1 - C`` -- where it
asks others to look. A reduction that is subtly wrong here would not fail
anywhere; it would just quietly change the paper's contribution into a
different algorithm.

So the reduction is pinned against the released code's exact expression, the
smoother's normalisation is pinned against the discrepancy it corrects (A16),
and the fault-relevant property -- a degraded agent reports lower confidence,
which is what makes bandwidth fall under a sensor fault -- is asserted rather
than assumed.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from cpbench.models import DetectionHead
from cpbench.observation import StatsTap, TapSet
from w2cbench.comm import GaussianSmoother, gaussian_kernel_2d
from w2cbench.models import SpatialConfidenceGenerator
from w2cbench.observation import validate_location


def _generator(smoothing: bool = False, **kwargs) -> SpatialConfidenceGenerator:
    smoother = GaussianSmoother(k_size=3, sigma=1.0) if smoothing else None
    return SpatialConfidenceGenerator(in_channels=16, smoother=smoother,
                                      **kwargs).eval()


# --------------------------------------------------------- the reduction --

def test_confidence_matches_the_released_reduction_exactly() -> None:
    """The released expression is

        ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(
            dim=1, keepdim=True)

    pinned here against a real head so a refactor cannot drift the semantics
    (A2)."""
    gen = _generator()
    features = torch.randn(3, 16, 8, 8)
    with torch.no_grad():
        out = gen(features)
        expected = out["cls"].sigmoid().max(dim=1, keepdim=True).values
    assert torch.equal(out["confidence"], expected)


def test_max_not_mean_over_anchors() -> None:
    """Max asks 'can this agent perceive SOMETHING here'. A mean is diluted by
    the anchors that found nothing, so a cell holding one clearly-seen vehicle
    would score below an ambiguous cell where every anchor is mildly unsure --
    and the clearly-seen vehicle is the one worth transmitting."""
    gen = _generator()
    with torch.no_grad():
        out = gen(torch.randn(2, 16, 8, 8))
    objectness = out["cls"].sigmoid()
    assert torch.equal(out["confidence"],
                       objectness.max(dim=1, keepdim=True).values)
    assert not torch.allclose(out["confidence"],
                              objectness.mean(dim=1, keepdim=True))


def test_confidence_is_a_probability_map() -> None:
    gen = _generator(smoothing=True)
    with torch.no_grad():
        confidence = gen(torch.randn(4, 16, 8, 8))["confidence"]
    assert confidence.shape == (4, 1, 8, 8)
    assert bool((confidence >= 0).all() and (confidence <= 1).all())


def test_shapes_survive_multiple_classes() -> None:
    """With n_cls > 1 the head's channel axis is A*n_cls and the max runs over
    all of it -- 'strongest evidence of any class at any anchor', which is
    still the right question."""
    gen = _generator(num_anchors=2, num_classes=3)
    with torch.no_grad():
        out = gen(torch.randn(2, 16, 8, 8))
    assert out["cls"].shape == (2, 6, 8, 8)
    assert out["confidence"].shape == (2, 1, 8, 8)


def test_regression_map_is_returned_even_though_unused_here() -> None:
    """At k=0 it is the released `rm_single`, which A11 supervises; the loss
    needs it and nothing else produces it."""
    gen = _generator()
    with torch.no_grad():
        out = gen(torch.randn(2, 16, 8, 8))
    assert out["reg"].shape == (2, 14, 8, 8)


# ----------------------------------------------------------- the smoother --

def test_kernel_is_normalised_and_symmetric() -> None:
    kernel = gaussian_kernel_2d(5, 1.5)
    assert torch.isclose(kernel.sum(), torch.tensor(1.0))
    assert torch.allclose(kernel, kernel.flip(0))
    assert torch.allclose(kernel, kernel.flip(1))
    assert kernel[2, 2] == kernel.max()


def test_released_kernel_attenuates_the_map_by_about_22_percent() -> None:
    """A16: the released filter is built from the continuous density and never
    renormalised, so it does not only smooth -- it scales the whole confidence
    map down before the threshold is applied, silently turning a configured
    0.01 into an effective ~0.0128. Pinned so the size of the discrepancy is a
    fact in the suite rather than a claim in a docstring."""
    raw = gaussian_kernel_2d(3, 1.0, normalize=False)
    assert 0.77 < float(raw.sum()) < 0.79

    flat = torch.full((1, 1, 8, 8), 1.0)
    released = GaussianSmoother(3, 1.0, normalize=False)
    corrected = GaussianSmoother(3, 1.0, normalize=True)
    centre = (slice(None), slice(None), slice(3, 5), slice(3, 5))
    assert float(released(flat)[centre].mean()) == pytest.approx(
        float(raw.sum()), abs=1e-5)
    assert float(corrected(flat)[centre].mean()) == pytest.approx(1.0, abs=1e-5)


def test_normalised_smoothing_preserves_a_uniform_map() -> None:
    """What 'smoothing' is supposed to mean: no signal, no change."""
    smoother = GaussianSmoother(3, 1.0)
    flat = torch.full((2, 1, 8, 8), 0.4)
    out = smoother(flat)
    assert torch.allclose(out[:, :, 2:6, 2:6], torch.tensor(0.4), atol=1e-6)


def test_isolated_spike_is_attenuated_more_than_a_solid_block() -> None:
    """The behaviour selection depends on: a detector artefact with no
    neighbourhood support should fall below the threshold while a genuine
    object footprint, several cells across, survives."""
    smoother = GaussianSmoother(3, 1.0)
    spike = torch.zeros(1, 1, 9, 9)
    spike[0, 0, 4, 4] = 1.0
    block = torch.zeros(1, 1, 9, 9)
    block[0, 0, 3:6, 3:6] = 1.0
    assert float(smoother(spike)[0, 0, 4, 4]) < float(smoother(block)[0, 0, 4, 4])


def test_k_size_one_is_a_free_identity() -> None:
    """'Smoothing disabled' should be a genuinely free code path, not a
    convolution that happens to change nothing."""
    smoother = GaussianSmoother(k_size=1)
    x = torch.randn(2, 1, 6, 6)
    assert smoother(x) is x


def test_zero_padding_pulls_the_border_down_and_replicate_does_not() -> None:
    """A real behavioural bias, not a rounding detail: the map edge is
    systematically less likely to be selected, so a pose error that shifts
    objects toward the boundary is penalised twice. The default matches the
    released code; the alternative is one config key away."""
    flat = torch.full((1, 1, 8, 8), 1.0)
    zeros = GaussianSmoother(3, 1.0, padding_mode="zeros")(flat)
    replicate = GaussianSmoother(3, 1.0, padding_mode="replicate")(flat)
    assert float(zeros[0, 0, 0, 0]) < 1.0
    assert float(replicate[0, 0, 0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_smoother_has_no_trainable_parameters() -> None:
    """The kernel is a buffer, so 'this filter is fixed' is structural rather
    than a requires_grad flag someone has to remember to set."""
    smoother = GaussianSmoother(3, 1.0)
    assert list(smoother.parameters()) == []
    assert "kernel" in dict(smoother.named_buffers())


def test_invalid_kernel_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        gaussian_kernel_2d(4, 1.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        gaussian_kernel_2d(3, 0.0)


# -------------------------------------------------------- the shared head --

def test_there_is_exactly_one_detection_head_in_the_model() -> None:
    """A2: the generator reuses the decoder's parameters. If these ever became
    two heads, the confidence map would stop reflecting what the model
    actually predicts, and selection would be driven by a second opinion the
    paper does not have."""
    gen = _generator()
    heads = [m for m in gen.modules() if isinstance(m, DetectionHead)]
    assert len(heads) == 1
    weights = [k for k in gen.state_dict() if "cls_head.weight" in k]
    assert len(weights) == 1


def test_decode_uses_the_same_parameters_as_the_generator() -> None:
    """The coupling that matters for fault analysis: a gradient or a fault
    that changes detection also changes what gets transmitted."""
    gen = _generator()
    features = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        assert torch.equal(gen(features)["cls"], gen.decode(features)["cls"])


# ------------------------------------------------- registry vs. reality --

def _emitted(gen: SpatialConfidenceGenerator, round_index: int = 0) -> Counter:
    tap = StatsTap()
    with torch.no_grad():
        gen(torch.randn(2, 16, 8, 8), taps=TapSet([tap], strict=True),
            round_index=round_index)
    return Counter(record.location for record in tap.records)


def test_generator_emits_exactly_the_registered_locations() -> None:
    counts = _emitted(_generator(smoothing=True))
    assert set(counts) == {
        "confidence/r0/cls_logits", "confidence/r0/reg_map",
        "confidence/r0/sigmoid", "confidence/r0/map", "confidence/r0/smoothed"}
    assert set(counts.values()) == {1}
    for name in counts:
        validate_location(name)


def test_smoothed_is_absent_when_smoothing_is_disabled() -> None:
    """A declared location that nothing emits is a promise the package does
    not keep -- but only when the feature is on. With A9 off, its absence is
    the correct behaviour and the taps config should show it."""
    counts = _emitted(_generator(smoothing=False))
    assert "confidence/r0/smoothed" not in counts
    assert "confidence/r0/map" in counts


def test_locations_carry_the_round_index() -> None:
    """Multi-round runs need per-round observation, and the registry stores
    these as {k} templates for exactly this reason."""
    counts = _emitted(_generator(smoothing=True), round_index=2)
    assert "confidence/r2/map" in counts
    assert "confidence/r0/map" not in counts
    validate_location("confidence/r2/map")


def test_registered_module_matches_the_emitting_class() -> None:
    tap = StatsTap()
    gen = _generator(smoothing=True)
    with torch.no_grad():
        gen(torch.randn(2, 16, 8, 8), taps=TapSet([tap], strict=True))
    for record in tap.records:
        declared = validate_location(record.location)
        assert record.module in declared.emitters(), (
            f"{record.location}: registry says {declared.module}, "
            f"emitted by {record.module}")


def test_head_taps_are_suppressed_during_generation() -> None:
    """A pre-fusion invocation is a different observation point from the final
    decode: one drives selection, the other is the model's answer. Letting both
    land on head/cls_logits would merge two semantically different tensors into
    one location, and the layer-wise clean-vs-faulted join would average them
    together."""
    generating = _emitted(_generator())
    assert not [name for name in generating if name.startswith("head/")]

    tap = StatsTap()
    gen = _generator()
    with torch.no_grad():
        gen.decode(torch.randn(1, 16, 8, 8), taps=TapSet([tap], strict=True))
    decoding = {record.location for record in tap.records}
    assert decoding == {"head/cls_logits", "head/reg_map", "head/cls_sigmoid"}


def test_taps_none_does_not_change_the_result() -> None:
    gen = _generator(smoothing=True)
    features = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        without = gen(features)["confidence"]
        with_taps = gen(features, taps=TapSet([StatsTap()], strict=True))["confidence"]
    assert torch.equal(without, with_taps)


# --------------------------------------------------------- fault relevance --

def test_an_agent_that_saw_nothing_reports_low_confidence() -> None:
    """The first link of the causal chain the package exists to demonstrate.
    A sensor fault that empties an agent's point cloud produces an all-zero
    feature map; the confidence head's focal-loss prior (bias -4.59, sigmoid
    ~0.01) then reports near-zero confidence, so the agent selects almost no
    cells and its transmitted volume collapses. If this ever stopped holding,
    'the fault lowered bandwidth' would be an artefact rather than an effect.
    """
    gen = _generator()
    blind = torch.zeros(1, 16, 8, 8)
    seeing = torch.randn(1, 16, 8, 8) * 3.0
    with torch.no_grad():
        blind_confidence = gen(blind)["confidence"]
        seeing_confidence = gen(seeing)["confidence"]
    assert float(blind_confidence.max()) < 0.05
    assert float(seeing_confidence.max()) > float(blind_confidence.max())
