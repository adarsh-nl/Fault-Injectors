"""
Tests for the shared image backbone.

The module moved out of ``cobevtbench`` when a second package needed it, so
what matters here is that the move is genuinely additive: the old import path
still resolves, both packages get the *same class object* rather than two
copies that could drift, and the tap names every registry declares are still
the ones emitted.

Nothing here downloads weights. ``pretrained=False`` throughout, because a test
suite that needs network access is a test suite that fails on a compute node.
"""

from __future__ import annotations

import pytest
import torch

from cpbench.models.image import IMAGENET_MEAN, ResnetEncoder


# The "the old cobevtbench path still resolves" assertion lives in
# cobevtbench/tests/test_backbone_reexport.py, not here: cpbench must not
# import a paper package even in a test, and the layering suite enforces it.


def test_cpbench_models_exposes_it_lazily() -> None:
    """Exported through ``__getattr__`` so importing ``cpbench.models`` does
    not pull in torchvision for the LiDAR-only packages."""
    import cpbench.models as models
    assert models.ResnetEncoder is ResnetEncoder
    with pytest.raises(AttributeError, match="has no attribute"):
        models.NotAThing


def test_the_pyramid_is_returned_fine_to_coarse() -> None:
    encoder = ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[1, 2])
    features = encoder(torch.rand(1, 4, 3, 64, 64))
    assert [tuple(f.shape) for f in features] == [(1, 4, 128, 8, 8),
                                                  (1, 4, 256, 4, 4)]
    assert encoder.out_channels == [128, 256]


def test_layers_beyond_the_deepest_pick_are_not_built() -> None:
    """They would cost forward time for a result nobody reads and leave
    parameters that never receive a gradient -- which inflates the model size
    a paper reports."""
    shallow = ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[0])
    deep = ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[3])
    assert len(shallow.layers) == 1 and len(deep.layers) == 4
    assert sum(p.numel() for p in shallow.parameters()) < \
        sum(p.numel() for p in deep.parameters())


def test_channels_last_bytes_and_channels_first_floats_both_work() -> None:
    """Dispatching on which axis is size 3 rather than on dtype: a dataset that
    converted to float but kept channels-last is the likely mistake, and
    silently treating H=3 as channels would train to noise."""
    encoder = ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[1])
    as_bytes = encoder(torch.randint(0, 255, (1, 2, 32, 32, 3)).float())
    as_floats = encoder(torch.rand(1, 2, 3, 32, 32))
    assert as_bytes[0].shape == as_floats[0].shape


def test_an_ambiguous_input_shape_is_rejected() -> None:
    encoder = ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[1])
    with pytest.raises(ValueError, match="size-3 channel axis"):
        encoder(torch.rand(1, 2, 5, 32, 32))
    with pytest.raises(ValueError, match=r"expected \(B, M, H, W, 3\)"):
        encoder(torch.rand(2, 3, 32, 32))


def test_normalisation_uses_the_imagenet_statistics() -> None:
    """Required, not decorative: the pretrained weights were fitted under this
    normalisation and produce degraded features without it."""
    encoder = ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[1])
    normalised = encoder._to_nchw(torch.full((1, 1, 3, 8, 8), 0.485))
    assert float(normalised[0, 0].abs().max()) < 1e-6      # channel 0 -> zero
    assert IMAGENET_MEAN[0] == pytest.approx(0.485)


def test_bad_configuration_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown backbone"):
        ResnetEncoder(arch="resnet101", pretrained=False)
    with pytest.raises(ValueError, match="id_pick must select"):
        ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[4])


def test_the_declared_tap_names_are_the_ones_emitted() -> None:
    """Both paper registries declare these names, deliberately identically, so
    a cross-paper comparison of image features is a straight join on
    ``location``. If the module drifted, that join would silently return
    nothing."""
    from cpbench.observation import StatsTap, TapSet

    tap = StatsTap()
    encoder = ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[1, 2])
    encoder(torch.rand(1, 2, 3, 32, 32), taps=TapSet([tap], strict=True))
    emitted = {record.location for record in tap.records}
    assert emitted == {"backbone/normalised", "backbone/feat_s0",
                       "backbone/feat_s1"}
    assert {r.module for r in tap.records} == {"ResnetEncoder"}
