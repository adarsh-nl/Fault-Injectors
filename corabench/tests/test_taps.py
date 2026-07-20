"""CoRA's observation registry and model-level tap invariance.

The tap MECHANISM is paper-agnostic and tested in ``cpbench/tests/test_taps.py``.
What is CoRA-specific is the registry of its 52 named locations, and that
CoRA's own forward pass is unchanged by observation.
"""

import torch

from cpbench.observation import StatsTap, TapSet
from corabench.observation import all_locations, validate_location


def test_forward_identical_with_and_without_taps(tiny_model, batch):
    """The measurement-plane invariant on the CoRA model."""
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
