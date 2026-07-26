"""Loss functions: finiteness, positivity, sane orderings, gradients."""

import warnings

import pytest
import torch

from corabench.training.losses import (CoRALoss, focal_loss_logits,
                                       focal_loss_prob, smooth_l1_reg_loss)


def _require_cuda(what: str) -> None:
    """Skip LOUDLY. A half-precision test that silently skips on a CPU box
    reads as a pass in the summary line, which is exactly how job 547612's
    defect survived: the saturation is only reachable in fp16, so a green
    CPU-only suite says nothing about it."""
    if not torch.cuda.is_available():
        message = (f"NOT VERIFIED: {what} requires CUDA for fp16 autocast; "
                   f"this defect is UNREACHABLE on CPU and therefore "
                   f"UNTESTED in this run. Run on a GPU node.")
        warnings.warn(message, UserWarning)
        pytest.skip(message)


def _targets(h=6, w=6, a=2, positives=((1, 2, 0),)):
    cls_t = torch.zeros(1, h, w, a)
    reg_t = torch.zeros(1, h, w, a, 7)
    for (i, j, k) in positives:
        cls_t[0, i, j, k] = 1.0
        reg_t[0, i, j, k] = 0.5
    return cls_t, reg_t


def test_focal_loss_prefers_correct_predictions():
    cls_t, _ = _targets()
    good = torch.full((1, 2, 6, 6), 0.02)
    good[0, 0, 1, 2] = 0.98
    bad = torch.full((1, 2, 6, 6), 0.98)
    assert focal_loss_prob(good, cls_t) < focal_loss_prob(bad, cls_t)


def test_focal_loss_ignores_ignore_label():
    cls_t, _ = _targets()
    cls_t[0, 3, 3, 1] = -1.0
    base = focal_loss_prob(torch.full((1, 2, 6, 6), 0.5), cls_t)
    # perturbing the ignored cell's prediction must not change the loss
    pred = torch.full((1, 2, 6, 6), 0.5)
    pred[0, 1, 3, 3] = 0.99
    assert torch.allclose(base, focal_loss_prob(pred, cls_t))


def test_reg_loss_only_on_positives():
    cls_t, reg_t = _targets()
    reg_map = torch.zeros(1, 14, 6, 6)
    loss0 = smooth_l1_reg_loss(reg_map, reg_t, cls_t)
    assert loss0 > 0                       # positive anchor target is 0.5
    exact = reg_t.permute(0, 3, 4, 1, 2).reshape(1, 14, 6, 6)
    assert smooth_l1_reg_loss(exact, reg_t, cls_t) < loss0


def test_cora_loss_composition(tiny_model, batch):
    tiny_model.train()
    out = tiny_model(batch, return_teacher=True)
    losses = CoRALoss()(out, batch)
    assert set(losses) == {"total", "cls", "reg", "align", "pac"}
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------- job 547612: saturated logits --

@pytest.mark.parametrize("logit_value", [20.0, -20.0, 205.5])
def test_focal_loss_is_finite_on_saturated_logits_under_fp16_autocast(
        logit_value):
    """(a) The defect that killed job 547612.

    In fp16 the spacing below 1.0 is 2^-11 = 4.9e-4, so `clamp(p, 1e-6,
    1 - 1e-6)` has an upper bound of exactly 1.0 -- the guard does not exist
    where it is needed. `log(1 - p)` is then `log(0) = -inf`, and for a
    NEGATIVE anchor `p_t = 1 - p = 0` makes the focal factor `(1-p_t)^gamma`
    exactly 1, so nothing damps it.

    205.5 is the real `head/cls_logits` amax observed in job 547612; sigmoid
    saturates to exactly 1.0 above a logit of roughly 8.3 in half precision,
    so all three values reach the same failure.

    Targets are all-negative on purpose: it is the `log(1 - p)` term, not
    `log(p)`, that goes non-finite.
    """
    _require_cuda("saturated-logit focal loss")
    cls_t = torch.zeros(1, 6, 6, 2, device="cuda")
    logits = torch.full((1, 2, 6, 6), logit_value, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        prob = torch.sigmoid(logits)
        assert not torch.isfinite(torch.log(1 - prob)).all() or abs(
            logit_value) < 8.3, "fp16 sigmoid did not saturate; test is moot"
        loss_prob = focal_loss_prob(prob, cls_t)
        loss_logits = focal_loss_logits(logits, cls_t)
    assert torch.isfinite(loss_prob), (
        f"probability-space focal loss is {loss_prob} at logit {logit_value}; "
        f"the float32 island is not holding")
    assert torch.isfinite(loss_logits), (
        f"logit-space focal loss is {loss_logits} at logit {logit_value}")


def test_uncertainty_network_still_receives_gradient(tiny_model, batch):
    """(c) The way the naive repair breaks things.

    Rewriting the LC/PAC branches in logit space by taking `out["lc"]["cls"]`
    directly would drop the `sigmoid(-U)` factor from the objective -- the
    only gradient path into the uncertainty network. Nothing else would fail:
    the loss would still be finite, still fall, and AP would still look
    plausible, with U untrained and the recalibration arbitrary.

    This asserts the path exists, so the fix for the saturation defect cannot
    quietly remove it.
    """
    tiny_model.train()
    tiny_model.zero_grad(set_to_none=True)
    out = tiny_model(batch, return_teacher=True)
    CoRALoss()(out, batch)["total"].backward()

    uncertainty = {n: p for n, p in tiny_model.named_parameters()
                   if "uncertainty" in n or ".u_" in n or "unc" in n}
    assert uncertainty, (
        "no uncertainty parameters found by name; if AdaptiveFusion's naming "
        "changed, update this test rather than deleting it -- it guards the "
        "only gradient path into U")
    got = {n: p for n, p in uncertainty.items()
           if p.grad is not None and p.grad.abs().sum() > 0}
    assert got, (
        f"the uncertainty network received NO gradient. Its only path is the "
        f"sigmoid(-U) factor in the recalibrated score the LC/PAC focal "
        f"losses consume (assumption A4). Checked: {sorted(uncertainty)}")
